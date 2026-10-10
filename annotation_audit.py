"""Read-only audit of tiled LabelMe masks; no label cleaning or model fitting."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode_annotation_mask(doc, target_label="1"):
    """Decode exact per-file mask pixels without filling, filtering or merging objects."""
    right, bottom = doc["imageWidth"], doc["imageHeight"]
    x = y = 0
    tile = np.zeros((bottom - y, right - x), dtype=bool)
    for shape in doc["shapes"]:
        if str(shape["label"]) != target_label or shape["shape_type"] != "mask":
            raise ValueError(
                "This importer accepts only the explicitly selected LabelMe mask class"
            )
        with Image.open(io.BytesIO(base64.b64decode(shape["mask"], validate=True))) as image:
            mask = np.asarray(image)
        if mask.ndim != 2 or not np.isin(mask, [0, 1, 255]).all():
            raise ValueError("Expected a binary embedded LabelMe mask")
        points = np.asarray(shape["points"], dtype=float)
        if points.shape != (2, 2) or not np.isfinite(points).all():
            raise ValueError("Invalid mask bounds")
        left, top = (int(round(n)) for n in points[0])
        h, w = mask.shape
        if not (0 <= left and 0 <= top and left + w <= tile.shape[1] and top + h <= tile.shape[0]):
            raise ValueError("Embedded mask extends beyond its tile")
        tile[top : top + h, left : left + w] |= mask > 0
    return tile


def read_annotations(root, target_label="1"):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    width, height = manifest["width"], manifest["height"]
    if any(type(n) is not int or n <= 0 for n in (width, height)):
        raise ValueError("Invalid scene dimensions")
    boxes = {}
    for tile in manifest["tiles"]:
        name, box = tile["file"], tile["bbox"]
        if Path(name).name != name or name in boxes:
            raise ValueError("Tile names must be unique basenames")
        if len(box) != 4 or any(type(n) is not int for n in box):
            raise ValueError("Invalid tile coordinates")
        x, y, r, b = box
        if not (0 <= x < r <= width and 0 <= y < b <= height):
            raise ValueError("Tile extends beyond the original scene")
        boxes[name] = box
    truth = np.zeros((height, width), dtype=bool)
    coverage = np.zeros_like(truth)
    conflict = np.zeros_like(truth)
    records = []
    for path in sorted(root.glob("*.json")):
        if path.name == "manifest.json":
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        name = str(doc["imagePath"]).replace("\\", "/").rsplit("/", 1)[-1]
        if name not in boxes or Path(name).stem != path.stem:
            raise ValueError(f"Unmapped annotation: {path.name}")
        x, y, right, bottom = boxes[name]
        if [doc["imageWidth"], doc["imageHeight"]] != [right - x, bottom - y]:
            raise ValueError(f"Annotation dimensions differ: {path.name}")
        tile = decode_annotation_mask(doc, target_label)
        region = np.s_[y:bottom, x:right]
        # Disagreement is a review flag, not evidence that either annotation is wrong.
        conflict[region] |= coverage[region] & (truth[region] != tile)
        truth[region] |= tile
        coverage[region] = True
        records.append(
            dict(
                file=path.name,
                image=name,
                bbox=boxes[name],
                shapes=len(doc["shapes"]),
                positive_pixels=int(tile.sum()),
                annotation_sha256=sha256(path),
            )
        )
    if not records:
        raise ValueError("No mapped annotation files")
    stored = root / "_full_scene/mask.bmp"
    if stored.is_file():
        with Image.open(stored) as image:
            if not np.array_equal(np.asarray(image) > 0, truth):
                raise ValueError("Decoded annotations differ from the stored full-scene mask")
    for row in records:
        x, y, r, b = row["bbox"]
        row["overlap_disagreement_pixels"] = int(conflict[y:b, x:r].sum())
    return truth, coverage, conflict, records, manifest


def connected_regions(mask):
    """8-connected regions as row runs; preserve even a single annotated pixel."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Expected a 2D binary mask")
    parent, runs = [], []

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    previous = []
    for y, row in enumerate(mask):
        changes = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        current, cursor = [], 0
        for left, right in zip(starts.tolist(), ends.tolist()):
            index = len(parent)
            parent.append(index)
            runs.append((y, left, right))
            while cursor < len(previous) and previous[cursor][1] < left:
                cursor += 1
            j = cursor
            while j < len(previous) and previous[j][0] <= right:
                parent[find(previous[j][2])] = find(index)
                j += 1
            current.append((left, right, index))
        previous = current
    groups = {}
    for index, (y, left, right) in enumerate(runs):
        key = find(index)
        group = groups.setdefault(key, dict(area_px=0, bbox=[left, y, right, y + 1], runs=[]))
        group["area_px"] += right - left
        box = group["bbox"]
        group["bbox"] = [min(box[0], left), min(box[1], y), max(box[2], right), y + 1]
        group["runs"].append((y, left, right))
    result = sorted(groups.values(), key=lambda value: (-value["area_px"], value["bbox"]))
    for index, group in enumerate(result):
        group["id"] = f"region_{index:05d}"
        group["equivalent_square_width_px"] = float(np.sqrt(group["area_px"]))
    return result


def region_summary(regions, scales=(1, 2, 4, 8)):
    areas = np.array([r["area_px"] for r in regions], dtype=float)
    return dict(
        count=len(regions),
        total_positive_pixels=int(areas.sum()),
        area_quantiles_px=dict(
            zip(
                ["min", "q25", "median", "q75", "max"],
                np.quantile(areas, [0, 0.25, 0.5, 0.75, 1]).tolist(),
            )
        )
        if len(areas)
        else {},
        sizes={
            name: int(condition.sum())
            for name, condition in [
                ("small_area_lt_256", areas < 256),
                ("medium_area_256_to_4095", (areas >= 256) & (areas < 4096)),
                ("large_area_ge_4096", areas >= 4096),
            ]
        },
        scale_risk={
            str(s): dict(
                ideal_area_below_one_input_pixel=int((areas / s**2 < 1).sum()),
                ideal_area_below_four_input_pixels=int((areas / s**2 < 4).sum()),
            )
            for s in scales
        },
        interpretation="Connected annotation regions, not independently verified mine instances. Area/scale^2 is an ideal size diagnostic, not measured information survival.",
    )


def select_review_tiles(records, count=20):
    """Fixed selection before predictions: four possible backgrounds + area quantiles."""
    if count < 1:
        raise ValueError("Review count must be positive")
    negative = [r for r in records if not r["positive_pixels"]]
    positive = sorted(
        (r for r in records if r["positive_pixels"]),
        key=lambda r: (r["positive_pixels"], r["file"]),
    )
    selected = negative[: min(4, count)]
    remaining = min(count - len(selected), len(positive))
    if remaining:
        selected += [positive[i] for i in np.linspace(0, len(positive) - 1, remaining, dtype=int)]
    selected += [r for r in negative if r not in selected][: max(0, count - len(selected))]
    return selected


def audit(root, source, output, count=20):
    if output.exists():
        raise ValueError("Output exists; choose a new audit directory")
    truth, scope, conflict, records, manifest = read_annotations(root)
    with Image.open(source) as original:
        if original.size != (manifest["width"], manifest["height"]):
            raise ValueError("Source and manifest dimensions differ")
        image = original.convert("RGB")
    chosen = select_review_tiles(records, count)
    # Verify coordinate alignment against the original tile images, before rendering.
    for row in chosen:
        with Image.open(root / row["image"]) as tile:
            if not np.array_equal(
                np.asarray(tile.convert("RGB")), np.asarray(image.crop(row["bbox"]))
            ):
                raise ValueError(f"Tile does not match source pixels: {row['image']}")
    regions = connected_regions(truth)
    report = dict(
        schema="label-audit-1",
        source_name=source.name,
        source_sha256=sha256(source),
        manifest_sha256=sha256(root / "manifest.json"),
        shape=list(truth.shape),
        annotation_files=records,
        saved_tile_scope_pixels=int(scope.sum()),
        overlap_disagreement_pixels=int(conflict.sum()),
        reviewed_background_confirmed=False,
        target_definition="complete mining subsidence fringe anomaly region; overlapping basins merged; confirmed small targets retained",
        potential_influence_zone="separate uncertain inference; not ground truth derived from BMP",
        central_uncertainty_policy="proposed: ignore unclear interior pixels until reviewed; do not infer coherence from RGB",
        regions=region_summary(regions),
        review_tiles=chosen,
        review_selection="before prediction; up to four zero-label tiles then positive-area quantiles; no model-success selection",
    )
    report["audit_id"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True)
    (output / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "regions.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "runs"} for r in regions], indent=2),
        encoding="utf-8",
    )
    from audit_report import write_audit_report

    write_audit_report(output, image, truth, conflict, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    report = audit(args.labels, args.source, args.out, args.count)
    print(
        json.dumps(
            {key: report[key] for key in ("audit_id", "overlap_disagreement_pixels", "regions")},
            ensure_ascii=False,
        )
    )
