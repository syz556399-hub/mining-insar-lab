"""Frozen range model diagnosis on original BMP and unchanged final LabelMe tiles.

BMP color codes are a display-angle proxy, never calibrated physical phase.
No phase-head output on BMP is reported as a physical measurement.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from annotation_audit import decode_annotation_mask, read_annotations, sha256
from mine_learning import MaskLogits, RangePhaseNet, encode, write_report
from phase_encoding import decode_display_palette
from segmentation import metrics_from_counts, segmentation_overlay, tile_starts, tiled_probability


def evaluate(root, source, labels):
    if (root / "real_results.json").exists() or (root / "real_protocol.json").exists():
        raise ValueError("Real diagnosis exists; preserve it")
    torch.set_num_threads(2)
    results = json.loads((root / "results.json").read_text())
    arm = results["preferred_by_validation"]
    checkpoint = root / (arm + ".pt")
    cp = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if cp["model_kind"] != "independent-range-phase-1":
        raise ValueError("Wrong model type")
    _, _, _, records, manifest = read_annotations(labels)
    with Image.open(source) as original:
        if original.size != (manifest["width"], manifest["height"]):
            raise ValueError("Source dimensions differ from label manifest")
        phase, visible, adapter = decode_display_palette(original)
        rgb = original.convert("RGB")
    for row in records:
        with Image.open(labels / row["image"]) as saved:
            if not np.array_equal(
                np.asarray(saved.convert("RGB")), np.asarray(rgb.crop(row["bbox"]))
            ):
                raise ValueError("Label images differ from original source pixels")
    protocol = dict(
        arm=arm,
        checkpoint_sha256=sha256(checkpoint),
        source_name=source.name,
        source_sha256=sha256(source),
        manifest_sha256=sha256(labels / "manifest.json"),
        annotations=records,
        input_adapter=adapter,
        threshold=0.5,
        inference="original pixels; 256 windows, stride192, Hann taper overlap mean; no component filter",
        evaluation="exact final per-file masks and original shape records; overlaps counted per file",
        status="previously used same-mine March development date, not independent test",
        scope="frozen model; no label/threshold changes, no training, no reported physical BMP phase",
    )
    (root / "real_protocol.json").write_text(json.dumps(protocol, indent=2))
    channels = encode(phase)
    channels[:, ~visible] = 0
    del phase, visible
    model = RangePhaseNet(cp["width"])
    model.load_state_dict(cp["state_dict"])
    total = len(tile_starts(channels.shape[1], 256, 192)) * len(
        tile_starts(channels.shape[2], 256, 192)
    )
    started = time.monotonic()

    def progress(done):
        if done % 100 == 0 or done == total:
            print(json.dumps(dict(completed=done, total=total)), flush=True)

    score = tiled_probability(
        MaskLogits(model),
        channels,
        torch.device("cpu"),
        tile_size=256,
        stride=192,
        batch_size=4,
        progress=progress,
    )
    np.save(root / "real_probability.npy", score)
    counts, tiles, coverage = np.zeros(4, np.int64), [], []
    for index, row in enumerate(records):
        doc = json.loads((labels / row["file"]).read_text())
        truth = decode_annotation_mask(doc)
        x, y, right, bottom = row["bbox"]
        local = score[y:bottom, x:right]
        prediction = local >= 0.5
        local_counts = [
            int((prediction & truth).sum()),
            int((prediction & ~truth).sum()),
            int((~prediction & truth).sum()),
            int((~prediction & ~truth).sum()),
        ]
        counts += local_counts
        objects = []
        for shape in doc["shapes"]:
            mask = decode_annotation_mask(dict(doc, shapes=[shape]))
            objects.append(float(prediction[mask].mean()) if mask.any() else 0.0)
        coverage.extend(objects)
        observed = rgb.crop(row["bbox"])
        ground_truth = np.repeat((truth[..., None] * 255).astype(np.uint8), 3, axis=2)
        board = Image.new("RGB", (768, 284), "#111c29")
        draw = ImageDraw.Draw(board)
        for j, (name, panel) in enumerate(
            (
                ("ORIGINAL BMP", observed),
                ("FINAL LABELME", Image.fromarray(ground_truth)),
                ("FROZEN PREDICTION", segmentation_overlay(observed, local)),
            )
        ):
            draw.text((j * 256 + 8, 8), name, fill="white")
            board.paste(panel.resize((256, 256)), (j * 256, 28))
        preview = f"real_{index:02d}.png"
        board.save(root / preview)
        tiles.append(
            dict(file=row["file"], counts=local_counts, object_coverage=objects, preview=preview)
        )
    metrics = metrics_from_counts(counts.tolist())
    metrics.update(
        annotation_objects=len(coverage),
        objects_any_overlap=sum(v > 0 for v in coverage),
        objects_half_covered=sum(v >= 0.5 for v in coverage),
    )
    unchanged = (
        sha256(source) == protocol["source_sha256"]
        and sha256(checkpoint) == protocol["checkpoint_sha256"]
        and sha256(labels / "manifest.json") == protocol["manifest_sha256"]
        and all(sha256(labels / r["file"]) == r["annotation_sha256"] for r in records)
    )
    if not unchanged:
        raise ValueError("Source/labels/checkpoint changed during diagnosis")
    result = dict(
        protocol=protocol,
        metrics=metrics,
        tiles=tiles,
        seconds=time.monotonic() - started,
        source_labels_checkpoint_unchanged=True,
    )
    (root / "real_results.json").write_text(json.dumps(result, indent=2))
    write_report(root)
    print(json.dumps(metrics), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.run, args.source, args.labels)
