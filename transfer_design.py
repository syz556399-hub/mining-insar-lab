"""Fit pixel-level design statistics from development labels; generate synthetic scenes.

No real image, mask texture or validation label is copied into a training sample.
Physical parameters remain uncalibrated; achieved label sizes are recorded.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from annotation_audit import decode_annotation_mask, read_annotations, sha256
from data_io import phase_rgb, png
from mine_model import simulate
from provenance import fingerprint, runtime_versions
from validate_dataset import audit_splits, rows_at


def fit(labels, output, size=256):
    if output.exists():
        raise ValueError("Profile already exists")
    _, _, _, records, _ = read_annotations(labels)
    areas = []
    for record in records:
        doc = json.loads((labels / record["file"]).read_text())
        mask = decode_annotation_mask(doc)
        for y in range(0, mask.shape[0], size):
            for x in range(0, mask.shape[1], size):
                crop = mask[y : y + size, x : x + size]
                if crop.shape == (size, size):
                    areas.append(int(crop.sum()))
    positive = [a for a in areas if a > 0]
    if not positive:
        raise ValueError("No positive development blocks")
    profile = dict(
        schema="pixel-design-1",
        size=size,
        positive_probability=len(positive) / len(areas),
        positive_area_samples_px=positive,
        blocks=len(areas),
        positive_blocks=len(positive),
        source_annotations=[dict(file=r["file"], sha256=r["annotation_sha256"]) for r in records],
        manifest_sha256=sha256(labels / "manifest.json"),
        interpretation="development-label pixel statistics only; no physical scale or visibility calibration",
        label_proxy="clean deformation phase support at fixed pi/2; not claimed equal to manual visible boundary",
    )
    profile["profile_sha256"] = fingerprint(profile)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2))
    return profile


def matched_scene(request, profile, index):
    request = json.loads(json.dumps(request))
    if request["settings"]["size"] != profile["size"]:
        raise ValueError("Pixel design size mismatch")
    rng = np.random.default_rng(np.random.SeedSequence([request["settings"]["seed"], index, 3901]))
    positive = rng.random() < profile["positive_probability"]
    for j, face in enumerate(request["faces"]):
        face["enabled"] = bool(positive and (j == 0 or rng.random() < 0.25))
    desired = int(rng.choice(profile["positive_area_samples_px"])) if positive else 0
    arrays, metadata = simulate(request)
    initial_area = int(arrays["fringe_mask"].sum())
    # Extent changes pixel coverage without stretching exported phase or truth images.
    for _ in range(2 if positive else 0):
        area = int(arrays["fringe_mask"].sum())
        if not area:
            for face in request["faces"]:
                if face["enabled"]:
                    face["thickness_m"] = min(8.0, face["thickness_m"] * 2)
        else:
            old = request["settings"]["extent_m"]
            new = float(np.clip(old * np.sqrt(area / desired), 400, 6000))
            request["settings"]["extent_m"] = new
            for face in request["faces"]:
                face["east_m"] *= new / old
                face["north_m"] *= new / old
        arrays, metadata = simulate(request)
    metadata["pixel_design"] = dict(
        profile_sha256=profile["profile_sha256"],
        positive_requested=bool(positive),
        requested_support_area_px=desired,
        initial_support_area_px=initial_area,
        achieved_support_area_px=int(arrays["fringe_mask"].sum()),
        adaptation="at most two bounded physical-domain extent/amplitude adjustments; no image warping or label painting",
    )
    return arrays, metadata


def export_matched(source, profile_file, output):
    if output.exists():
        raise ValueError("Output already exists")
    profile = json.loads(profile_file.read_text())
    claimed = profile.pop("profile_sha256")
    if fingerprint(profile) != claimed:
        raise ValueError("Profile checksum mismatch")
    profile["profile_sha256"] = claimed
    if not 0 < profile["positive_probability"] <= 1 or any(
        a < 1 or a > profile["size"] ** 2 for a in profile["positive_area_samples_px"]
    ):
        raise ValueError("Invalid pixel profile")
    records = rows_at(source)
    output.mkdir(parents=True)
    rows = []
    sizes = []
    for i, row in enumerate(records):
        saved = json.loads((source / row["metadata"]).read_text())
        arrays, metadata = matched_scene(saved["request"], profile, i)
        category = (
            "negative"
            if not any(f["enabled"] for f in metadata["request"]["faces"])
            else (
                "single"
                if sum(f["enabled"] for f in metadata["request"]["faces"]) == 1
                else row["category"]
            )
        )
        if category == "negative" and any(f["enabled"] for f in metadata["request"]["faces"]):
            category = "overlapping"
        new = dict(row, group_id=metadata["group_id"], category=category)
        metadata.update(
            scene_id=row["scene_id"],
            split=row["split"],
            category=category,
            sampling_profile="pixel_matched",
        )
        for key in ["image", "mask", "arrays", "metadata"]:
            (output / row[key]).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / row["arrays"], **arrays)
        (output / row["image"]).write_bytes(png(phase_rgb(arrays["wrapped_phase_rad"])))
        (output / row["mask"]).write_bytes(png(arrays["mask"] * 255))
        (output / row["metadata"]).write_text(json.dumps(metadata, indent=2))
        sizes.append(dict(scene_id=row["scene_id"], split=row["split"], **metadata["pixel_design"]))
        rows.append(new)
        if (i + 1) % 20 == 0:
            print(json.dumps(dict(generated=i + 1, total=len(records))), flush=True)
    with (output / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    info = json.loads((source / "dataset.json").read_text())
    info.update(
        software_versions=runtime_versions(),
        sampling_profile="pixel_matched",
        pixel_design=profile,
        category_note="realized scenario categories; split assignments inherited from paired baseline indices",
    )
    (output / "dataset.json").write_text(json.dumps(info, indent=2))
    (output / "design_achieved.json").write_text(json.dumps(sizes, indent=2))
    audit_splits([output])
    return {
        "count": len(rows),
        "positive_fraction": float(np.mean([r["achieved_support_area_px"] > 0 for r in sizes])),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--labels", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("generate")
    g.add_argument("--baseline", type=Path, required=True)
    g.add_argument("--profile", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = (
        fit(a.labels, a.out) if a.command == "fit" else export_matched(a.baseline, a.profile, a.out)
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in [
                    "blocks",
                    "positive_blocks",
                    "positive_probability",
                    "count",
                    "positive_fraction",
                ]
            }
        )
    )
