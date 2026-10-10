"""Validate saved arrays and provenance; optionally audit multiple dataset splits."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from data_io import phase_rgb
from mine_model import ENGINE, simulate, wrap
from provenance import fingerprint, group_identity, runtime_versions
from task_labels import fringe_support


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows_at(root):
    with (Path(root) / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def audit_splits(roots):
    seen = {}
    repeats = 0
    for root in roots:
        for row in rows_at(root):
            group, split = row["group_id"], row["split"]
            require(split in ("train", "val", "test"), f"Unknown split: {split}")
            if group in seen:
                require(
                    seen[group] == split,
                    f"Scene group {group} occurs across {seen[group]} and {split}",
                )
                repeats += 1
            seen[group] = split
    return {
        "unique_groups": len(seen),
        "same_split_repeated_groups": repeats,
        "cross_split_leakage": False,
    }


def validate(root, replay=True):
    root = Path(root)
    rows = rows_at(root)
    require(bool(rows), "Dataset is empty")
    info = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    require(info["count"] == len(rows), "Manifest count differs from dataset.json")
    hashes, reproduced = set(), set()
    audit = audit_splits([root])
    for row in rows:
        metadata = json.loads((root / row["metadata"]).read_text(encoding="utf-8"))
        settings = metadata["request"]["settings"]
        require(metadata["split"] == row["split"], "Metadata split differs from manifest")
        require(metadata["group_id"] == row["group_id"], "Metadata group differs from manifest")
        if "request_sha256" in metadata:
            require(
                metadata["request_sha256"] == fingerprint(metadata["request"]),
                "Request checksum differs",
            )
            require(
                metadata["group_id"] == group_identity(metadata["request"]),
                "Geometry/time group checksum differs",
            )
        with np.load(root / row["arrays"], allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        for key, value in arrays.items():
            require(value.shape == (settings["size"], settings["size"]), f"Shape differs: {key}")
            require(np.isfinite(value).all(), f"Non-finite values: {key}")
            expected_dtype = "uint8" if key.endswith("mask") else "float32"
            require(value.dtype == expected_dtype, f"Dtype differs: {key}")
        a = arrays
        np.testing.assert_allclose(
            a["delta_down_m"], a["subsidence_after_m"] - a["subsidence_before_m"], atol=1e-7
        )
        np.testing.assert_allclose(
            a["delta_down_m"], a["face_1_delta_m"] + a["face_2_delta_m"], atol=1e-7
        )
        inc, az = np.deg2rad(settings["incidence_deg"]), np.deg2rad(settings["radar_azimuth_deg"])
        los = a["delta_down_m"] * np.cos(inc) - (
            a["delta_east_m"] * np.sin(az) + a["delta_north_m"] * np.cos(az)
        ) * np.sin(inc)
        np.testing.assert_allclose(a["delta_los_m"], los, atol=1e-7)
        np.testing.assert_allclose(
            a["deformation_phase_rad"],
            -4 * np.pi * a["delta_los_m"] / settings["wavelength_m"],
            atol=2e-5,
        )
        reference = (
            a["deformation_phase_rad"]
            - 4 * np.pi * a["atmosphere_equivalent_los_m"] / settings["wavelength_m"]
            + a["orbit_phase_rad"]
            + a.get("applied_topographic_phase_rad", 0)
        )
        np.testing.assert_allclose(a["reference_phase_rad"], reference, atol=2e-5)
        np.testing.assert_allclose(
            wrap(a["wrapped_phase_rad"] - np.arctan2(a["observation_imag"], a["observation_real"])),
            0,
            atol=5e-7,
        )
        for mask, field in [
            ("mask", "delta_down_m"),
            ("face_1_mask", "face_1_delta_m"),
            ("face_2_mask", "face_2_delta_m"),
        ]:
            np.testing.assert_array_equal(
                a[mask], a[field] >= settings["target_threshold_mm"] / 1000
            )
        if "fringe_mask" in a:
            np.testing.assert_array_equal(
                a["fringe_mask"],
                fringe_support(a["deformation_phase_rad"], settings["fringe_threshold_rad"]),
            )
            np.testing.assert_array_equal(a["fringe_valid_mask"], np.ones_like(a["mask"]))
        if "valid_mask" in a:
            threshold = settings.get("valid_coherence_threshold", 0.2)
            np.testing.assert_array_equal(
                a["valid_mask"], (a["water_mask"] == 0) & (a["model_coherence"] >= threshold)
            )
            for key in ("model_coherence", "estimated_coherence"):
                require(((a[key] >= 0) & (a[key] <= 1)).all(), f"Coherence outside 0..1: {key}")
        with Image.open(root / row["mask"]) as mask_image:
            np.testing.assert_array_equal(np.asarray(mask_image), a["mask"] * 255)
        with Image.open(root / row["image"]) as phase_image:
            np.testing.assert_array_equal(
                np.asarray(phase_image), phase_rgb(a["wrapped_phase_rad"])
            )
        hashes.add(hashlib.sha256(a["wrapped_phase_rad"].tobytes()).hexdigest())
        if replay and row["category"] not in reproduced:
            require(
                metadata["engine"] == ENGINE,
                "Replay requires the original engine. Use --no-replay for numerical checks of older exports.",
            )
            saved = metadata.get("software_versions")
            if saved:
                require(
                    saved == runtime_versions(),
                    "Replay requires matching software/dependency versions; use --no-replay to check stored data.",
                )
            replayed, _ = simulate(metadata["request"])
            for key in a:
                np.testing.assert_array_equal(a[key], replayed[key])
            reproduced.add(row["category"])
    require(len(hashes) == len(rows), "Repeated observed phase images")
    result = {
        "passed": True,
        "samples": len(rows),
        "splits": dict(Counter(r["split"] for r in rows)),
        "categories": dict(Counter(r["category"] for r in rows)),
        "exact_replay_categories": sorted(reproduced),
        "split_audit": audit,
        "checks": [
            "finite arrays",
            "epoch differences",
            "component sums",
            "LOS projection",
            "phase conversion",
            "complex phase",
            "total/component masks",
            "validity mask",
            "coherence range",
            "PNG values",
            "scene identity",
            "runtime-gated replay",
        ],
    }
    (root / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument(
        "--no-replay", action="store_true", help="Check saved values without regenerating them"
    )
    args = parser.parse_args()
    reports = [validate(root, replay=not args.no_replay) for root in args.roots]
    print(
        json.dumps(
            {"datasets": reports, "combined_split_audit": audit_splits(args.roots)},
            ensure_ascii=False,
            indent=2,
        )
    )
