"""本工程自己的图像、参数与数据集接口。"""

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from mine_model import ENGINE, default_request, parse_request, request_dict, simulate
from phase_encoding import quantized_phase
from provenance import SCHEMA_VERSION, fingerprint, runtime_versions


def phase_rgb(phase):
    p = np.asarray(phase, dtype=float)
    channels = [0.5 + 0.5 * np.cos(p - shift) for shift in (0, 2 * np.pi / 3, 4 * np.pi / 3)]
    return np.stack(channels, axis=-1).__mul__(255).round().astype(np.uint8)


def magnitude_rgb(values):
    maximum = max(float(np.max(values)), 1e-12)
    t = np.clip(values / maximum, 0, 1)
    anchors = np.array([[14, 24, 42], [18, 83, 114], [34, 180, 168], [253, 219, 143]])
    return np.stack(
        [np.interp(t, [0, 0.3, 0.65, 1], anchors[:, i]) for i in range(3)], axis=-1
    ).astype(np.uint8)


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def images(arrays):
    observed = phase_rgb(arrays["wrapped_phase_rad"])
    mask = arrays["mask"].astype(bool)
    interior = mask.copy()
    interior[1:-1, 1:-1] &= mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False
    overlay = observed.copy()
    overlay[mask & ~interior] = [255, 255, 255]
    fringe = arrays["fringe_mask"].astype(bool)
    edge = fringe.copy()
    inside = np.pad(fringe, 1)
    edge &= ~(inside[:-2, 1:-1] & inside[2:, 1:-1] & inside[1:-1, :-2] & inside[1:-1, 2:])
    fringe_overlay = observed.copy()
    fringe_overlay[edge] = [255, 255, 255]
    return {
        "fringe_mask": arrays["fringe_mask"] * 255,
        "fringe_overlay": fringe_overlay,
        "observed": observed,
        "deformation": phase_rgb(arrays["deformation_phase_rad"]),
        "reference": phase_rgb(arrays["reference_phase_rad"]),
        "displacement": magnitude_rgb(arrays["delta_down_m"]),
        "mask": arrays["mask"] * 255,
        "overlay": overlay,
        "dem": magnitude_rgb(arrays["dem_m"] - arrays["dem_m"].min()),
        "coherence": np.round(arrays["estimated_coherence"] * 255).astype(np.uint8),
        "water": np.where(
            arrays["water_mask"][..., None] > 0, np.array([38, 136, 200]), np.array([230, 234, 224])
        ).astype(np.uint8),
        "valid": arrays["valid_mask"] * 255,
    }


def single_archive(request):
    arrays, metadata = simulate(request)
    data = io.BytesIO()
    np.savez_compressed(data, **arrays)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("arrays.npz", data.getvalue())
        output.writestr(
            "parameters.json", json.dumps(metadata["request"], ensure_ascii=False, indent=2)
        )
        output.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        for name, array in images(arrays).items():
            output.writestr(name + ".png", png(array))
        output.writestr(
            "README.txt",
            "Independent working-face simulator\n"
            "Coordinates: East, North, Down; length=m; time=day; phase=rad.\n"
            "mask = delta_down_m >= target_threshold_mm/1000.\n"
            "fringe_mask = abs(clean deformation phase) >= fringe_threshold_rad; experimental task support, not calibrated visible fringe extent. fringe_valid_mask supervises all pixels.\n"
            "reference_phase_rad contains atmosphere, orbit and applied topography; deformation_phase_rad does not.\n"
            "See metadata.observation_model. Statistical complex pairs are not a complete SAR image simulator. Synthetic DEM and water are not measured data. valid_mask follows the exported validity rule and threshold.\n",
        )
    return archive.getvalue()


def draw_sample(base, index, category, profile="standard"):
    if profile not in ("standard", "sparse_mine"):
        raise ValueError("Unknown sampling profile")
    if category not in ("negative", "single", "separated", "overlapping", "staggered"):
        raise ValueError("Unknown sampling category")
    request = json.loads(json.dumps(base))
    rng = np.random.default_rng(np.random.SeedSequence([base["settings"]["seed"], index, 1201]))
    s = request["settings"]
    s.update(
        seed=int(rng.integers(0, 2**32)),
        day_before=float(rng.uniform(25, 70)),
        atmosphere_mm=float(rng.uniform(0.5, 4)),
        orbit_cycles=float(rng.uniform(-5, 5)),
        phase_spread_rad=float(rng.uniform(0.15, 0.9)),
        land_coherence=float(rng.uniform(0.7, 0.98)),
        water_coherence=float(rng.uniform(0, 0.08)),
        relief_m=float(rng.uniform(30, 350)),
        dem_error_m=float(rng.uniform(0, 12)),
        disturbed_patch=bool(rng.random() < 0.25),
    )
    s["day_after"] = s["day_before"] + float(rng.uniform(6, 18))
    while len(request["faces"]) < 2:
        request["faces"].append(default_request()["faces"][1])
    separation = float(rng.uniform(0.07, 0.14) * s["extent_m"])
    if category == "separated":
        separation = float(rng.uniform(0.28, 0.40) * s["extent_m"])
    midpoint = rng.uniform(-0.12, 0.12, 2) * s["extent_m"]
    direction = rng.uniform(0, 2 * np.pi)
    for j, face in enumerate(request["faces"]):
        face.update(
            enabled=category != "negative" and (j == 0 or category != "single"),
            east_m=float(midpoint[0] + (j - 0.5) * separation * np.cos(direction)),
            north_m=float(midpoint[1] + (j - 0.5) * separation * np.sin(direction)),
            length_m=float(rng.uniform(220, 540)),
            width_m=float(rng.uniform(100, 220)),
            depth_m=float(rng.uniform(160, 360)),
            bearing_deg=float(rng.uniform(-180, 180)),
            advance_m_day=float(rng.uniform(4, 10)),
            thickness_m=float(rng.uniform(1, 3)),
            subsidence_factor=float(rng.uniform(0.4, 0.8)),
            response_days=float(rng.uniform(50, 130)),
            start_day=float(rng.uniform(0, 15)),
        )
        if category == "staggered" and j == 1:
            face["start_day"] = s["day_before"] + float(rng.uniform(-5, 3))
    if profile == "sparse_mine":
        # Design ranges, not parameters calibrated from this or any measured mine.
        s.update(
            extent_m=float(rng.uniform(1800, 4200)),
            atmosphere_mm=float(rng.uniform(0.5, 7)),
            atmosphere_scale_m=float(rng.uniform(120, 1500)),
            orbit_cycles=float(rng.uniform(-18, 18)),
            land_coherence=float(rng.uniform(0.35, 0.95)),
            looks=int(rng.integers(1, 5)),
            spatial_window=int(rng.choice([1, 3], p=[0.7, 0.3])),
            dem_error_m=float(rng.uniform(0, 30)),
            water_enabled=bool(base["settings"]["water_enabled"] and rng.random() < 0.25),
        )
        middle = rng.uniform(-0.35, 0.35, 2) * s["extent_m"]
        gap = rng.uniform(0.03, 0.10) * s["extent_m"]
        if category == "separated":
            gap = rng.uniform(0.18, 0.32) * s["extent_m"]
        for j, face in enumerate(request["faces"]):
            face.update(
                east_m=float(middle[0] + (j - 0.5) * gap * np.cos(direction)),
                north_m=float(middle[1] + (j - 0.5) * gap * np.sin(direction)),
                length_m=float(rng.uniform(120, 450)),
                width_m=float(rng.uniform(50, 180)),
                depth_m=float(rng.uniform(120, 380)),
                influence_tangent=float(rng.uniform(1.6, 3.0)),
                thickness_m=float(rng.uniform(0.4, 2.0)),
            )
    return request


def export_dataset(
    root, count, base, progress=lambda done, total: None, stop=None, profile="standard"
):
    if profile not in ("standard", "sparse_mine"):
        raise ValueError("Unknown sampling profile")
    if type(count) is not int or not 50 <= count <= 2000:
        raise ValueError("样本数需为 50–2000 的整数")
    settings, faces = parse_request(base)
    base = request_dict(settings, faces)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(base["settings"]["seed"])
    categories = ("negative", "single", "separated", "overlapping", "staggered")
    records = []
    for j, category in enumerate(categories):
        ids = np.arange(j, count, len(categories))
        rng.shuffle(ids)
        holdout = max(1, round(len(ids) * 0.1))
        for rank, index in enumerate(ids):
            split = "test" if rank < holdout else ("val" if rank < 2 * holdout else "train")
            records.append((int(index), category, split))
    records.sort()
    rows, thumbs = [], []
    for index, category, split in records:
        if stop is not None and stop.is_set():
            break
        request = draw_sample(base, index, category, profile)
        arrays, metadata = simulate(request)
        group = f"face_scene_{index:05d}"
        metadata.update(scene_id=group, category=category, split=split, sampling_profile=profile)
        paths = {
            name: f"{split}/{folder}/{group}{suffix}"
            for name, folder, suffix in [
                ("image", "images", ".png"),
                ("mask", "masks", ".png"),
                ("arrays", "arrays", ".npz"),
                ("metadata", "metadata", ".json"),
            ]
        }
        for value in paths.values():
            (root / value).parent.mkdir(parents=True, exist_ok=True)
        (root / paths["image"]).write_bytes(png(phase_rgb(arrays["wrapped_phase_rad"])))
        (root / paths["mask"]).write_bytes(png(arrays["mask"] * 255))
        np.savez_compressed(root / paths["arrays"], **arrays)
        (root / paths["metadata"]).write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        rows.append(
            {
                "scene_id": group,
                "group_id": metadata["group_id"],
                "split": split,
                "category": category,
                **paths,
            }
        )
        if len(thumbs) < 10:
            thumbs.append(
                Image.fromarray(phase_rgb(arrays["wrapped_phase_rad"])).resize((192, 192))
            )
        progress(len(rows), count)
    with (root / "manifest.csv").open("w", newline="") as stream:
        names = (
            list(rows[0])
            if rows
            else [
                "scene_id",
                "group_id",
                "split",
                "category",
                "image",
                "mask",
                "arrays",
                "metadata",
            ]
        )
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    info = {
        "schema_version": SCHEMA_VERSION,
        "software_versions": runtime_versions(),
        "base_request_sha256": fingerprint(base),
        "engine": ENGINE,
        "count": len(rows),
        "requested_count": count,
        "cancelled": len(rows) != count,
        "base_request": base,
        "split_rule": "independent working-face scenes; no scene occurs in more than one split",
        "categories": list(categories),
        "sampling_profile": profile,
        "category_note": "scenario design labels; actual target overlap is in each metadata.overlap_pixels",
        "label_rule": "vertical subsidence increment threshold; not LOS or noisy image threshold",
    }
    (root / "dataset.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    if thumbs:
        canvas = Image.new("RGB", (192 * 5, 192 * 2), "#101b2b")
        for index, thumb in enumerate(thumbs):
            canvas.paste(thumb, ((index % 5) * 192, (index // 5) * 192))
        canvas.save(root / "preview.png")
    return {"path": str(root.resolve()), "count": len(rows), "cancelled": info["cancelled"]}


class PhaseDataset:
    """可直接交给 PyTorch DataLoader；torch 只在读取训练样本时需要。"""

    def __init__(
        self,
        root,
        split="train",
        return_valid=False,
        input_mode="phase",
        require_valid=False,
        phase_bins=0,
        augment_phase=False,
        target_mode="physical",
    ):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        if input_mode not in ("phase", "rgb"):
            raise ValueError("input_mode must be phase or rgb")
        if target_mode not in ("physical", "fringe"):
            raise ValueError("Unknown target mode")
        self.target_mode = target_mode
        self.return_valid = return_valid
        self.input_mode = input_mode
        self.require_valid = require_valid
        quantized_phase(np.zeros(1), phase_bins)
        if (phase_bins or augment_phase) and input_mode != "phase":
            raise ValueError("Phase quantization/augmentation requires phase input")
        self.phase_bins = phase_bins
        self.augment_phase = augment_phase
        self.root = Path(root)
        with (self.root / "manifest.csv").open() as stream:
            self.rows = [row for row in csv.DictReader(stream) if row["split"] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch

        with np.load(self.root / self.rows[index]["arrays"], allow_pickle=False) as data:
            if self.input_mode == "phase":
                phase = quantized_phase(data["wrapped_phase_rad"], self.phase_bins)
                if self.augment_phase:
                    phase = phase * np.random.choice([-1, 1]) + np.random.uniform(-np.pi, np.pi)
                x = np.stack((np.sin(phase), np.cos(phase))).astype(np.float32)
            else:
                with Image.open(self.root / self.rows[index]["image"]) as image:
                    x = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255
            y = data["mask"][None].astype(np.float32)
            if self.require_valid and "valid_mask" not in data:
                raise ValueError("Training requires exported valid_mask; regenerate legacy data")
            valid = (
                data["valid_mask"][None].astype(np.float32)
                if "valid_mask" in data
                else np.ones_like(y)
            )
        if self.target_mode == "fringe":
            with np.load(self.root / self.rows[index]["arrays"], allow_pickle=False) as data:
                y = data["fringe_mask"][None].astype(np.float32)
                valid = data["fringe_valid_mask"][None].astype(np.float32)
        if x.shape[1:] != y.shape[1:] or valid.shape != y.shape:
            raise ValueError("Input, label and validity dimensions differ")
        if self.return_valid:
            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(valid)
        return torch.from_numpy(x), torch.from_numpy(y)
