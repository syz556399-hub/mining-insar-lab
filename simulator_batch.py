"""Expand an explicitly accepted review batch using the unchanged simulator core."""

import argparse
import csv
import json
import shutil
import time
from collections import Counter, defaultdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from annotation_audit import decode_annotation_mask
from simulator_review import labelme_doc, now, read_json, sha, write_json
from simulator_v3 import ENGINE, candidate_mask, overlay, simulate, validate, views

ROOT = Path(__file__).resolve().parent
SAMPLING = {
    "name": "accepted-style-moderate-variation-v1",
    "scope": "design ranges around accepted configurations, not fitted physical parameters",
    "extent_multiplier": [0.85, 1.15],
    "source_cycles_multiplier": [0.8, 1.2],
    "background_cycles_multiplier": [0.65, 1.35],
    "background_angle_offset_deg": [-35, 35],
    "target_radius_multiplier": [0.9, 1.1],
    "coherence_offset": [-0.03, 0.03],
    "atmosphere_multiplier": [0.8, 1.2],
    "synthetic_water_probability": 0.15,
    "boundary_rad": 0.8,
    "footprint": "rectangular",
    "style": "bmp",
}


def accepted_templates(run):
    manifest = read_json(run / "manifest.json")
    templates = []
    for row in manifest["samples"]:
        scene = run / row["folder"]
        review, meta = read_json(scene / "review.json"), read_json(scene / "metadata.json")
        if review["status"] != "accepted":
            raise ValueError(f"Baseline {row['id']} is not explicitly accepted")
        receipt = review["accepted_receipt"]
        if receipt["image_sha256"] != sha(scene / f"{meta['config']['style']}.png") or receipt[
            "mask_sha256"
        ] != sha(scene / "mask.png"):
            raise ValueError(f"Baseline {row['id']} changed after acceptance")
        templates.append(
            {
                "id": row["id"],
                "revision": row["revision"],
                "config": validate(meta["config"]),
                "group_id": meta["group_id"],
                "reference_hashes": {
                    name: sha(scene / name)
                    for name in (
                        "bmp.png",
                        "mask.png",
                        "arrays.npz",
                        "metadata.json",
                        "review.json",
                    )
                },
            }
        )
    if len(templates) != 20 or len({r["group_id"] for r in templates}) != 20:
        raise ValueError("Use the complete, individually accepted 20-scene review batch")
    return templates, read_json(run / "display_profile.json")


def sampling_plan(templates, count, seed):
    if type(count) is not int or not 20 <= count <= 2000:
        raise ValueError("count must be an integer in 20..2000")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32 integer")
    forbidden = {r["config"]["seed"] for r in templates}
    rows = []
    for index in range(count):
        template = templates[index % len(templates)]
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, 5101]))
        source_seed = int(rng.integers(0, 2**32))
        while source_seed in forbidden:
            source_seed = int(rng.integers(0, 2**32))
        forbidden.add(source_seed)
        c = dict(template["config"])
        c.update(
            seed=source_seed,
            size=256,
            style="bmp",
            footprint="rectangular",
            boundary_rad=0.8,
            extent_m=float(np.clip(c["extent_m"] * rng.uniform(0.85, 1.15), 1600, 4600)),
            cycles=float(np.clip(c["cycles"] * rng.uniform(0.8, 1.2), 1, 6.5)),
            background_cycles=float(
                np.clip(c["background_cycles"] * rng.uniform(0.65, 1.35), 0.8, 7)
            ),
            background_angle_deg=float(
                (c["background_angle_deg"] + rng.uniform(-35, 35) + 180) % 360 - 180
            ),
            target_radius_fraction=float(
                np.clip(c["target_radius_fraction"] * rng.uniform(0.9, 1.1), 0.04, 0.14)
            ),
            atmosphere_mm=float(c["atmosphere_mm"] * rng.uniform(0.8, 1.2)),
            coherence=float(np.clip(c["coherence"] + rng.uniform(-0.03, 0.03), 0.6, 0.95)),
            terrain_relief_m=float(c["terrain_relief_m"] * rng.uniform(0.75, 1.25)),
            dem_error_m=float(c["dem_error_m"] * rng.uniform(0.6, 1.4)),
            baseline_m=float(c["baseline_m"] * rng.uniform(0.8, 1.2)),
            water=bool(rng.random() < SAMPLING["synthetic_water_probability"]),
        )
        rows.append(
            {
                "id": f"B{index + 1:05d}",
                "template_id": template["id"],
                "template_revision": template["revision"],
                "config": validate(c),
            }
        )
    # Stratify by scenario; all variants of any actual source remain in one group.
    by_case = defaultdict(list)
    for i, row in enumerate(rows):
        by_case[row["config"]["case"]].append(i)
    split_rng = np.random.default_rng(np.random.SeedSequence([seed, 5102]))
    keys = sorted(by_case)
    holdout = count // 10
    quotas = {key: len(by_case[key]) // 10 for key in keys}
    ranked = sorted(keys, key=lambda key: (-(len(by_case[key]) % 10), key))
    for key in ranked[: holdout - sum(quotas.values())]:
        quotas[key] += 1
    for key in keys:
        indices = split_rng.permutation(by_case[key])
        for rank, i in enumerate(indices):
            rows[int(i)]["split"] = (
                "test" if rank < quotas[key] else "val" if rank < 2 * quotas[key] else "train"
            )
    return rows


def make_preview(root, rows):
    chosen = rows[:20]
    width, height = 512, 284
    sheet = Image.new("RGB", (width * 4, height * 5), "#14212a")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(chosen):
        x, y = index % 4 * width, index // 4 * height
        with Image.open(root / row["image"]) as image:
            rgb = np.asarray(image)
            sheet.paste(image, (x, y))
        with Image.open(root / row["mask"]) as image:
            mask = np.asarray(image) > 0
        sheet.paste(Image.fromarray(overlay(rgb, mask)), (x + 256, y))
        draw.text(
            (x + 5, y + 260),
            f"{row['id']}  {row['case']}  {row['split']} | input / boundary",
            fill="white",
        )
    sheet.save(root / "preview_20.jpg", quality=92)


def audit_dataset(root):
    doc = read_json(root / "manifest.json")
    if doc["generation_status"] != "complete" or len(doc["samples"]) != doc["requested_count"]:
        raise ValueError("Batch is incomplete")
    groups, image_hashes = {}, set()
    baseline_groups = {
        r["group_id"] for r in read_json(root / "baseline_receipt.json")["templates"]
    }
    empty, warnings = [], []
    for row in doc["samples"]:
        if row["status"] != "draft" or row["source"] != "synthetic":
            raise ValueError("New automatic labels must remain unreviewed synthetic labels")
        group = row["group_id"]
        if group in baseline_groups:
            raise ValueError("A new sample reused an accepted baseline source")
        if group in groups:
            raise ValueError("A source group was duplicated or crossed a split")
        groups[group] = row["split"]
        if row["image_sha256"] in image_hashes:
            raise ValueError("Duplicate rendered image")
        image_hashes.add(row["image_sha256"])
        for key in ("image", "mask", "arrays", "metadata", "labelme"):
            path = root / row[key]
            if (
                not path.is_file()
                or path.is_symlink()
                or root.resolve() not in path.resolve().parents
            ):
                raise ValueError(f"Invalid batch file: {row[key]}")
            if sha(path) != row[f"{key}_sha256"]:
                raise ValueError(f"File changed: {row[key]}")
        with Image.open(root / row["image"]) as image:
            if image.mode != "RGB" or image.size != (256, 256):
                raise ValueError("Expected a native 256-square RGB image")
        with Image.open(root / row["mask"]) as image:
            mask = np.array(image)
            if image.mode != "L" or image.size != (256, 256) or not np.isin(mask, [0, 255]).all():
                raise ValueError("Expected an aligned binary mask")
        with np.load(root / row["arrays"], allow_pickle=False) as arrays:
            if any(v.shape != (256, 256) or not np.isfinite(v).all() for v in arrays.values()):
                raise ValueError("Numerical arrays are not finite aligned native images")
            if not np.array_equal(
                mask > 0, candidate_mask(arrays["deformation_phase_rad"], 0.8) > 0
            ):
                raise ValueError("Saved label differs from the declared automatic rule")
        if not np.array_equal(mask > 0, decode_annotation_mask(read_json(root / row["labelme"]))):
            raise ValueError("LabelMe round trip changed the label")
        if row["case"] == "negative":
            if mask.any():
                raise ValueError("Negative example contains a foreground label")
            empty.append(row["id"])
        elif not mask.any():
            raise ValueError("A positive example has an empty label; review the design")
        if row["warnings"]:
            warnings.append(row["id"])
    return {
        "checked_at": now(),
        "count": len(doc["samples"]),
        "split_counts": dict(Counter(r["split"] for r in doc["samples"])),
        "case_counts": dict(Counter(r["case"] for r in doc["samples"])),
        "water_count": sum(r["water_percent"] > 0 for r in doc["samples"]),
        "negative_count": len(empty),
        "unique_source_groups": len(groups),
        "unique_images": len(image_hashes),
        "exact_labelme_roundtrips": len(doc["samples"]),
        "candidate_rule_matches": len(doc["samples"]),
        "numerical_warning_samples": warnings,
        "scope": "file, split and numerical checks; not per-image human approval or real accuracy",
    }


def sample_paths(sid):
    return {
        key: f"{folder}/{sid}{suffix}"
        for key, folder, suffix in (
            ("image", "images", ".png"),
            ("mask", "masks", ".png"),
            ("arrays", "arrays", ".npz"),
            ("metadata", "metadata", ".json"),
            ("labelme", "labelme", ".json"),
        )
    }


def sample_record(output, row, meta):
    paths = sample_paths(row["id"])
    return {
        "id": row["id"],
        "group_id": meta["group_id"],
        "source": "synthetic",
        "status": "draft",
        "split": row["split"],
        "case": row["config"]["case"],
        "style": "bmp",
        "template_id": row["template_id"],
        "template_revision": row["template_revision"],
        "label_percent": meta["label_percent"],
        "water_percent": meta["water_percent"],
        "warnings": meta["warnings"],
        **paths,
        **{f"{key}_sha256": sha(output / path) for key, path in paths.items()},
    }


def recover_completed(output, plan, recorded):
    """Adopt only a contiguous set of complete, plan-bound files; never overwrite a partial sample."""
    known = {r["id"]: r for r in recorded}
    if len(known) != len(recorded) or not set(known) <= {r["id"] for r in plan}:
        raise ValueError("The recorded IDs do not belong to this sampling plan")
    found, missing = [], False
    for row in plan:
        paths = sample_paths(row["id"])
        present = [(output / p).is_file() for p in paths.values()]
        if not any(present):
            missing = True
            if row["id"] in known:
                raise ValueError("A recorded sample is missing")
            continue
        if missing or not all(present):
            raise ValueError("Partial or non-contiguous files require inspection before resuming")
        meta = read_json(output / paths["metadata"])
        if (
            meta["config"] != row["config"]
            or meta["sample_id"] != row["id"]
            or meta["split"] != row["split"]
            or meta["template_id"] != row["template_id"]
        ):
            raise ValueError("Existing files do not belong to this sampling plan")
        record = sample_record(output, row, meta)
        if row["id"] in known and record != known[row["id"]]:
            raise ValueError("A recorded sample changed before resuming")
        found.append(record)
    return found


def build_batch(run, output, count=500, seed=20261010, resume=False, resume_note=""):
    templates, profile = accepted_templates(run)
    plan = sampling_plan(templates, count, seed)
    source_hashes = {
        name: sha(ROOT / name)
        for name in (
            "simulator_v3.py",
            "simulator_batch.py",
            "mine_model.py",
            "scene_environment.py",
        )
    }
    if resume:
        doc = read_json(output / "manifest.json")
        if (
            doc["generation_status"] == "complete"
            or doc["requested_count"] != count
            or doc["seed"] != seed
        ):
            raise ValueError("Cannot resume a completed or different batch")
        baseline = read_json(output / "baseline_receipt.json")
        if baseline["templates"] != templates or baseline["profile_sha256"] != sha(
            run / "display_profile.json"
        ):
            raise ValueError("The accepted baseline changed")
        if (
            read_json(output / "sampling_plan.json")["samples"] != plan
            or read_json(output / "display_profile.json") != profile
        ):
            raise ValueError("The sampling plan or display profile changed")
        doc["samples"] = recover_completed(output, plan, doc["samples"])
        doc.setdefault("execution_events", []).append(
            {
                "at": now(),
                "action": "resume",
                "completed_before_resume": len(doc["samples"]),
                "source_hashes": source_hashes,
                "reason": resume_note,
            }
        )
        print(json.dumps({"resuming_from": len(doc["samples"]), "total": count}), flush=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
        for folder in ("images", "masks", "arrays", "metadata", "labelme"):
            (output / folder).mkdir()
        write_json(
            output / "baseline_receipt.json",
            {"templates": templates, "profile_sha256": sha(run / "display_profile.json")},
        )
        write_json(
            output / "sampling_plan.json", {"seed": seed, "sampling": SAMPLING, "samples": plan}
        )
        write_json(output / "display_profile.json", profile)
        shutil.copyfile(ROOT / "web/simulator-batch/index.html", output / "index.html")
        (output / "load_dataset.py").write_text(LOADER, encoding="utf-8")
        doc = {
            "engine": ENGINE,
            "created_at": now(),
            "status": "draft",
            "generation_status": "running",
            "requested_count": count,
            "seed": seed,
            "sampling": SAMPLING,
            "label_rule": "abs(clean spatially averaged deformation phase) >= 0.8 rad",
            "label_status": "automatically generated; individual human review pending",
            "source_policy": "accepted configuration rules only; no copied images, masks or third-party outputs",
            "split_policy": "scenario-stratified independent source groups; every derived view/patch inherits its group split",
            "samples": [],
            "source_hashes": source_hashes,
        }
    write_json(output / "manifest.json", doc)
    started = time.monotonic()
    for row in plan[len(doc["samples"]) :]:
        sid = row["id"]
        arrays, meta = simulate(row["config"])
        rgb = views(arrays, profile)["bmp"]
        mask = arrays["candidate_mask"]
        paths = sample_paths(sid)
        Image.fromarray(rgb).save(output / paths["image"])
        Image.fromarray(mask * 255).save(output / paths["mask"])
        np.savez_compressed(output / paths["arrays"], **arrays)
        meta.update(
            sample_id=sid,
            split=row["split"],
            template_id=row["template_id"],
            template_revision=row["template_revision"],
            sampling_profile=SAMPLING["name"],
            label_rule=doc["label_rule"],
            label_status=doc["label_status"],
            display_profile_sha256=sha(output / "display_profile.json"),
        )
        write_json(output / paths["metadata"], meta)
        labelme = labelme_doc(sid, mask, 256)
        labelme["imagePath"] = f"../images/{sid}.png"
        write_json(output / paths["labelme"], labelme)
        doc["samples"].append(sample_record(output, row, meta))
        write_json(output / "manifest.json", doc)
        done = len(doc["samples"])
        if done % 10 == 0 or done == count:
            write_json(output / "manifest.json", doc)
            write_json(
                output / "progress.json",
                {"generated": done, "total": count, "elapsed_seconds": time.monotonic() - started},
            )
            print(
                json.dumps(
                    {
                        "generated": done,
                        "total": count,
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
        if done == 20:
            make_preview(output, doc["samples"])
    # A completed generation is distinct from human approval of automatic labels.
    doc.update(generation_status="complete", completed_at=now())
    write_json(output / "manifest.json", doc)
    checks = audit_dataset(output)
    current_templates, _ = accepted_templates(run)
    if current_templates != templates:
        raise ValueError("Accepted baseline changed during generation")
    checks.update(accepted_20_unchanged=True, elapsed_seconds=time.monotonic() - started)
    write_json(output / "validation.json", checks)
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "id",
            "group_id",
            "split",
            "case",
            "image",
            "mask",
            "arrays",
            "metadata",
            "labelme",
            "status",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(doc["samples"])
    (output / "README.md").write_text(
        "# 已确认风格的批量合成样本\n\n"
        f"共 {count} 个独立场景，256×256 RGB 影像。以已确认的 20 张参数规则为模板，使用新种子和温和参数变化；没有复制模板图像或标签。\n\n"
        "输入在 images；精确二值标签在 masks；物理/观测数组在 arrays；参数和标签规则在 metadata；LabelMe JSON 的 imagePath 指向 ../images。\n\n"
        "标签规则：abs(无噪声、空间平均后的形变相位) >= 0.8 rad。新样本保持 draft，尚未逐张人工确认。load_dataset.py 的 SyntheticRanges 需明确 allow_draft=True 才可用于初步实验。\n\n"
        "划分已按独立源场景与场景类型固定，详情见 manifest.json/manifest.csv。同源裁片、颜色变体和增强必须继承源划分。合成测试集仅检查合成域，真实验证需按独立日期/地点另建。\n\n"
        "水体、DEM、大气与反射纹理为合成或经验模型；参数范围为设计增强，不是实测统计或校准。亮度沿用开发 BMP 的显示统计，未读取真实保留验证集。\n\n"
        "sampling_plan.json 保存生成计划，baseline_receipt.json 保存模板确认哈希。validation.json 记录文件、标签、数值与划分校验。index.html 是本机查看页，preview_20.jpg 是前 20 张影像/边界对照。\n",
        encoding="utf-8",
    )
    print(json.dumps(checks, ensure_ascii=False), flush=True)
    return checks


LOADER = '''"""Native RGB / exact binary labels. Generated labels require explicit opt-in."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class SyntheticRanges(Dataset):
    def __init__(self, root, split=None, allow_draft=False):
        self.root = Path(root)
        doc = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if doc["generation_status"] != "complete":
            raise ValueError("Generation is incomplete")
        if doc["status"] != "reviewed" and not allow_draft:
            raise ValueError("Automatic labels require explicit allow_draft=True")
        if split not in (None, "train", "val", "test"):
            raise ValueError("Unknown split")
        self.rows = [r for r in doc["samples"] if split is None or r["split"] == split]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        for key in ("image", "mask"):
            if hashlib.sha256((self.root / r[key]).read_bytes()).hexdigest() != r[key + "_sha256"]:
                raise ValueError("Image or label changed")
        with Image.open(self.root / r["image"]) as im:
            x = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
        with Image.open(self.root / r["mask"]) as im:
            y = (np.asarray(im) > 0).astype(np.float32)
        if x.shape[:2] != y.shape:
            raise ValueError("Image/mask alignment error")
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(y[None].copy()), r["group_id"]
'''


def serve(root, port):
    if read_json(root / "manifest.json")["generation_status"] != "complete":
        raise ValueError("Wait until the batch is complete")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def do_GET(self):
            if self.headers.get("Host", "").split(":")[0] not in ("127.0.0.1", "localhost"):
                self.send_error(403)
                return
            super().do_GET()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Batch viewer: http://127.0.0.1:{port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--review-run", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--count", type=int, default=500)
    build.add_argument("--seed", type=int, default=20261010)
    build.add_argument("--resume", action="store_true")
    build.add_argument("--resume-note", default="Explicit resume after inspecting the interruption")
    audit = commands.add_parser("validate")
    audit.add_argument("root", type=Path)
    viewer = commands.add_parser("serve")
    viewer.add_argument("root", type=Path)
    viewer.add_argument("--port", type=int, default=8782)
    args = parser.parse_args()
    if args.command == "build":
        build_batch(
            args.review_run.resolve(),
            args.out.resolve(),
            args.count,
            args.seed,
            args.resume,
            args.resume_note,
        )
    elif args.command == "validate":
        print(json.dumps(audit_dataset(args.root.resolve()), ensure_ascii=False, indent=2))
    else:
        serve(args.root.resolve(), args.port)
