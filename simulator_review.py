"""Local 20-scene review desk and portable, auditable segmentation exports."""

import argparse
import base64
import hashlib
import io
import json
import mimetypes
import threading
import zipfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
from PIL import Image, ImageDraw

from simulator_v3 import (
    CASE_NAMES,
    ENGINE,
    batch_configs,
    candidate_mask,
    fit_display_profile,
    overlay,
    save_scene,
    simulate,
    validate,
)

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web" / "simulator-v3"
LOCK = threading.RLock()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def contact_sheet(folder, manifest):
    sheet = Image.new("RGB", (5 * 256, 4 * 292), "#17222d")
    draw = ImageDraw.Draw(sheet)
    for i, item in enumerate(manifest["samples"]):
        scene = folder / item["folder"]
        meta = read_json(scene / "metadata.json")
        style = meta["config"]["style"]
        with Image.open(scene / f"{style}_overlay.png") as im:
            sheet.paste(im, ((i % 5) * 256, (i // 5) * 292))
        draw.text(
            ((i % 5) * 256 + 8, (i // 5) * 292 + 260),
            f"{item['id']}  {style}  {meta['config']['case']}",
            fill="white",
        )
    sheet.save(folder / "contact_sheet.jpg", quality=92)


def build_batch(folder, source=None, seed=20261009):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    profile = fit_display_profile(source) if source else None
    write_json(folder / "display_profile.json", profile)
    manifest = {
        "engine": ENGINE,
        "created_at": now(),
        "iteration": 1,
        "boundary_policy": {
            "recommended_threshold_rad": 0.8,
            "status": "candidate, individual review required",
        },
        "purpose": "20-source visual/label review, not a held-out generalization benchmark",
        "samples": [],
        "reference_usage": "statistics only; no copied BMP patches or external simulator outputs in samples",
    }
    for i, config in enumerate(batch_configs(seed)):
        sample_id = f"S{i + 1:02d}"
        arrays, meta = simulate(config)
        meta["sample_id"] = sample_id
        save_scene(folder / sample_id, arrays, meta, profile)
        write_json(
            folder / sample_id / "review.json",
            {
                "status": "pending",
                "revision": 1,
                "threshold_rad": config["boundary_rad"],
                "manual_mask": False,
                "note": "",
                "history": [],
            },
        )
        manifest["samples"].append({"id": sample_id, "folder": sample_id, "revision": 1})
        print(
            f"{sample_id}: {config['style']} {config['case']} | mask {meta['label_percent']:.1f}% | {meta['source_cycles']:.2f} cycles",
            flush=True,
        )
    write_json(folder / "manifest.json", manifest)
    contact_sheet(folder, manifest)
    (folder / "README.md").write_text(
        "# 模拟器重做 · 第一轮 20 张\n\n"
        "20 个独立几何场景，每个同时保留 BMP 风格和研究图风格，属于同一 group。"
        "两种显示不能分入不同数据集。S10/S20 是负样本，用于检查背景误报。\n\n"
        "标签尚待逐张确认：当前边界是无噪声形变相位支持轮廓的候选线，"
        "不是已校准的真实沉降范围。可以改阈值、画笔修正，确认后进入可训练导出。"
        "原有真实 LabelMe 标注没有改动。\n\n"
        "真实 BMP 只用于显示色表与亮度分布统计；没有复制其图块或恢复真实物理相位。"
        "DEM/水体都是合成场景，水体只在 4 张中出现。"
        "源模型使用本工程自己实现的概率积分法，加入明确记录的幅度/形状增强，"
        "不包含第三方模拟器源码、权重或图像。\n\n"
        "运行：在工程根目录执行 `python simulator_review.py serve --run <本目录> --port 8781`。\n",
        encoding="utf-8",
    )
    return manifest


def summary(folder):
    manifest = read_json(folder / "manifest.json")
    items = []
    for entry in manifest["samples"]:
        scene = folder / entry["folder"]
        meta, review = read_json(scene / "metadata.json"), read_json(scene / "review.json")
        items.append(
            {
                **entry,
                "config": meta["config"],
                "review": review,
                "case_name": CASE_NAMES[meta["config"]["case"]],
                "label_percent": meta["label_percent"],
                "source_cycles": meta["source_cycles"],
                "water_percent": meta["water_percent"],
                "warnings": meta["warnings"],
            }
        )
    return {
        "engine": ENGINE,
        "samples": items,
        "run_name": folder.name,
        "accepted": sum(i["review"]["status"] == "accepted" for i in items),
        "boundary_policy": manifest.get("boundary_policy", {"recommended_threshold_rad": 0.8}),
        "profile": read_json(folder / "display_profile.json"),
    }


def scene_entry(folder, sample_id):
    manifest = read_json(folder / "manifest.json")
    for entry in manifest["samples"]:
        if entry["id"] == sample_id:
            return entry, folder / entry["folder"]
    raise ValueError("Unknown sample")


def boundary_preview(scene, style, kind):
    """Same crop for all candidates; zoom is display only, never exported for training."""
    meta, review = read_json(scene / "metadata.json"), read_json(scene / "review.json")
    threshold = review["threshold_rad"]
    with np.load(scene / "arrays.npz", allow_pickle=False) as arrays:
        phase = arrays["deformation_phase_rad"]
        wider = candidate_mask(phase, max(0.02, threshold / 2))
        tighter = candidate_mask(phase, min(10, threshold * 2))
    with Image.open(scene / "mask.png") as image:
        current = (np.asarray(image) > 0).astype(np.uint8)
    with Image.open(scene / f"{style}.png") as image:
        rgb = np.asarray(image.convert("RGB"))
    extent = wider | current
    ys, xs = np.nonzero(extent)
    n = meta["config"]["size"]
    if len(xs):
        span = min(n, max(64, int(max(np.ptp(xs), np.ptp(ys))) + 24))
        x = int(np.clip((xs.min() + xs.max() - span) / 2, 0, n - span))
        y = int(np.clip((ys.min() + ys.max() - span) / 2, 0, n - span))
        box = (x, y, x + span, y + span)
    else:
        box = (0, 0, n, n)
    if kind != "input":
        chosen = {"wide": wider, "current": current, "tight": tighter}[kind]
        rgb = overlay(rgb, chosen)
    im = Image.fromarray(rgb).crop(box).resize((320, 320), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def decode_mask(encoded, size):
    if not isinstance(encoded, str) or not encoded.startswith("data:image/png;base64,"):
        raise ValueError("Expected a PNG mask")
    raw = base64.b64decode(encoded.split(",", 1)[1], validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.size != (size, size):
            raise ValueError("Mask dimensions changed")
        pixels = np.asarray(image.convert("L"))
    # Canvas antialiasing is explicitly resolved at 128; no spatial resize/filter.
    return (pixels >= 128).astype(np.uint8)


def save_boundary(folder, sample_id, payload):
    _, scene = scene_entry(folder, sample_id)
    meta, review = read_json(scene / "metadata.json"), read_json(scene / "review.json")
    threshold = payload.get("threshold_rad", review["threshold_rad"])
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("Invalid threshold")
    with np.load(scene / "arrays.npz", allow_pickle=False) as data:
        mask = candidate_mask(data["deformation_phase_rad"], threshold)
    manual = payload.get("mask_png")
    if manual is not None:
        mask = decode_mask(manual, meta["config"]["size"])
    previous_sha = sha(scene / "mask.png")
    history_path = scene / "mask_history" / f"{previous_sha}.png"
    history_path.parent.mkdir(exist_ok=True)
    if not history_path.exists():
        history_path.write_bytes((scene / "mask.png").read_bytes())
    Image.fromarray(mask * 255).save(scene / "mask.png")
    for style in ("bmp", "research"):
        with Image.open(scene / f"{style}.png") as image:
            rgb = np.asarray(image.convert("RGB"))
        Image.fromarray(overlay(rgb, mask)).save(scene / f"{style}_overlay.png")
    review.update(
        status="pending",
        threshold_rad=float(threshold),
        manual_mask=manual is not None,
        note=str(payload.get("note", review["note"]))[:2000],
    )
    review["history"].append(
        {
            "at": now(),
            "action": "boundary_edited",
            "manual": manual is not None,
            "threshold_rad": float(threshold),
            "mask_sha256": sha(scene / "mask.png"),
            "previous_mask_file": history_path.relative_to(scene).as_posix(),
        }
    )
    review.pop("accepted_receipt", None)
    meta["label_percent"] = float(mask.mean() * 100)
    meta["label_status"] = "candidate; human review pending"
    write_json(scene / "metadata.json", meta)
    write_json(scene / "review.json", review)
    return review


def decide(folder, sample_id, payload):
    _, scene = scene_entry(folder, sample_id)
    status = payload.get("status")
    if status not in ("accepted", "rejected", "pending"):
        raise ValueError("Invalid review decision")
    review = read_json(scene / "review.json")
    meta = read_json(scene / "metadata.json")
    note = str(payload.get("note", ""))[:2000]
    event = {
        "at": now(),
        "action": status,
        "note": note,
        "mask_sha256": sha(scene / "mask.png"),
        "image_sha256": sha(scene / f"{meta['config']['style']}.png"),
    }
    review.update(status=status, note=note)
    review["history"].append(event)
    if status == "accepted":
        review["accepted_receipt"] = event
    else:
        review.pop("accepted_receipt", None)
    meta["label_status"] = "human accepted" if status == "accepted" else status
    write_json(scene / "metadata.json", meta)
    write_json(scene / "review.json", review)
    return review


def regenerate(folder, sample_id, config):
    manifest = read_json(folder / "manifest.json")
    entry, old = scene_entry(folder, sample_id)
    c = validate(config)
    revision = entry["revision"] + 1
    name = f"{sample_id}_r{revision}"
    arrays, meta = simulate(c)
    meta["sample_id"] = sample_id
    save_scene(folder / name, arrays, meta, read_json(folder / "display_profile.json"))
    write_json(
        folder / name / "review.json",
        {
            "status": "pending",
            "revision": revision,
            "threshold_rad": c["boundary_rad"],
            "manual_mask": False,
            "note": "",
            "history": [],
        },
    )
    for row in manifest["samples"]:
        if row["id"] == sample_id:
            row.update(folder=name, revision=revision)
    write_json(folder / "manifest.json", manifest)
    return {"id": sample_id, "revision": revision, "previous_saved": old.name}


def labelme_doc(sample_id, mask, size):
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8)).save(buf, format="PNG")
    shapes = []
    if mask.any():
        shapes = [
            {
                "label": "1",
                "points": [[0.0, 0.0], [float(size - 1), float(size - 1)]],
                "group_id": None,
                "description": "complete subsidence range, exact raster union",
                "shape_type": "mask",
                "flags": {},
                "mask": base64.b64encode(buf.getvalue()).decode(),
            }
        ]
    return {
        "version": "5.8.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": f"{sample_id}.png",
        "imageData": None,
        "imageHeight": size,
        "imageWidth": size,
    }


def export_archive(folder, draft=False):
    rows = []
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as output:
        for entry in read_json(folder / "manifest.json")["samples"]:
            scene = folder / entry["folder"]
            meta, review = read_json(scene / "metadata.json"), read_json(scene / "review.json")
            if not draft and review["status"] != "accepted":
                continue
            sample_id, style = entry["id"], meta["config"]["style"]
            if not draft:
                receipt = review["accepted_receipt"]
                if receipt["mask_sha256"] != sha(scene / "mask.png") or receipt[
                    "image_sha256"
                ] != sha(scene / f"{style}.png"):
                    raise ValueError("Accepted data changed; review this sample again")
            with Image.open(scene / "mask.png") as image:
                mask = (np.asarray(image) > 0).astype(np.uint8)
            # Exactly one selected style per source; alternate view retained outside dataset.
            output.write(scene / f"{style}.png", f"images/{sample_id}.png")
            output.write(scene / "mask.png", f"masks/{sample_id}.png")
            output.write(scene / "arrays.npz", f"arrays/{sample_id}.npz")
            label_meta = {
                **meta,
                "review": review,
                "label_rule": "manual exact raster boundary"
                if review["manual_mask"]
                else f"abs(clean deformation phase) >= {review['threshold_rad']} rad",
                "label_status": "draft" if draft else "human accepted",
            }
            output.writestr(
                f"metadata/{sample_id}.json", json.dumps(label_meta, ensure_ascii=False, indent=2)
            )
            output.write(scene / f"{style}.png", f"labelme/{sample_id}.png")
            output.writestr(
                f"labelme/{sample_id}.json",
                json.dumps(
                    labelme_doc(sample_id, mask, meta["config"]["size"]),
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            rows.append(
                {
                    "id": sample_id,
                    "group_id": meta["group_id"],
                    "source": "synthetic",
                    "split": "unassigned",
                    "status": "draft" if draft else "accepted",
                    "image": f"images/{sample_id}.png",
                    "mask": f"masks/{sample_id}.png",
                    "image_sha256": sha(scene / f"{style}.png"),
                    "mask_sha256": sha(scene / "mask.png"),
                    "style": style,
                    "revision": entry["revision"],
                }
            )
        if not rows:
            raise ValueError("还没有确认的样本。先逐张确认影像和边界，或下载标明待审核的预览包。")
        manifest = {
            "engine": ENGINE,
            "status": "draft" if draft else "reviewed",
            "created_at": now(),
            "samples": rows,
            "split_policy": "assign by group_id; alternate views/derived patches of a source stay in one split; real validation dates/sites remain independent",
            "use": "synthetic supplement, no arbitrary-real-image generalization claim",
        }
        output.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        output.writestr(
            "README.md",
            "# 合成分割数据\n\n状态："
            + ("待审核预览，禁止当作已确认标签报告。" if draft else "逐张经用户确认。")
            + "\n\nimages 为 RGB PNG；masks 为同尺寸 0/255 单通道 PNG，读取后转 0/1。"
            "arrays 保存 float32 形变/相位/DEM/观测量。metadata 保存参数、显示适配、审核记录。"
            "labelme 保存同一精确栅格掩膜，不填洞、不丢小区域。\n\n"
            "split 尚未指定。先按 group_id 划分，再裁片；同源的替代显示和增强不得跨划分。"
            "真实验证影像不得用于色表/统计拟合。混合训练时另行记录真实与合成比例。"
            "这 20 张是外观验收，不是充分的训练集或真实泛化验证。\n",
        )
        output.writestr("load_dataset.py", LOADER)
        output.writestr(
            "display_profile.json",
            json.dumps(read_json(folder / "display_profile.json"), ensure_ascii=False, indent=2),
        )
    return stream.getvalue(), manifest


LOADER = '''"""Minimal dataset adapter; no training or automatic split assignment."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class SyntheticRanges(Dataset):
    def __init__(self, root, allow_draft=False):
        self.root = Path(root)
        doc = json.loads((self.root / "manifest.json").read_text())
        if doc["status"] != "reviewed" and not allow_draft:
            raise ValueError("Human review pending; draft data requires explicit opt-in")
        self.rows = doc["samples"]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(self.root / row["image"]) as im:
            x = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
        with Image.open(self.root / row["mask"]) as im:
            y = (np.asarray(im) > 0).astype(np.float32)
        if x.shape[:2] != y.shape:
            raise ValueError("Image/mask alignment error")
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(y[None].copy()), row["group_id"]
'''


def handler(run):
    class Handler(BaseHTTPRequestHandler):
        def allowed(self):
            host = self.headers.get("Host", "").split(":")[0]
            return host in ("127.0.0.1", "localhost")

        def respond(self, value, kind="application/json; charset=utf-8", code=200, filename=None):
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False).encode()
            elif isinstance(value, str):
                value = value.encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(value)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(value)

        def do_GET(self):
            if not self.allowed():
                return self.respond({"error": "Local access only"}, code=403)
            route = unquote(urlsplit(self.path).path)
            try:
                with LOCK:
                    if route == "/api/summary":
                        return self.respond(summary(run))
                    if route.startswith("/api/sample/"):
                        _, scene = scene_entry(run, route.rsplit("/", 1)[-1])
                        return self.respond(read_json(scene / "metadata.json"))
                    if route.startswith("/download/"):
                        draft = route == "/download/draft"
                        if route not in ("/download/draft", "/download/reviewed"):
                            raise ValueError("Unknown export")
                        data, _ = export_archive(run, draft)
                        return self.respond(
                            data,
                            "application/zip",
                            filename="mining-simulator-draft.zip"
                            if draft
                            else "mining-simulator-reviewed.zip",
                        )
                    if route.startswith("/scene/"):
                        parts = route.strip("/").split("/")
                        if len(parts) != 3:
                            raise ValueError("Invalid scene path")
                        _, scene = scene_entry(run, parts[1])
                        name = parts[2]
                        if name.startswith("crop_"):
                            allowed = {
                                f"crop_{style}_{kind}.png": (style, kind)
                                for style in ("bmp", "research")
                                for kind in ("input", "wide", "current", "tight")
                            }
                            if name not in allowed:
                                raise ValueError("Unknown boundary preview")
                            return self.respond(
                                boundary_preview(scene, *allowed[name]), "image/png"
                            )
                        if name not in (
                            "bmp.png",
                            "research.png",
                            "bmp_overlay.png",
                            "research_overlay.png",
                            "mask.png",
                            "dem_m.png",
                            "estimated_coherence.png",
                            "delta_down_m.png",
                        ):
                            raise ValueError("Unknown image")
                        return self.respond((scene / name).read_bytes(), "image/png")
                    name = "index.html" if route == "/" else route.lstrip("/")
                    if name not in ("index.html", "app.js", "style.css"):
                        return self.respond({"error": "Not found"}, code=404)
                    return self.respond(
                        (WEB / name).read_bytes(), mimetypes.guess_type(name)[0] or "text/plain"
                    )
            except (ValueError, KeyError, FileNotFoundError) as error:
                self.respond({"error": str(error)}, code=400)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if not self.allowed() or (
                origin and origin not in (f"http://{self.headers.get('Host')}",)
            ):
                return self.respond({"error": "Local same-origin requests only"}, code=403)
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 2_000_000:
                    raise ValueError("Request too large or empty")
                payload = json.loads(self.rfile.read(length))
                parts = urlsplit(self.path).path.strip("/").split("/")
                if len(parts) != 3 or parts[0] != "api":
                    raise ValueError("Unknown action")
                with LOCK:
                    if parts[1] == "boundary":
                        result = save_boundary(run, parts[2], payload)
                    elif parts[1] == "decision":
                        result = decide(run, parts[2], payload)
                    elif parts[1] == "regenerate":
                        result = regenerate(run, parts[2], payload)
                    else:
                        raise ValueError("Unknown action")
                self.respond(result)
            except (ValueError, KeyError, TypeError) as error:
                self.respond({"error": str(error)}, code=400)

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build")
    build.add_argument("--run", type=Path, required=True)
    build.add_argument("--bmp", type=Path)
    build.add_argument("--seed", type=int, default=20261009)
    serve = sub.add_parser("serve")
    serve.add_argument("--run", type=Path, required=True)
    serve.add_argument("--port", type=int, default=8781)
    serve.add_argument("--open", action="store_true", help="Open the local review page")
    args = parser.parse_args()
    if args.action == "build":
        build_batch(args.run, args.bmp, args.seed)
    else:
        run = args.run.resolve()
        read_json(run / "manifest.json")
        url = f"http://127.0.0.1:{args.port}/"
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(run))
        print(f"Simulator review: {url}", flush=True)
        if args.open:
            import webbrowser

            webbrowser.open(url)
        server.serve_forever()


if __name__ == "__main__":
    main()
