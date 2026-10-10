"""Independent detection-first pilot: PIM scenes -> full-resolution heatmap regression.

Inspired by the task separation of Wu et al., TGRS 2022, not a reproduction.
No external implementation or weights are imported. This is not phase unwrapping.
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

from data_io import magnitude_rgb, phase_rgb, png
from mine_model import default_request, simulate
from provenance import fingerprint, runtime_versions


class FringeDetector(nn.Module):
    """Residual dilated convolutions without pooling or image resizing."""

    def __init__(self, width=8, dilations=(1, 2, 4, 8, 4, 2, 1)):
        super().__init__()
        if width < 4 or width % 4:
            raise ValueError("width must be a multiple of four")
        if not dilations or any(type(d) is not int or d < 1 for d in dilations):
            raise ValueError("dilations must be positive integers")
        self.stem = nn.Conv2d(2, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width, width, 3, padding=d, dilation=d),
                nn.GroupNorm(4, width),
                nn.SiLU(),
            )
            for d in dilations
        )
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = x + block(x) * 0.5
        return self.head(x).sigmoid()


def detection_target(clean_phase):
    """Relative clean deformation magnitude; not a calibrated probability/mask."""
    phase = np.asarray(clean_phase, dtype=np.float32)
    if phase.ndim != 2 or not np.isfinite(phase).all():
        raise ValueError("Expected finite 2D phase")
    amplitude = np.abs(phase)
    peak = float(amplitude.max())
    return amplitude / peak if peak > 1e-6 else np.zeros_like(amplitude)


def clear_request(seed, index, negative=False):
    rng = np.random.default_rng(seed)
    request = default_request()
    settings = request["settings"]
    settings.update(
        seed=seed,
        size=256,
        extent_m=float(rng.uniform(1100, 1600)),
        day_before=65.0,
        day_after=89.0,
        orbit_cycles=float(rng.uniform(3, 6)),
        atmosphere_mm=float(rng.uniform(1.2, 3)),
        atmosphere_scale_m=float(rng.uniform(250, 600)),
        water_enabled=False,
        relief_m=50.0,
        dem_error_m=1.0,
        height_atmosphere_mm=0.3,
        land_coherence=0.98,
        decorrelation_days=10000.0,
        slope_coherence_loss=0.05,
        spatial_window=1,
        looks=12,
        horizontal_factor=0.1,
    )
    cx, cy = rng.uniform(-140, 140, 2)
    separation = float(rng.uniform(90, 170))
    angle = float(rng.uniform(-np.pi, np.pi))
    # Physical working faces vary in location and orientation; no RGB compositing.
    for j, face in enumerate(request["faces"]):
        sign = 2 * j - 1
        face.update(
            enabled=not negative and (j == 0 or index % 4 != 0),
            east_m=float(cx + sign * separation * np.cos(angle) / 2),
            north_m=float(cy + sign * separation * np.sin(angle) / 2),
            length_m=float(rng.uniform(200, 360)),
            width_m=float(rng.uniform(90, 150)),
            bearing_deg=float(rng.uniform(-175, 175)),
            depth_m=float(rng.uniform(170, 240)),
            thickness_m=float(rng.uniform(1.5, 2.5)),
            influence_tangent=float(rng.uniform(1.8, 2.4)),
            advance_m_day=7.0,
            start_day=float(rng.uniform(0, 10)),
            response_days=70.0,
        )
    return request


def make_scene(seed, index, negative=False):
    request = clear_request(seed, index, negative)
    arrays, meta = simulate(request)
    if not negative:
        # Conditional physical sampling: change thickness, then rerun the model.
        # Never scale the displayed RGB or paint a target into observed phase.
        target_cycles = float(np.random.default_rng(seed + 17).uniform(3, 6))
        ratio = target_cycles / max(meta["phase_cycles"], 1e-6)
        for face in request["faces"]:
            face["thickness_m"] = float(np.clip(face["thickness_m"] * ratio, 0.1, 8))
        arrays, meta = simulate(request)
        if not 2 <= meta["phase_cycles"] <= 8:
            raise ValueError("Scene failed the clean deformation cycle requirement")
        area = float(arrays["fringe_mask"].mean())
        if not 0.015 <= area <= 0.4:
            raise ValueError("Scene failed the support area requirement")
    arrays["detection_target"] = detection_target(arrays["deformation_phase_rad"])
    meta["detection_target_rule"] = (
        "abs(clean deformation phase) / scene maximum; zero for negative"
    )
    meta["pilot_profile"] = "clear-positive-v1; conditional selection, not population sampling"
    return arrays, meta


def panel(arrays):
    board = Image.new("RGB", (1024, 284), "#111c29")
    draw = ImageDraw.Draw(board)
    for i, (name, values) in enumerate(
        [
            ("INPUT", phase_rgb(arrays["wrapped_phase_rad"])),
            ("CLEAN DEFORMATION", phase_rgb(arrays["deformation_phase_rad"])),
            ("DETECTION TARGET", magnitude_rgb(arrays["detection_target"])),
            ("EXPERIMENTAL SUPPORT", arrays["fringe_mask"] * 255),
        ]
    ):
        draw.text((i * 256 + 8, 8), name, fill="white")
        board.paste(Image.fromarray(values), (i * 256, 28))
    return board


def generate(output, seed=20261019):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for i in range(28):
        negative = i >= 20
        arrays, metadata = make_scene(seed + i, i, negative)
        split = (
            ("train" if i < 12 else "val" if i < 16 else "test")
            if not negative
            else ("train" if i < 24 else "val" if i < 26 else "test")
        )
        name = f"scene_{i:02d}"
        np.savez_compressed(output / f"{name}.npz", **arrays)
        (output / f"{name}.json").write_text(json.dumps(metadata, indent=2))
        panel(arrays).save(output / f"{name}.png")
        rows.append(
            dict(
                name=name,
                split=split,
                negative=negative,
                group_id=metadata["group_id"],
                cycles=metadata["phase_cycles"],
            )
        )
    if len({r["group_id"] for r in rows}) != len(rows):
        raise ValueError("Repeated physical scene geometry")
    manifest = dict(
        schema="detection-first-pilot-1",
        seed=seed,
        samples=rows,
        runtime=runtime_versions(),
        input="sin/cos numeric wrapped phase",
        split_rule="unique independent physical scene per split; no real labels used",
        target="normalized clean deformation magnitude; not mask or probability",
        limitations="Clear synthetic pilot only; no real-data validation or unwrapping",
    )
    manifest["fingerprint"] = fingerprint(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_gallery(output)
    return manifest


def train_pilot(output, epochs=6):
    if epochs < 1 or (output / "detector.pt").exists():
        raise ValueError("Positive epochs and a new checkpoint required")
    torch.set_num_threads(2)
    torch.manual_seed(19)
    manifest = json.loads((output / "manifest.json").read_text())
    partitions = {key: [] for key in ("train", "val", "test")}
    for row in manifest["samples"]:
        with np.load(output / f"{row['name']}.npz") as data:
            phase = data["wrapped_phase_rad"]
            x = torch.from_numpy(np.stack([np.sin(phase), np.cos(phase)]))
            y = torch.from_numpy(data["detection_target"][None])
        partitions[row["split"]].append((x, y))
    model = FringeDetector()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    rng = torch.Generator().manual_seed(20)
    history, best = [], float("inf")

    def evaluate(split):
        model.eval()
        errors, zeros = [], []
        with torch.no_grad():
            for x, y in partitions[split]:
                weight = 1 + 4 * y
                errors.append(float(((model(x[None])[0] - y).square() * weight).mean()))
                zeros.append(float((y.square() * weight).mean()))
        return float(np.mean(errors)), float(np.mean(zeros))

    for epoch in range(epochs):
        model.train()
        losses = []
        order = torch.randperm(len(partitions["train"]), generator=rng).tolist()
        for start in range(0, len(order), 2):
            samples = [partitions["train"][i] for i in order[start : start + 2]]
            x, y = [torch.stack(items) for items in zip(*samples)]
            optimizer.zero_grad()
            loss = ((model(x) - y).square() * (1 + 4 * y)).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val, zero = evaluate("val")
        history.append(
            dict(
                epoch=epoch + 1, train_loss=float(np.mean(losses)), val_loss=val, zero_baseline=zero
            )
        )
        if val < best:
            best = val
            torch.save(
                dict(
                    state_dict=model.state_dict(),
                    architecture="FringeDetector-v1",
                    width=8,
                    epoch=epoch + 1,
                    manifest=manifest["fingerprint"],
                    target=manifest["target"],
                    input=manifest["input"],
                ),
                output / "detector.pt",
            )
        print(json.dumps(history[-1]), flush=True)
    saved = torch.load(output / "detector.pt", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    test, zero = evaluate("test")
    result = dict(
        history=history,
        best_epoch=saved["epoch"],
        test_loss=test,
        test_zero_baseline=zero,
        beats_zero_baseline=test < zero,
        loss="mean((prediction-target)^2 * (1+4*target)); independent design",
        status="pipeline pilot; not a validated detector",
        real_data_used=False,
    )
    (output / "training.json").write_text(json.dumps(result, indent=2))
    save_predictions(output)
    write_gallery(output)
    return result


@torch.inference_mode()
def save_predictions(output):
    """All held-out examples, fixed 0..1 display scale; never per-image stretched."""
    manifest = json.loads((output / "manifest.json").read_text())
    saved = torch.load(output / "detector.pt", weights_only=True)
    if saved["manifest"] != manifest["fingerprint"]:
        raise ValueError("Checkpoint and dataset differ")
    model = FringeDetector(saved["width"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    for row in manifest["samples"]:
        if row["split"] != "test":
            continue
        with np.load(output / f"{row['name']}.npz") as data:
            phase = data["wrapped_phase_rad"]
            x = torch.from_numpy(np.stack([np.sin(phase), np.cos(phase)]))[None]
            predicted = model(x)[0, 0].numpy()
            target = data["detection_target"]
        np.save(output / f"{row['name']}_prediction.npy", predicted)
        board = Image.new("RGB", (768, 284), "#111c29")
        draw = ImageDraw.Draw(board)
        for i, (title, array) in enumerate(
            [
                ("HELD-OUT INPUT", phase_rgb(phase)),
                ("TARGET 0..1", np.round(target * 255).astype(np.uint8)),
                ("PREDICTED SCORE 0..1", np.round(predicted * 255).astype(np.uint8)),
            ]
        ):
            draw.text((i * 256 + 8, 8), title, fill="white")
            board.paste(Image.fromarray(array), (i * 256, 28))
        board.save(output / f"{row['name']}_prediction.png")


def panel_html(path, titles):
    with Image.open(path) as board:
        elements = []
        for i, title in enumerate(titles):
            crop = np.asarray(board.crop((256 * i, 28, 256 * (i + 1), 284)))
            encoded = base64.b64encode(png(crop)).decode()
            elements.append(
                f'<figure><figcaption>{title}</figcaption><img src="data:image/png;base64,{encoded}" alt="{title}"></figure>'
            )
    return '<div class="panels">' + "".join(elements) + "</div>"


def write_gallery(output):
    manifest = json.loads((output / "manifest.json").read_text())
    cards = []
    for row in manifest["samples"]:
        panels = panel_html(
            output / f"{row['name']}.png",
            ["输入干涉图", "干净形变", "检测训练目标", "实验范围标签"],
        )
        title = "背景负样本" if row["negative"] else "清晰沉降正样本"
        cards.append(
            f'<article data-negative="{str(row["negative"]).lower()}"><h3>{row["name"]} · {title}</h3>'
            f"<p>净形变 {row['cycles']:.1f} 圈 · {row['split']}</p>"
            f"{panels}</article>"
        )
    training = "尚未训练"
    if (output / "training.json").exists():
        result = json.loads((output / "training.json").read_text())
        training = (
            f"流程试验完成：{len(result['history'])} 轮；独立合成测试加权误差 "
            f"{result['test_loss']:.5f}，全零基线 {result['test_zero_baseline']:.5f}。"
            "这不是真实矿区精度，尚未进行真实数据验证。"
        )
    html = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>先检测，再分析 · 清晰沉降样本</title><style>body{background:#101923;color:#e4ebf3;font:16px system-ui;margin:24px auto;max-width:1100px;padding:0 16px}a{color:#7ee3ca}p{line-height:1.7}article{background:#1a2838;padding:12px;margin:20px 0;border-radius:14px}img{width:100%;height:auto}button{padding:12px;border-radius:8px;cursor:pointer}h1{font-size:30px}.note{border-left:4px solid #7ee3ca;padding:8px 18px}.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:8px}figure{margin:0}figcaption{font-size:14px;margin:6px 0}summary{cursor:pointer;color:#7ee3ca}</style>
<a href="/">返回模拟器</a><h1>先检测，再分析</h1><p>20 张清晰正样本 · 8 张独立背景负样本 · 自主实现</p>
<p>先看生成图，再看模型能否找到沉降。热图用于找位置，尚不等同于完整范围分割。</p>
<details><summary>查看方法与当前限制</summary>
<p class="note">热图是相对形变强度，不能当作发生概率；范围标签仍是净相位阈值，尚未等同于你的 LabelMe 完整条纹边界。</p>
<p>先用模拟图训练保留原始分辨率的网络，学习沉降位置，再评估真实干涉图。此页是清晰样本起步版；合成地形已保留，水体和强干扰留到后续难度组。没有修改真实标注，没有导入他人代码或权重。</p>
<p>方法参考：<a href="https://doi.org/10.1109/TGRS.2021.3121907">Wu、Wang 等，TGRS 2022</a>。沿用任务思路，不宣称复现 DDNet 或 PUNet。</p>
"""
    html += f"<p>{training}</p></details><p><a href='#predictions'>查看模型测试结果 ↓</a></p><button onclick=\"document.querySelectorAll('[data-negative=true]').forEach(e=>e.hidden=!e.hidden)\">显示 / 隐藏背景负样本</button>"
    html += "".join(cards)
    html += "<h2 id='predictions'>全部独立合成测试样本</h2><p>目标和预测均固定黑=0、白=1，未逐图拉伸亮度。仅代表本次小规模试验。</p>"
    for row in manifest["samples"]:
        path = output / f"{row['name']}_prediction.png"
        if row["split"] == "test" and path.exists():
            html += (
                f"<article><h3>{row['name']}</h3>"
                + panel_html(path, ["独立测试输入", "目标热图", "模型预测热图"])
                + "</article>"
            )
    html += "</html>"
    (output / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "train"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    if args.action == "generate":
        generate(args.out)
    else:
        train_pilot(args.out, args.epochs)
