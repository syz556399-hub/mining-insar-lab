"""Controlled continuation of the independent detection-first experiment."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from annotation_audit import connected_regions
from data_io import phase_rgb
from detection_first import FringeDetector, detection_target, make_scene, panel, panel_html
from mine_model import simulate
from phase_encoding import quantized_phase
from provenance import fingerprint, runtime_versions

DILATIONS = (1, 2, 4, 8, 16, 8, 4, 2, 1)
DIFFICULTIES = ("clear", "medium", "hard")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(output):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for split, count, offset in (("train", 96, 0), ("val", 24, 1000), ("test", 24, 2000)):
        for i in range(count):
            seed = 20263000 + offset + i
            difficulty = DIFFICULTIES[(i // 4) % 3]
            negative = i % 4 == 3
            arrays, metadata = make_scene(seed, i, negative)
            request = metadata["request"]
            rng = np.random.default_rng(seed + 99)
            settings = request["settings"]
            # Identical nuisance distributions for positive and negative classes.
            settings["orbit_cycles"] = float(rng.choice([-1, 1]) * rng.uniform(2, 9))
            if difficulty != "clear":
                hard = difficulty == "hard"
                settings.update(
                    atmosphere_mm=float(rng.uniform(4, 9) if hard else rng.uniform(2, 5)),
                    land_coherence=float(
                        rng.uniform(0.65, 0.82) if hard else rng.uniform(0.84, 0.94)
                    ),
                    looks=4 if hard else 8,
                    relief_m=200.0 if hard else 100.0,
                    dem_error_m=10.0 if hard else 4.0,
                    water_enabled=bool(rng.random() < (0.6 if hard else 0.25)),
                    water_mode="random",
                    disturbed_patch=hard,
                )
            arrays, metadata = simulate(request)
            # Change the orientation of the entire scene, not just a label/image.
            turns = int(rng.integers(4))
            keep = ("wrapped_phase_rad", "deformation_phase_rad", "fringe_mask", "water_mask")
            saved = {key: np.rot90(arrays[key], turns).copy() for key in keep}
            saved["detection_target"] = detection_target(saved["deformation_phase_rad"])
            metadata["pixel_transform"] = dict(
                rot90=turns, scope="all saved arrays after physical simulation"
            )
            name = f"{split}_{i:03d}"
            np.savez_compressed(output / f"{name}.npz", **saved)
            (output / f"{name}.json").write_text(json.dumps(metadata, indent=2))
            if split != "train":
                panel(saved).save(output / f"{name}.png")
            rows.append(
                dict(
                    name=name,
                    split=split,
                    negative=negative,
                    difficulty=difficulty,
                    group_id=metadata["group_id"],
                    sha256=file_hash(output / f"{name}.npz"),
                )
            )
            if len(rows) % 12 == 0:
                print(json.dumps(dict(generated=len(rows), total=144)), flush=True)
    if len({r["group_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate physical scene")
    manifest = dict(
        schema="detection-study-2",
        samples=rows,
        runtime=runtime_versions(),
        physical_engine="unchanged; nuisance parameter changes and aligned rotations",
        augmentation="training only: rotations/reflections, phase origin/sign, optional 16-bin quantization",
        evaluation="fixed 16-bin display angle, numerical phase proxy; not calibrated GAMMA phase",
    )
    manifest["fingerprint"] = fingerprint(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_dataset(output):
    manifest = json.loads((output / "manifest.json").read_text())
    content = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if fingerprint(content) != manifest["fingerprint"]:
        raise ValueError("Manifest checksum mismatch")
    groups = {}
    partitions = {key: [] for key in ("train", "val", "test")}
    for row in manifest["samples"]:
        if row["group_id"] in groups:
            raise ValueError("Repeated underlying physical scene")
        groups[row["group_id"]] = row["split"]
        path = output / f"{row['name']}.npz"
        if file_hash(path) != row["sha256"]:
            raise ValueError("Sample checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            partitions[row["split"]].append(
                (data["wrapped_phase_rad"].copy(), data["detection_target"].copy(), row)
            )
    if any(not part for part in partitions.values()):
        raise ValueError("All splits must be nonempty")
    return manifest, partitions


def encode(phase):
    return np.stack([np.sin(phase), np.cos(phase)]).astype(np.float32)


def augment(phase, target, rng):
    turns = int(rng.integers(4))
    phase, target = np.rot90(phase, turns), np.rot90(target, turns)
    if rng.random() < 0.5:
        phase, target = phase[:, ::-1], target[:, ::-1]
    phase = phase * rng.choice([-1, 1]) + rng.uniform(-np.pi, np.pi)
    phase = quantized_phase(phase, 16 if rng.random() < 0.5 else 0)
    return encode(phase), np.ascontiguousarray(target[None])


def regression_loss(prediction, target):
    fit = ((prediction - target).square() * (1 + 4 * target)).mean()
    # Penalize disagreement with the target gradient, not all legitimate edges.
    gradient = sum(
        F.l1_loss(torch.diff(prediction, dim=d), torch.diff(target, dim=d)) for d in (-2, -1)
    )
    return fit + 0.2 * gradient


def candidates(score, threshold=0.25, min_area=64, padding=24):
    """Candidate crop windows only; neither final segmentation nor instance count."""
    height, width = score.shape
    result = []
    for region in connected_regions(score >= threshold):
        if region["area_px"] < min_area:
            continue
        x1, y1, x2, y2 = region["bbox"]
        peak = max(float(score[y, left:right].max()) for y, left, right in region["runs"])
        result.append(
            dict(
                bbox=[
                    max(0, x1 - padding),
                    max(0, y1 - padding),
                    min(width, x2 + padding),
                    min(height, y2 + padding),
                ],
                score=peak,
                support_pixels=region["area_px"],
            )
        )
    return result


@torch.inference_mode()
def evaluate(model, samples, return_predictions=False):
    model.eval()
    records, predictions = [], {}
    for phase, target, row in samples:
        x = torch.from_numpy(encode(quantized_phase(phase, 16)))[None]
        prediction = model(x)[0, 0].numpy()
        y, xpeak = np.unravel_index(np.argmax(prediction), prediction.shape)
        record = dict(
            name=row["name"],
            difficulty=row["difficulty"],
            negative=row["negative"],
            wmse=float(np.mean((prediction - target) ** 2 * (1 + 4 * target))),
            zero_wmse=float(np.mean(target**2 * (1 + 4 * target))),
            peak=float(prediction.max()),
            peak_hit=bool(prediction.max() >= 0.25 and target[y, xpeak] >= 0.25),
            false_alarm=bool(row["negative"] and len(candidates(prediction)) > 0),
            texture_error=float(
                sum(
                    np.abs(np.diff(prediction, axis=d) - np.diff(target, axis=d)).mean()
                    for d in (0, 1)
                )
            ),
        )
        records.append(record)
        if return_predictions:
            predictions[row["name"]] = prediction
    summaries = {}
    for difficulty in ("all", *DIFFICULTIES):
        subset = [r for r in records if difficulty == "all" or r["difficulty"] == difficulty]
        if not subset:
            continue
        positive = [r for r in subset if not r["negative"]]
        negative = [r for r in subset if r["negative"]]
        summaries[difficulty] = dict(
            count=len(subset),
            wmse=float(np.mean([r["wmse"] for r in subset])),
            zero_wmse=float(np.mean([r["zero_wmse"] for r in subset])),
            positive_peak_hits=sum(r["peak_hit"] for r in positive),
            positives=len(positive),
            negative_false_alarms=sum(r["false_alarm"] for r in negative),
            negatives=len(negative),
            texture_error=float(np.mean([r["texture_error"] for r in subset])),
        )
    return dict(summary=summaries, records=records), predictions


def train(output, baseline, epochs=16):
    if epochs < 1 or (output / "detector_v2.pt").exists():
        raise ValueError("Positive epochs and new output required")
    torch.set_num_threads(2)
    torch.manual_seed(23)
    manifest, data = load_dataset(output)
    # Fixed before training and before reading test metrics.
    protocol = dict(
        epochs=epochs,
        seed=23,
        model=dict(width=12, dilations=DILATIONS),
        batch_size=4,
        lr=0.001,
        lr_decay="cosine",
        threshold=0.25,
        min_area=64,
        selection="minimum validation weighted MSE; earliest tie",
        loss="weighted MSE + 0.2 target-gradient L1; independent design",
        baseline_sha256=file_hash(baseline),
        dataset=manifest["fingerprint"],
        comparison="combined model/data/training update; not a single-factor ablation",
        scope="synthetic only; no true instance counts or real accuracy",
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    model = FringeDetector(width=12, dilations=DILATIONS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
    rng = np.random.default_rng(23)
    history, best = [], float("inf")
    for epoch in range(epochs):
        model.train()
        losses = []
        order = rng.permutation(len(data["train"]))
        for start in range(0, len(order), 4):
            batch = [augment(*data["train"][i][:2], rng) for i in order[start : start + 4]]
            x, target = (torch.from_numpy(np.stack(items)) for items in zip(*batch))
            optimizer.zero_grad()
            loss = regression_loss(model(x), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val, _ = evaluate(model, data["val"])
        score = val["summary"]["all"]["wmse"]
        history.append(
            dict(
                epoch=epoch + 1,
                train_loss=float(np.mean(losses)),
                val_wmse=score,
                val=val["summary"],
                lr=optimizer.param_groups[0]["lr"],
            )
        )
        if score < best:
            best = score
            torch.save(
                dict(
                    state_dict=model.state_dict(),
                    model_config=protocol["model"],
                    epoch=epoch + 1,
                    protocol=protocol,
                ),
                output / "detector_v2.pt",
            )
        scheduler.step()
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps({k: v for k, v in history[-1].items() if k != "val"}), flush=True)
    checkpoint = torch.load(output / "detector_v2.pt", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    final, new_predictions = evaluate(model, data["test"], True)
    old = torch.load(baseline, weights_only=True)
    old_model = FringeDetector(old["width"])
    old_model.load_state_dict(old["state_dict"])
    initial, old_predictions = evaluate(old_model, data["test"], True)
    result = dict(
        baseline=initial, candidate=final, best_epoch=checkpoint["epoch"], protocol=protocol
    )
    (output / "comparison.json").write_text(json.dumps(result, indent=2))
    for phase, target, row in data["test"]:
        name = row["name"]
        prediction = new_predictions[name]
        np.save(output / f"{name}_prediction.npy", prediction)
        boxes = candidates(prediction)
        (output / f"{name}_candidates.json").write_text(json.dumps(boxes, indent=2))
        rgb = Image.fromarray(phase_rgb(quantized_phase(phase, 16)))
        board = Image.new("RGB", (1024, 284), "#111c29")
        draw = ImageDraw.Draw(board)
        for j, (title, image) in enumerate(
            zip(
                ("INPUT", "TARGET", "PREVIOUS", "CURRENT"),
                (
                    rgb,
                    *(
                        Image.fromarray(np.round(a * 255).astype(np.uint8))
                        for a in (target, old_predictions[name], prediction)
                    ),
                ),
            )
        ):
            draw.text((j * 256 + 8, 8), title, fill="white")
            board.paste(image, (j * 256, 28))
        board.save(output / f"{name}_comparison.png")
        overlay = rgb.copy()
        draw = ImageDraw.Draw(overlay)
        for j, box in enumerate(boxes):
            draw.rectangle(box["bbox"], outline="#fff080", width=2)
            rgb.crop(box["bbox"]).save(output / f"{name}_crop_{j:02d}.png")
        overlay.save(output / f"{name}_candidates.png")
    write_report(output)
    return result


def write_report(output):
    result = json.loads((output / "comparison.json").read_text())
    labels = dict(all="全部", clear="清晰", medium="中等干扰", hard="强干扰")
    lines = []
    for difficulty in labels:
        a, b = (result[key]["summary"][difficulty] for key in ("baseline", "candidate"))
        lines.append(
            f"<tr><td>{labels[difficulty]}</td><td>{a['wmse']:.4f} → {b['wmse']:.4f}</td><td>{a['positive_peak_hits']}/{a['positives']} → {b['positive_peak_hits']}/{b['positives']}</td><td>{a['negative_false_alarms']}/{a['negatives']} → {b['negative_false_alarms']}/{b['negatives']}</td></tr>"
        )
    html = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>沉降检测 · 第二轮对照</title>
<style>body{background:#101923;color:#e4ebf3;font:16px system-ui;margin:24px auto;max-width:1100px;padding:0 16px}a,summary{color:#7ee3ca}p{line-height:1.7}article{background:#1a2838;padding:12px;margin:20px 0;border-radius:14px}img{width:100%;height:auto}.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:8px}figure{margin:0}figcaption{font-size:14px;margin:6px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #456;text-align:left}.table{overflow:auto}h1{font-size:28px}</style>
<a href="/">返回模拟器</a><h1>沉降检测 · 第二轮对照</h1><p>144 张合成图 · 清晰 / 中等 / 强干扰 · 新旧模型同图比较</p>
<p>96 张训练、24 张选模型、24 张最终测试。全部测试图都在下方，不只展示成功样本。</p>
<div class="table"><table><tr><th>难度</th><th>加权误差 ↓</th><th>正样本定位命中 ↑</th><th>负样本误报 ↓</th></tr>"""
    html += "".join(lines) + "</table></div>"
    html += f"<p>表中为上一版 → 本版；最佳模型来自第 {result['best_epoch']} 轮。仅为合成测试，未证明真实矿区泛化。</p>"
    html += """<details><summary>方法、判据和限制</summary><p>独立实现全分辨率残差空洞卷积，扩大上下文。训练加入旋转、镜像、相位起点与正负号变化及16档显示量化；验证与测试固定16档。监督仍为干净形变相位绝对值归一化；损失增加与目标梯度的一致性。新旧对比包含数据、结构和训练的共同变化，不能单独归因于某一个修改。</p><p>定位命中：整幅最高分≥0.25，且位于目标强度≥0.25区域；不是每个盆地的召回率。负样本误报：出现得分≥0.25且面积≥64像素的连通区域。裁框向外扩24像素，仅供后续分析，不是最终边界或真实矿区个数。所有阈值在训练前固定。</p><p>水体、地形为合成场景；颜色编码适配不等于恢复GAMMA物理相位。未使用真实标签训练，没有复制作者代码，尚未做相位解缠。</p><p>参考：<a href="https://doi.org/10.3390/rs15092310">Wang、Zhang、Wu（2023）</a>；<a href="https://github.com/Wu-Patrick/Deformation-Monitoring-Dev">作者训练仓库</a>。</p></details><h2 id="predictions">全部留出样本</h2><p>目标、旧预测和新预测均固定黑=0、白=1，没有逐图拉伸。</p>"""
    for row in result["candidate"]["records"]:
        name = row["name"]
        title = "背景负样本" if row["negative"] else "沉降正样本"
        html += f"<article><h3>{name} · {labels[row['difficulty']]} · {title}</h3>"
        html += panel_html(
            output / f"{name}_comparison.png", ["输入", "训练目标", "上一版", "本版"]
        )
        boxes = json.loads((output / f"{name}_candidates.json").read_text())
        encoded = base64.b64encode((output / f"{name}_candidates.png").read_bytes()).decode()
        html += f'<details><summary>候选区域与自动裁块：{len(boxes)} 个窗口</summary><p>黄色矩形是后续分析窗口，不是最终分割边界。</p><div class="panels"><figure><figcaption>候选窗口</figcaption><img src="data:image/png;base64,{encoded}" alt="候选窗口"></figure>'
        for j in range(len(boxes)):
            encoded = base64.b64encode((output / f"{name}_crop_{j:02d}.png").read_bytes()).decode()
            html += f'<figure><figcaption>裁块 {j + 1}</figcaption><img src="data:image/png;base64,{encoded}" alt="分析裁块"></figure>'
        html += "</div></details>"
        html += "</article>"
    (output / "index.html").write_text(html + "</html>", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "train"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--epochs", type=int, default=16)
    args = parser.parse_args()
    if args.action == "generate":
        generate(args.out)
    elif args.baseline is None:
        parser.error("train requires --baseline")
    else:
        train(args.out, args.baseline, args.epochs)
