"""Independent synthetic mining range segmentation with an auxiliary phase task.

No external simulator, network implementation, dataset or weights are imported.
The auxiliary target is clean deformation, not total interferometric unwrapping.
The range target remains an explicitly experimental clean-phase support proxy.
"""

import argparse
import html
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F

from annotation_audit import sha256
from audit_report import image_uri
from data_io import phase_rgb
from mine_model import default_request, simulate
from phase_encoding import quantized_phase
from provenance import fingerprint, runtime_versions
from segmentation import masked_loss, metrics_from_counts, segmentation_overlay
from task_labels import fringe_support

DILATIONS = (1, 2, 4, 8, 16, 8, 4, 2, 1)
CATEGORIES = ("single", "overlap", "separated", "negative")
SIZES = ("large", "medium", "small")
DIFFICULTIES = ("clear", "moderate", "strong")
SPLITS = {"train": 128, "val": 32, "test": 32}
SEED = 20261009
PHASE_SCALE = float(2 * np.pi)


class RangePhaseNet(nn.Module):
    """Full-resolution residual features and independent mask/phase heads."""

    def __init__(self, width=8):
        super().__init__()
        if type(width) is not int or width < 4 or width % 4:
            raise ValueError("width must be an integer multiple of four")
        self.stem = nn.Conv2d(2, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width, width, 3, padding=d, dilation=d),
                nn.GroupNorm(4, width),
                nn.SiLU(),
            )
            for d in DILATIONS
        )
        self.mask_head = nn.Conv2d(width, 1, 1)
        self.phase_head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = x + 0.5 * block(x)
        return self.mask_head(x), self.phase_head(x)


class MaskLogits(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


def request_for(seed, index):
    """Preset pixel scales are experimental ranges, not fitted physical calibration."""
    rng = np.random.default_rng(seed)
    category = CATEGORIES[index % 4]
    size = SIZES[(index // 4) % 3]
    difficulty = DIFFICULTIES[(index // 12) % 3]
    extent = float(
        rng.uniform(*{"large": (850, 1500), "medium": (1500, 3000), "small": (3000, 5500)}[size])
    )
    request = default_request()
    severity = DIFFICULTIES.index(difficulty)
    request["settings"].update(
        seed=seed,
        size=256,
        extent_m=extent,
        segments=24,
        day_before=65.0,
        day_after=89.0,
        horizontal_factor=float(rng.uniform(0, 0.15)),
        incidence_deg=float(rng.uniform(25, 45)),
        orbit_cycles=float(rng.choice([-1, 1]) * rng.uniform(2, 12)),
        atmosphere_mm=float(rng.uniform(*((0.3, 2), (2, 4), (4, 7))[severity])),
        atmosphere_scale_m=float(rng.uniform(0.2, 0.7) * extent),
        relief_m=float(rng.uniform(30, 180)),
        dem_error_m=float(rng.uniform(0, (2, 6, 12)[severity])),
        height_atmosphere_mm=float(rng.uniform(-0.6, 0.6)),
        land_coherence=float(rng.uniform(*((0.94, 0.99), (0.85, 0.95), (0.75, 0.9))[severity])),
        decorrelation_days=10000.0,
        slope_coherence_loss=0.05,
        looks=(8, 6, 4)[severity],
        spatial_window=1,
        water_enabled=bool(rng.random() < 0.25),
        water_mode="random",
        disturbed_patch=bool(severity == 2 and rng.random() < 0.3),
    )
    cx, cy = rng.uniform(-0.32, 0.32, 2) * extent
    separation = float(rng.uniform(60, 160) if category == "overlap" else rng.uniform(300, 500))
    angle = rng.uniform(-np.pi, np.pi)
    for j, face in enumerate(request["faces"]):
        offset = (j - 0.5) * separation if category in ("overlap", "separated") else 0
        face.update(
            enabled=category != "negative" and (j == 0 or category != "single"),
            east_m=float(cx + offset * np.cos(angle)),
            north_m=float(cy + offset * np.sin(angle)),
            length_m=float(rng.uniform(200, 420)),
            width_m=float(rng.uniform(90, 180)),
            bearing_deg=float(rng.uniform(-175, 175)),
            depth_m=float(rng.uniform(140, 280)),
            thickness_m=2.0,
            influence_tangent=float(rng.uniform(1.5, 2.8)),
            advance_m_day=7.0,
            start_day=float(rng.uniform(0, 10)),
            response_days=70.0,
        )
    return request, dict(category=category, size_class=size, difficulty=difficulty)


def generate(output):
    if output.exists():
        raise ValueError("Use a new dataset directory")
    output.mkdir(parents=True)
    rows = []
    for split, count in SPLITS.items():
        for i in range(count):
            index = len(rows)
            request, design = request_for(SEED + 30000 + index, i)
            arrays, metadata = simulate(request)
            if design["category"] != "negative":
                # Change physical thickness and rerun; never scale an RGB image or paint labels.
                desired = float(np.random.default_rng(SEED + index + 900).uniform(1.5, 5.5))
                ratio = desired / max(metadata["phase_cycles"], 1e-6)
                for face in request["faces"]:
                    face["thickness_m"] = float(np.clip(face["thickness_m"] * ratio, 0.1, 8))
                arrays, metadata = simulate(request)
            rng = np.random.default_rng(SEED + index + 600)
            turns = int(rng.integers(4))
            keys = (
                "wrapped_phase_rad",
                "deformation_phase_rad",
                "reference_phase_rad",
                "fringe_mask",
                "water_mask",
                "dem_m",
                "model_coherence",
            )
            saved = {k: np.rot90(arrays[k], turns).copy() for k in keys}
            name = f"{split}_{i:03d}"
            metadata.update(
                design=design, pixel_transform=dict(rot90=turns, scope="all saved arrays")
            )
            np.savez_compressed(output / (name + ".npz"), **saved)
            (output / (name + ".json")).write_text(json.dumps(metadata, indent=2))
            rows.append(
                dict(
                    name=name,
                    split=split,
                    seed=request["settings"]["seed"],
                    **design,
                    group_id=metadata["group_id"],
                    sha256=sha256(output / (name + ".npz")),
                    metadata_sha256=sha256(output / (name + ".json")),
                    positive_pixels=int(saved["fringe_mask"].sum()),
                    cycles=metadata["phase_cycles"],
                    water_percent=metadata["water_percent"],
                )
            )
            if len(rows) % 16 == 0:
                print(json.dumps(dict(generated=len(rows), total=sum(SPLITS.values()))), flush=True)
    if len({r["group_id"] for r in rows}) != len(rows):
        raise ValueError("Repeated physical geometry")
    manifest = dict(
        schema="mining-range-phase-1",
        seed=SEED,
        splits=SPLITS,
        samples=rows,
        runtime=runtime_versions(),
        target="abs(clean deformation phase) >= pi/2; synthetic support proxy, not manual visible boundary",
        input="sin/cos of numeric wrapped phase; validation/test quantized to 16 display codes",
        selection="no aesthetic or positive-area rejection; all sampled scenes retained",
        scope="same independent PIM/complex observation engine; no real inputs or upstream code/weights",
    )
    manifest["fingerprint"] = fingerprint(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_data(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
        != manifest["fingerprint"]
    ):
        raise ValueError("Manifest checksum mismatch")
    groups, data = set(), {s: [] for s in SPLITS}
    for row in manifest["samples"]:
        if row["group_id"] in groups:
            raise ValueError("Repeated physical scene")
        groups.add(row["group_id"])
        for suffix, key in ((".npz", "sha256"), (".json", "metadata_sha256")):
            if sha256(root / (row["name"] + suffix)) != row[key]:
                raise ValueError("Sample checksum mismatch")
        with np.load(root / (row["name"] + ".npz"), allow_pickle=False) as saved:
            phase = saved["wrapped_phase_rad"].copy()
            clean = saved["deformation_phase_rad"].copy()
            target = saved["fringe_mask"].copy().astype(np.float32)
        if (
            phase.shape != clean.shape
            or target.shape != clean.shape
            or not np.isfinite([phase, clean]).all()
        ):
            raise ValueError("Phase shapes/values invalid")
        if not np.array_equal(target, fringe_support(clean)):
            raise ValueError("Range target differs from its declared clean-phase rule")
        data[row["split"]].append((phase, target, clean, row))
    if any(len(data[s]) != manifest["splits"][s] for s in SPLITS):
        raise ValueError("Incomplete split")
    return manifest, data


def encode(phase):
    return np.stack([np.sin(phase), np.cos(phase)]).astype(np.float32)


def augment(phase, target, clean, rng):
    turns = int(rng.integers(4))
    phase, target, clean = (np.rot90(a, turns) for a in (phase, target, clean))
    if rng.random() < 0.5:
        phase, target, clean = (a[:, ::-1] for a in (phase, target, clean))
    sign = rng.choice([-1, 1])
    phase = sign * phase + rng.uniform(-np.pi, np.pi)
    # Sign applies to both observed phase and the signed physical auxiliary target.
    clean = sign * clean
    phase = quantized_phase(phase, 16 if rng.random() < 0.5 else 0)
    return (
        encode(phase),
        np.ascontiguousarray(target[None]),
        np.ascontiguousarray(clean[None] / PHASE_SCALE),
    )


def learning_loss(logits, predicted_phase, target, clean_cycles, phase_weight):
    mask_loss = masked_loss(logits, target, torch.ones_like(target))
    # Keep amplitude information in cycles; no per-image maximum normalization.
    phase_loss = (
        F.smooth_l1_loss(predicted_phase, clean_cycles, reduction="none") * (1 + 4 * target)
    ).mean()
    return mask_loss + phase_weight * phase_loss


@torch.inference_mode()
def evaluate(model, samples, retain=False):
    model.eval()
    counts, rows, predictions = np.zeros(4, np.int64), [], {}
    for phase, target, clean, row in samples:
        x = torch.from_numpy(encode(quantized_phase(phase, 16)))[None]
        logits, reconstructed = model(x)
        probability = logits.sigmoid()[0, 0].numpy()
        reconstructed = reconstructed[0, 0].numpy() * PHASE_SCALE
        pred, truth = probability >= 0.5, target > 0
        local = [
            int((pred & truth).sum()),
            int((pred & ~truth).sum()),
            int((~pred & truth).sum()),
            int((~pred & ~truth).sum()),
        ]
        counts += local
        rows.append(
            dict(
                name=row["name"],
                category=row["category"],
                difficulty=row["difficulty"],
                size_class=row["size_class"],
                metrics=metrics_from_counts(local),
                predicted_pixels=int(pred.sum()),
                positive_pixels=int(truth.sum()),
                phase_rmse_rad=float(np.sqrt(np.mean((reconstructed - clean) ** 2))),
                zero_phase_rmse_rad=float(np.sqrt(np.mean(clean**2))),
            )
        )
        if retain:
            predictions[row["name"]] = (probability, reconstructed)
    metrics = metrics_from_counts(counts.tolist())
    negative = [r for r in rows if r["positive_pixels"] == 0]
    metrics.update(
        negative_scenes=len(negative),
        negative_scenes_with_any_false_positive=sum(r["predicted_pixels"] > 0 for r in negative),
        phase_mean_per_image_rmse_rad=float(np.mean([r["phase_rmse_rad"] for r in rows])),
        zero_phase_mean_per_image_rmse_rad=float(np.mean([r["zero_phase_rmse_rad"] for r in rows])),
    )
    return dict(metrics=metrics, records=rows), predictions


def train(root, epochs=20):
    if epochs < 1 or (root / "results.json").exists() or (root / "protocol.json").exists():
        raise ValueError("Use positive epochs and an untrained dataset")
    torch.set_num_threads(2)
    manifest, data = load_data(root)
    protocol = dict(
        seed=SEED,
        epochs=epochs,
        batch_size=4,
        lr=0.001,
        width=8,
        threshold=0.5,
        selection="maximum synthetic validation foreground IoU; earliest tie",
        dataset=manifest["fingerprint"],
        phase_scale_rad=PHASE_SCALE,
        arms={"range_only": 0.0, "range_and_phase": 0.2},
        comparison="identical initialization, scenes, augmentation order, budget; auxiliary loss weight only differs",
        real_policy="no real image/label gradient updates; previously used March date for later frozen diagnosis only",
        limitations="single seed and small synthetic pilot; not DDNet/PUNet or total phase unwrapping",
    )
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2))
    results = {}
    for arm, weight in protocol["arms"].items():
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        model = RangePhaseNet(protocol["width"])
        optimizer = torch.optim.Adam(model.parameters(), lr=protocol["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=0.0001
        )
        history, best, selected, started = [], -1.0, 0, time.monotonic()
        for epoch in range(epochs):
            model.train()
            losses = []
            order = rng.permutation(len(data["train"]))
            for start in range(0, len(order), protocol["batch_size"]):
                batch = [
                    augment(*data["train"][i][:3], rng)
                    for i in order[start : start + protocol["batch_size"]]
                ]
                x, target, clean = (torch.from_numpy(np.stack(items)) for items in zip(*batch))
                optimizer.zero_grad(set_to_none=True)
                logits, phase = model(x)
                loss = learning_loss(logits, phase, target, clean, weight)
                if not torch.isfinite(loss).item():
                    raise ValueError("Non-finite loss")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            val, _ = evaluate(model, data["val"])
            score = val["metrics"]["foreground_iou"]
            if score is None:
                raise ValueError("Validation has no foreground union")
            if score > best:
                best, selected = score, epoch + 1
                torch.save(
                    dict(
                        model_kind="independent-range-phase-1",
                        state_dict=model.state_dict(),
                        width=protocol["width"],
                        epoch=selected,
                        arm=arm,
                        phase_weight=weight,
                        protocol=protocol,
                        runtime=runtime_versions(),
                    ),
                    root / (arm + ".pt"),
                )
            history.append(
                dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), val=val["metrics"])
            )
            scheduler.step()
            (root / (arm + "_history.json")).write_text(json.dumps(history, indent=2))
            print(
                json.dumps(
                    dict(
                        arm=arm,
                        epoch=epoch + 1,
                        train_loss=history[-1]["train_loss"],
                        val_iou=score,
                    )
                ),
                flush=True,
            )
        model.load_state_dict(torch.load(root / (arm + ".pt"), weights_only=True)["state_dict"])
        result, predictions = evaluate(model, data["test"], True)
        result.update(
            selected_epoch=selected,
            checkpoint_sha256=sha256(root / (arm + ".pt")),
            seconds=time.monotonic() - started,
        )
        results[arm] = result
        np.savez_compressed(
            root / (arm + "_test_predictions.npz"),
            **{
                name + suffix: array
                for name, arrays in predictions.items()
                for suffix, array in zip(("_probability", "_phase_rad"), arrays)
            },
        )
    # Deployment candidate chosen by validation only, never by synthetic/real test scores.
    preferred = max(
        protocol["arms"],
        key=lambda arm: max(
            r["val"]["foreground_iou"]
            for r in json.loads((root / (arm + "_history.json")).read_text())
        ),
    )
    result = dict(
        protocol=protocol,
        arms=results,
        preferred_by_validation=preferred,
        interpretation="synthetic proxy scores only; real boundary transfer remains unproven",
    )
    (root / "results.json").write_text(json.dumps(result, indent=2))
    write_report(root)
    return result


def comparison_panel(phase, target, baseline_probability, probability, reconstructed, clean):
    board = Image.new("RGB", (6 * 256, 284), "#111c29")
    panels = [
        ("OBSERVED", Image.fromarray(phase_rgb(phase))),
        ("SYNTHETIC RANGE", Image.fromarray((target * 255).astype(np.uint8)).convert("RGB")),
        (
            "RANGE ONLY",
            segmentation_overlay(Image.fromarray(phase_rgb(phase)), baseline_probability),
        ),
        ("RANGE + PHASE", segmentation_overlay(Image.fromarray(phase_rgb(phase)), probability)),
        ("CLEAN PHASE TRUTH", Image.fromarray(phase_rgb(clean))),
        ("PHASE PREDICTION", Image.fromarray(phase_rgb(reconstructed))),
    ]
    draw = ImageDraw.Draw(board)
    for i, (title, panel) in enumerate(panels):
        draw.text((i * 256 + 8, 8), title, fill="white")
        board.paste(panel, (i * 256, 28))
    return board


def percentage(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def write_report(root):
    manifest, data = load_data(root)
    results = json.loads((root / "results.json").read_text())
    preferred = results["preferred_by_validation"]
    metrics_rows = ""
    names = {"range_only": "范围分割", "range_and_phase": "范围分割＋相位辅助"}
    for arm, result in results["arms"].items():
        m = result["metrics"]
        values = [
            f"{100 * m[k]:.2f}%" if m[k] is not None else "—"
            for k in ("foreground_iou", "precision", "recall")
        ]
        metrics_rows += (
            f"<tr><td>{names[arm]}</td><td>{result['selected_epoch']}</td>"
            + "".join(f"<td>{v}</td>" for v in values)
            + f"<td>{m['negative_scenes_with_any_false_positive']}/{m['negative_scenes']}</td></tr>"
        )
    cards = []
    with (
        np.load(root / "range_only_test_predictions.npz") as baseline,
        np.load(root / "range_and_phase_test_predictions.npz") as predictions,
    ):
        for phase, target, clean, row in data["test"]:
            probability = predictions[row["name"] + "_probability"]
            reconstructed = predictions[row["name"] + "_phase_rad"]
            panel = comparison_panel(
                phase,
                target,
                baseline[row["name"] + "_probability"],
                probability,
                reconstructed,
                clean,
            )
            panel.save(root / (row["name"] + "_comparison.png"))
            cards.append(
                f'<article><h3>{row["name"]} · {row["category"]} · {row["size_class"]} · {row["difficulty"]}</h3><img alt="输入、合成标签、范围预测、干净相位、相位预测" src="{image_uri(panel)}"></article>'
            )
    real = "<p>真实开发验证尚未执行。</p>"
    if (root / "real_results.json").exists():
        report = json.loads((root / "real_results.json").read_text())
        m = report["metrics"]
        real = f'<p>已在 {len(report["tiles"])} 份原始最终标注切片上验证：IoU {percentage(m["foreground_iou"])}，精确率 {percentage(m["precision"])}，召回率 {percentage(m["recall"])}；至少覆盖一半的标注记录 {m["objects_half_covered"]}/{m["annotation_objects"]}。日期已用于开发，不是独立测试。重叠切片分别计数；不修改标注范围与个数。</p><p><a href="#real-gallery">查看真实对照</a></p>'
        real += (
            '<details id="real-gallery"><summary>展开全部真实影像对照</summary>'
            + "".join(
                f'<article><h3>{html.escape(row["file"])}</h3><img alt="真实影像、最终标注、冻结预测" src="{image_uri(Image.open(root / row["preview"]))}"></article>'
                for row in report["tiles"]
            )
            + "</details>"
        )
    aux = results["arms"]["range_and_phase"]["metrics"]
    aux_note = f"相位辅助输出的合成测试平均逐景 RMSE 为 {aux['phase_mean_per_image_rmse_rad']:.3f} rad；全零基线为 {aux['zero_phase_mean_per_image_rmse_rad']:.3f} rad。相位面板以缠绕颜色展示，不能凭颜色接近判断解缠成功。无辅助训练组的相位头未训练，不报告其相位精度。"
    if aux["phase_mean_per_image_rmse_rad"] >= aux["zero_phase_mean_per_image_rmse_rad"]:
        aux_note += " 本轮相位误差高于全零基线，辅助相位尚未达到可用水平。"
    delta = 100 * (
        aux["foreground_iou"] - results["arms"]["range_only"]["metrics"]["foreground_iou"]
    )
    conclusion = f"<p class=notice>辅助任务使本轮合成 IoU 变化 {delta:+.2f} 个百分点。差异较小，单种子结果不能证明稳定改进。</p>"
    if (root / "real_results.json").exists():
        m = json.loads((root / "real_results.json").read_text())["metrics"]
        conclusion += f"<p class=notice>真实开发验证仍未达到可用程度：标注像素覆盖 {percentage(m['recall'])}，预测精确率 {percentage(m['precision'])}；{m['objects_half_covered']}/{m['annotation_objects']} 条标注覆盖过半。误报和漏检都仍明显，不能以合成分数代替真实效果。</p>"
    body = (
        f"""<header id="top"><a href="/">返回模拟器</a><h1>我的矿区模型 · 范围分割实验</h1><p>自己的模拟器 → 数值相位与范围标签 → 从零训练 → 合成留出验证 → 冻结真实开发验证。</p><p>生成 {len(manifest["samples"])} 景：训练 {SPLITS["train"]}、验证 {SPLITS["val"]}、测试 {SPLITS["test"]}。单盆地、重叠、分离和背景，三档画幅与干扰；DEM、水体保留，水体不是每景必有。</p><p class="notice">这里的范围真值为 |干净形变相位| ≥ π/2，是合成支持区代理。它仍不等于你在 LabelMe 圈出的可见条纹边界，真实标注按原样用于验证。没有导入第三方代码、数据或权重。</p><h2>两个训练方法的对照</h2><p>相同初始化、数据、空间和相位增广、训练预算；只改变相位辅助损失权重。范围直接用 BCE＋Dice 学习；相位辅助为干净形变相位回归，不是总干涉相位解缠。</p><div class="scroll"><table><tr><th>方法</th><th>选择轮次</th><th>IoU</th><th>精确率</th><th>召回率</th><th>背景误报景数</th></tr>{metrics_rows}</table></div><p>按合成验证 IoU 选出的候选为「{names[preferred]}」。阈值固定 0.5，无面积筛除、外扩框或真实标签调参；这是一轮小样本、单种子实验，不能证明真实矿区泛化。</p><p>{aux_note}</p><h2>真实开发验证</h2>{real}</header><h2 id="predictions">全部 {SPLITS["test"]} 景留出结果</h2><p>固定编号顺序展示，未挑图。每图从左到右：模拟输入、合成范围标签、范围单任务预测、范围＋相位联合预测、干净形变相位真值、联合模型的相位预测。范围预测绿填充、黄边线。</p>"""
        + "".join(cards)
    )
    body = body.replace("<h2>两个训练方法的对照</h2>", conclusion + "<h2>两个训练方法的对照</h2>")
    body = body.replace("没有导入第三方代码、数据或权重。", "未导入对方仓库的代码、数据或权重。")
    if (root / "validation_curves.png").is_file():
        learning = (
            "<details><summary>模型怎样学习</summary><p>输入只有观测条纹的 sin/cos 数值。模型逐像素预测范围，与合成范围标签比较后计算误差，再反向传播更新权重；联合组同时比较干净形变相位。每轮结束检查合成验证集，保存最好的轮次。测试和真实标注没有参与权重更新。</p>"
            + f'<img alt="两组在20轮学习中的合成验证IoU" src="{image_uri(Image.open(root / "validation_curves.png"))}">'
            + "</details>"
        )
        body = body.replace("<h2>真实开发验证</h2>", learning + "<h2>真实开发验证</h2>")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>我的矿区模型</title><style>body{margin:0;background:#101c29;color:#e6eef5;font:16px/1.65 system-ui}main{max-width:1380px;padding:24px;margin:auto}header,article{background:#192a3a;padding:22px;margin:18px 0;border-radius:16px;border:1px solid #375167}h1{font-size:28px}h2{font-size:22px}h3{font-size:16px}p{color:#c4d5e2}a{color:#70ddd0}.notice{border-left:4px solid #edba73;padding-left:15px}img{width:100%;display:block}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #375167;text-align:left}@media(max-width:800px){main{padding:12px}header,article{padding:16px}article{overflow:auto}article img{min-width:850px}}</style><main>BODY</main></html>"""
    (root / "index.html").write_text(page.replace("BODY", body), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "train", "report"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.out)
    elif args.command == "train":
        train(args.out, args.epochs)
    else:
        write_report(args.out)
