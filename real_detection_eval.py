"""Frozen whole-scene inference, exact final per-file LabelMe evaluation."""

import argparse
import html
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from annotation_audit import decode_annotation_mask, read_annotations, sha256
from audit_report import image_uri
from detect_scene import ScoreLogits
from detection_first import FringeDetector
from detection_study import candidates
from phase_encoding import decode_display_palette
from segmentation import mask_boundary, metrics_from_counts, tile_starts, tiled_probability


def threshold_counts(score, truth):
    pred = score >= 0.25
    return [
        int((pred & truth).sum()),
        int((pred & ~truth).sum()),
        int((~pred & truth).sum()),
        int((~pred & ~truth).sum()),
    ]


def object_coverage(score, doc):
    values = []
    for shape in doc["shapes"]:
        mask = decode_annotation_mask(dict(doc, shapes=[shape]))
        values.append(float(np.mean(score[mask] >= 0.25)) if mask.any() else 0.0)
    return values


def evaluate(args):
    if args.out.exists():
        raise ValueError("Choose a new run directory")
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    _, _, _, records, manifest = read_annotations(args.labels)
    with Image.open(args.source) as original:
        if original.size != (manifest["width"], manifest["height"]):
            raise ValueError("Source dimensions differ")
        phase, visible, adapter = decode_display_palette(original)
        rgb = original.convert("RGB")
    for row in records:
        with Image.open(args.labels / row["image"]) as saved:
            if not np.array_equal(
                np.asarray(saved.convert("RGB")), np.asarray(rgb.crop(row["bbox"]))
            ):
                raise ValueError("Annotation image and source pixels differ")
    channels = np.stack([np.sin(phase), np.cos(phase)]).astype(np.float32)
    channels[:, ~visible] = 0
    del phase, visible
    checkpoints = dict(previous=args.previous, current=args.current)
    protocol = dict(
        source_name=args.source.name,
        source_sha256=sha256(args.source),
        manifest_sha256=sha256(args.labels / "manifest.json"),
        annotations=records,
        checkpoints={k: sha256(v) for k, v in checkpoints.items()},
        input_adapter=adapter,
        inference="whole original image, native 256-pixel tiles, stride192, tapered overlap mean",
        threshold=0.25,
        candidate_min_area=64,
        candidate_padding=24,
        evaluation="each saved final annotation tile separately; overlaps counted per tile; unannotated tiles excluded",
        status="previously used real development date; not an untouched test",
        scope="no training, threshold selection, label edits or phase calibration",
    )
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    results = {}
    total = len(tile_starts(channels.shape[1], 256, 192)) * len(
        tile_starts(channels.shape[2], 256, 192)
    )
    for name, checkpoint in checkpoints.items():
        cp = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = FringeDetector(**cp.get("model_config", dict(width=cp.get("width", 8))))
        model.load_state_dict(cp["state_dict"])
        started = time.monotonic()

        def progress(done):
            if done % 100 == 0 or done == total:
                print(json.dumps(dict(model=name, completed=done, total=total)), flush=True)

        score = tiled_probability(
            ScoreLogits(model),
            channels,
            torch.device("cpu"),
            tile_size=256,
            stride=192,
            batch_size=4,
            progress=progress,
        )
        np.save(args.out / f"{name}_score.npy", score)
        totals = np.zeros(4, dtype=np.int64)
        objects, tile_results = [], []
        for row in records:
            doc = json.loads((args.labels / row["file"]).read_text())
            truth = decode_annotation_mask(doc)
            x, y, r, b = row["bbox"]
            local = score[y:b, x:r]
            counts = threshold_counts(local, truth)
            totals += counts
            covered = object_coverage(local, doc)
            objects.extend(covered)
            windows = candidates(local)
            touched = 0
            for box in windows:
                a, c, d, e = box["bbox"]
                touched += bool(truth[c:e, a:d].any())
            tile_results.append(
                dict(
                    file=row["file"],
                    counts=counts,
                    object_coverage=covered,
                    windows=windows,
                    windows_touching_label=touched,
                )
            )
        metrics = metrics_from_counts(totals.tolist())
        metrics.update(
            annotation_objects=len(objects),
            objects_any_overlap=sum(v > 0 for v in objects),
            objects_half_covered=sum(v >= 0.5 for v in objects),
            candidate_windows=sum(len(r["windows"]) for r in tile_results),
            candidate_windows_without_label=sum(
                len(r["windows"]) - r["windows_touching_label"] for r in tile_results
            ),
        )
        results[name] = dict(
            metrics=metrics, tiles=tile_results, seconds=time.monotonic() - started
        )
        (args.out / f"{name}_metrics.json").write_text(json.dumps(results[name], indent=2))
        print(json.dumps(dict(model=name, metrics=metrics)), flush=True)
        del score, model
    if sha256(args.source) != protocol["source_sha256"] or any(
        sha256(args.labels / r["file"]) != r["annotation_sha256"] for r in records
    ):
        raise ValueError("Source changed during evaluation")
    if any(sha256(path) != protocol["checkpoints"][name] for name, path in checkpoints.items()):
        raise ValueError("Checkpoint changed during evaluation")
    report = dict(protocol=protocol, results=results, sources_unchanged=True)
    (args.out / "results.json").write_text(json.dumps(report, indent=2))
    write_report(args.out, args.labels, rgb)
    return report


def write_report(output, labels, rgb):
    report = json.loads((output / "results.json").read_text())
    records = report["protocol"]["annotations"]
    scores = {k: np.load(output / f"{k}_score.npy", mmap_mode="r") for k in ("previous", "current")}
    headers = []

    def percent(value):
        return "无预测，未定义" if value is None else f"{value:.2%}"

    for name, title in (("previous", "上一版"), ("current", "第二轮模型")):
        m = report["results"][name]["metrics"]
        headers.append(
            f"<tr><td>{title}</td><td>{percent(m['precision'])}</td><td>{percent(m['recall'])}</td><td>{m['objects_half_covered']}/{m['annotation_objects']}</td><td>{m['candidate_windows_without_label']}/{m['candidate_windows']}</td></tr>"
        )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>真实干涉图 · 冻结检测对照</title><style>body{background:#101923;color:#e4ebf3;font:16px system-ui;max-width:1200px;margin:24px auto;padding:0 16px}a,summary{color:#7ee3ca}p{line-height:1.7}article{background:#1a2838;padding:12px;margin:20px 0;border-radius:14px}.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px}figure{margin:0}img{width:100%;height:auto}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #456}.table{overflow:auto}h1{font-size:28px}</style><a href="/">返回模拟器</a><h1 id="top">真实干涉图 · 冻结检测对照</h1>"""
    page += f"<p>{html.escape(report['protocol']['source_name'])} · 全图原始分辨率推理 · {len(records)} 份最终标注参与评价</p>"
    for name, title in (("previous", "上一版"), ("current", "第二轮模型")):
        m = report["results"][name]["metrics"]
        page += f"<p><strong>{title}</strong>：标注像素覆盖 {percent(m['recall'])}；{m['objects_half_covered']}/{m['annotation_objects']} 条标注覆盖达到一半；{m['candidate_windows_without_label']}/{m['candidate_windows']} 个候选框未碰到标注。</p>"
    page += (
        '<p>这是此前已用过的真实开发日期。两版模型和阈值保持冻结，没有训练或修改标注。下列数值不能与合成图的“最高分位置命中”直接比较。</p><div class="table"><table><tr><th>模型</th><th>高分像素精确率</th><th>标注像素覆盖率</th><th>覆盖≥一半的标注对象</th><th>未碰到标注的候选框</th></tr>'
        + "".join(headers)
        + "</table></div>"
    )
    page += """<details><summary>指标口径与限制</summary><p>高分阈值固定0.25；热图不是最终分割，像素精确率/覆盖率仅用于诊断。对象按原始LabelMe shape记录，保留小对象；重叠切片中的记录分别计数，不推断真实矿区总数。候选框采用64像素最小高分区域与24像素外扩；框碰到标注只是宽松命中，不能说明边界正确。未标注切片不计入精度；亮度为零处仍保留人工标注。GAMMA调色板转换是显示角度代理，不是恢复物理相位。</p></details><h2>全部标注切片</h2><p>绿：你的最终范围；黄：候选框；灰度热图固定黑=0、白=1。页面缩略图仅用于浏览，模型推理没有缩小原图。</p>"""
    for i, row in enumerate(records):
        x, y, r, b = row["bbox"]
        image = rgb.crop(row["bbox"])
        doc = json.loads((labels / row["file"]).read_text())
        truth = decode_annotation_mask(doc)
        pixels = np.asarray(image).copy()
        pixels[mask_boundary(truth)] = [50, 255, 140]
        annotated = Image.fromarray(pixels)
        panels = [("原始影像", image), ("最终标注", annotated)]
        for name, title in (("previous", "上一版"), ("current", "第二轮模型")):
            overlay = annotated.copy()
            draw = ImageDraw.Draw(overlay)
            for box in report["results"][name]["tiles"][i]["windows"]:
                draw.rectangle(box["bbox"], outline="#ffd65c", width=2)
            panels.extend(
                [
                    (title + "候选框", overlay),
                    (
                        title + "热图",
                        Image.fromarray(np.round(scores[name][y:b, x:r] * 255).astype(np.uint8)),
                    ),
                ]
            )
        page += f'<article><h3>{html.escape(row["image"])}</h3><p>{row["shapes"]} 个标注记录 · {row["positive_pixels"]} 个标注像素</p><div class="panels">'
        for title, panel in panels:
            panel.thumbnail((512, 512))
            page += f'<figure><figcaption>{title}</figcaption><img loading="lazy" src="{image_uri(panel)}" alt="{title}"></figure>'
        page += "</div></article>"
    (output / "index.html").write_text(page + "</html>", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "labels", "previous", "current", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    evaluate(parser.parse_args())
