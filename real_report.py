"""Show final LabelMe targets and matched before/after development predictions."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from annotation_audit import sha256
from audit_report import image_uri
from real_labels import FinalLabelScene
from segmentation import SmallUNet, mask_boundary


@torch.inference_mode()
def block_probability(model, channels, size, batch_size=8):
    model.eval()
    _, height, width = channels.shape
    result = np.zeros((height, width), np.float32)
    blocks = []

    def flush():
        outputs = model(torch.from_numpy(np.stack([b[0] for b in blocks]))).sigmoid()[:, 0].numpy()
        for output, (_, y, x, h, w) in zip(outputs, blocks):
            result[y : y + h, x : x + w] = output[:h, :w]
        blocks.clear()

    for y in range(0, height, size):
        for x in range(0, width, size):
            crop = channels[:, y : y + size, x : x + size]
            h, w = crop.shape[-2:]
            crop = np.pad(crop, ((0, 0), (0, size - h), (0, size - w)), mode="edge")
            blocks.append((crop, y, x, h, w))
            if len(blocks) == batch_size:
                flush()
    if blocks:
        flush()
    return result


def draw_mask(image, mask):
    pixels = np.asarray(image).copy()
    pixels[mask] = (pixels[mask] * 0.6 + np.array([30, 230, 130]) * 0.4).astype(np.uint8)
    pixels[mask_boundary(mask)] = [255, 220, 50]
    return Image.fromarray(pixels)


def report(args):
    config = json.loads((args.run / "config.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    if sha256(args.initial) != config["init_checkpoint_sha256"]:
        raise ValueError("Initial checkpoint differs")
    scene = FinalLabelScene(args.labels, args.source)
    if scene.provenance != config["splits"]["val"]:
        raise ValueError("Validation data changed")
    torch.set_num_threads(2)
    initial = torch.load(args.initial, map_location="cpu", weights_only=True)
    selected = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    models = []
    for cp in [initial, selected]:
        model = SmallUNet(**cp["model_config"])
        model.load_state_dict(cp["state_dict"])
        models.append(model)
    ranked = sorted(
        range(len(scene.tiles)),
        key=lambda i: (scene.tiles[i]["target"].sum(), scene.tiles[i]["record"]["file"]),
    )
    selected_ids = [
        ranked[i] for i in np.linspace(0, len(ranked) - 1, min(6, len(ranked)), dtype=int)
    ]
    cards = []
    with Image.open(args.source) as source:
        rgb = source.convert("RGB")
    for i in selected_ids:
        tile = scene.tiles[i]
        row = tile["record"]
        image = rgb.crop(row["bbox"])
        outputs = [
            block_probability(model, tile["channels"], config["crop_size"]) for model in models
        ]
        panels = [
            ("原图", image),
            ("你的最终标注", draw_mask(image, tile["target"])),
            ("纯合成预训练", draw_mask(image, outputs[0] >= 0.5)),
            ("真实标注微调后", draw_mask(image, outputs[1] >= 0.5)),
        ]
        figures = "".join(
            f'<figure><figcaption>{title}</figcaption><img alt="{title}" src="{image_uri(img)}"></figure>'
            for title, img in panels
        )
        cards.append(
            f'<article><h2>{html.escape(row["image"])}</h2><p>原始标注对象：{len(tile["object_ids"])}；掩膜面积：{int(tile["target"].sum())} 像素。黄线为边界，绿色为区域。</p><label><input class="native" type="checkbox"> 原始像素查看（横向滚动）</label><div class="panels">{figures}</div></article>'
        )

    def pct(x):
        return "—" if x is None else f"{x * 100:.2f}%"

    rows = []
    for name, metric in [
        ("纯合成预训练", summary["baseline"]),
        ("微调后选定模型", summary["best_finetuned_or_baseline"]),
    ]:
        rows.append(
            "<tr><td>"
            + name
            + "</td>"
            + "".join(
                f"<td>{pct(metric[k])}</td>" for k in ["foreground_iou", "precision", "recall"]
            )
            + "</tr>"
        )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>最终标注训练对照</title><style>body{background:#101c29;color:#e6eef5;font:16px/1.65 system-ui;margin:0}main{max-width:1200px;margin:auto;padding:22px}header,article{background:#192a3a;padding:22px;border-radius:16px;margin:18px 0;border:1px solid #375167}h1{font-size:27px}h2{font-size:18px;overflow-wrap:anywhere}p{color:#c4d5e2}a{color:#70ddd0}table{width:100%;border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #375167;text-align:left}.scroll{overflow:auto}.panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}figure{margin:0;overflow:auto}img{display:block;width:100%;margin-top:8px}.full img{width:auto;max-width:none}.notice{border-left:4px solid #efb766;padding-left:12px}@media(max-width:600px){.panels{grid-template-columns:1fr}main{padding:12px}header,article{padding:16px}}</style><main><header><a href="/">返回模拟器</a><h1>最终标注 · 训练对照</h1><p>以你的 LabelMe 为最终标准：保留范围、孔洞、小目标及每条对象记录。没有自动填补、扩张、删除或合并标注对象。</p>SUMMARY<div class="scroll"><table><tr><th>模型</th><th>IoU</th><th>精确率</th><th>召回率</th></tr>ROWS</table></div><p class="notice">同矿区跨日期的开发验证结果，不是独立测试。按每张已保存的标注切片评价，重叠视图分别计入，因此不能直接与此前整景去重指标比较。语义分割学习区域；尚没有独立的目标计数模型。</p><p>以下 6 张按标注面积固定抽取，不按模型效果筛选。未使用四月保留数据。</p></header>CARDS</main><script>for(const c of document.querySelectorAll('article')){c.querySelector('.native').onchange=e=>c.querySelector('.panels').classList.toggle('full',e.target.checked)}</script></html>"""
    train, val = config["splits"]["train"], config["splits"]["val"]
    detail = f"<p>训练：{train['annotation_files']} 张切片、{train['annotation_objects']} 条对象记录；验证：{val['annotation_files']} 张切片、{val['annotation_objects']} 条对象记录。跨切片对象可能重复，不据此推算整景独立矿区个数。</p><p>原始像素输入，{config['crop_size']} × {config['crop_size']} 裁块；{config['epochs']} 轮微调，选定第 {summary['best_epoch']} 轮，阈值固定 0.5。未标注区域只在已保存的最终标注切片内作为背景。</p>"
    (args.run / "index.html").write_text(
        page.replace("SUMMARY", detail)
        .replace("ROWS", "".join(rows))
        .replace("CARDS", "".join(cards)),
        encoding="utf-8",
    )
    (args.run / "preview_selection.json").write_text(
        json.dumps([scene.tiles[i]["record"] for i in selected_ids], indent=2)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["run", "initial", "labels", "source"]:
        p.add_argument("--" + name, type=Path, required=True)
    report(p.parse_args())
