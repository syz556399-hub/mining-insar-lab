"""Controlled synthetic-only experiment report; real labels are evaluation only."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from audit_report import image_uri
from data_io import phase_rgb
from provenance import fingerprint
from validate_dataset import audit_splits, rows_at


def dataset_stats(root):
    values = []
    for row in rows_at(root):
        with np.load(root / row["arrays"], allow_pickle=False) as data:
            values.append(
                dict(
                    split=row["split"],
                    physical_area=int(data["mask"].sum()),
                    support_area=int(data["fringe_mask"].sum()),
                )
            )
    areas = np.array([v["support_area"] for v in values])
    positive = areas[areas > 0]
    return dict(
        scenes=len(values),
        positive_fraction=float(np.mean(areas > 0)),
        support_area_quantiles=np.quantile(positive, [0.1, 0.5, 0.9]).tolist()
        if len(positive)
        else [],
        split_audit=audit_splits([root]),
        rows=values,
    )


def evaluate(args):
    import torch
    from torch.utils.data import DataLoader

    from annotation_audit import sha256
    from real_labels import FinalLabelScene, RealCrops
    from segmentation import SmallUNet
    from train import run_epoch

    scene = FinalLabelScene(args.labels, args.source)
    profile = json.loads(args.profile.read_text())
    validation_hashes = {r["annotation_sha256"] for r in scene.provenance["annotations"]}
    if validation_hashes & {r["sha256"] for r in profile["source_annotations"]}:
        raise ValueError("Design fit labels overlap evaluation labels")
    fit_dates = set().union(
        *(set(re.findall(r"(?<!\d)\d{8}(?!\d)", r["file"])) for r in profile["source_annotations"])
    )
    eval_dates = set(re.findall(r"(?<!\d)\d{8}(?!\d)", args.source.name))
    if not fit_dates or len(eval_dates) != 2 or fit_dates & eval_dates:
        raise ValueError("Design and evaluation need separate identifiable acquisition dates")
    loader = DataLoader(RealCrops(scene, size=256, training=False), batch_size=8)
    torch.set_num_threads(2)
    results = []
    control = None
    arms = [
        ("A_original", args.original),
        ("B_supervision", args.supervision),
        ("C_matched", args.matched_run),
    ]
    if getattr(args, "balanced_run", None):
        arms.append(("D_matched_sampling", args.balanced_run))
    for name, path in arms:
        cp = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
        config = cp["configuration"]
        budget = {
            k: config[k]
            for k in [
                "seed",
                "epochs",
                "batch_size",
                "learning_rate",
                "model_config",
                "phase_bins",
                "phase_augmentation",
            ]
        }
        if control is None:
            control = budget
        if budget != control:
            raise ValueError("Experiment budget or architecture differs")
        if cp["tile_size"] != 256 or cp["input_mode"] != "phase":
            raise ValueError("Expected native 256 phase model")
        model = SmallUNet(**cp["model_config"])
        model.load_state_dict(cp["state_dict"])
        metrics = run_epoch(model, loader, "cpu", 0.5)
        results.append(
            dict(
                arm=name,
                checkpoint_sha256=sha256(path / "best.pt"),
                selected_synthetic_epoch=cp["epoch"],
                metrics=metrics,
            )
        )
    report = dict(
        schema="simulator-study-1",
        training="synthetic images only; no real image gradient updates",
        design_fit="training-date final label statistics only",
        budget=control,
        results=results,
        evaluation=scene.provenance,
        threshold=0.5,
        independent_test_performed=False,
        scope="same-mine cross-date development comparison; overlapping annotation tiles count as separate views; no output suppression at zero brightness",
    )
    (args.out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def write_report(args):
    args.out.mkdir(parents=True, exist_ok=True)
    profile = json.loads(args.profile.read_text())
    stats = {
        name: dataset_stats(root)
        for name, root in [("baseline", args.baseline), ("matched", args.matched)]
    }
    (args.out / "design_statistics.json").write_text(json.dumps(stats, indent=2))
    rows = rows_at(args.baseline)
    new = rows_at(args.matched)
    if len(rows) != len(new):
        raise ValueError("Paired exports must have equal scene counts")
    split_counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    cards = []
    for a, b in zip(rows[:20], new[:20]):
        with (
            np.load(args.baseline / a["arrays"]) as old,
            np.load(args.matched / b["arrays"]) as changed,
        ):
            panels = [
                ("原模拟条纹", phase_rgb(old["wrapped_phase_rad"])),
                ("物理标签", old["mask"] * 255),
                ("实验支持区", old["fringe_mask"] * 255),
                ("匹配后条纹", phase_rgb(changed["wrapped_phase_rad"])),
                ("匹配后支持区", changed["fringe_mask"] * 255),
            ]
            figures = "".join(
                f'<figure><figcaption>{name}</figcaption><img alt="{name}" src="{image_uri(Image.fromarray(array))}"></figure>'
                for name, array in panels
            )
            cards.append(
                f'<article><h2>{a["scene_id"]}</h2><div class="panels">{figures}</div></article>'
            )
    result_path = args.out / "results.json"
    result = "训练进行中。三个模型均从相同随机初始化开始，只使用合成数据。"
    if result_path.exists():
        results = json.loads(result_path.read_text())["results"]
        result = '<div class="scroll"><table><tr><th>实验</th><th>IoU</th><th>精确率</th><th>召回率</th></tr>'
        names = {
            "A_original": "A 原生成＋物理监督",
            "B_supervision": "B 同图＋支持区监督",
            "C_matched": "C 匹配频率与面积＋支持区监督",
            "D_matched_sampling": "D 同C数据＋目标优先抽样",
        }
        for row in results:
            m = row["metrics"]
            result += (
                "<tr><td>"
                + names[row["arm"]]
                + "</td>"
                + "".join(
                    f"<td>{100 * m[k]:.2f}%</td>" if m[k] is not None else "<td>—</td>"
                    for k in ["foreground_iou", "precision", "recall"]
                )
                + "</tr>"
            )
        result += "</table></div><p>固定阈值 0.5，原始像素输入；D 保持 C 的生成数据和验证集不变，仅训练时以 80% 的概率抽取含目标场景，仍每轮抽取 160 次；该组在 C 的合成验证出现退化后追加，未依据真实评价选择。真实开发集仅用于评价。不同指标的得失需一起看；不依据此表调整保留测试集。</p>"
    conclusion = ""
    if result_path.exists():
        baseline_metrics = results[0]["metrics"]
        joint = [
            r
            for r in results[1:]
            if r["metrics"]["foreground_iou"] > baseline_metrics["foreground_iou"]
            and r["metrics"]["recall"] > baseline_metrics["recall"]
        ]
        if not joint:
            conclusion = "<p class=notice><b>本轮未证实改进有效。</b>没有改动组同时提高 IoU 和召回率。统计比例匹配不等于识别效果提升；保留原默认设置，新增功能仅作实验选项。</p>"
    body = (
        f"""<header><a href="/">返回模拟器</a><h1>模拟器改进 · 合成训练对照</h1><p>先改变生成数据，再检验真实识别。对照组使用同一个网络、初始化种子和训练预算，没有真实影像微调。</p><p>开发标注：{profile["blocks"]} 个完整 256×256 裁块，{profile["positive_probability"] * 100:.2f}% 含目标。原生成数据支持区阳性比例 {stats["baseline"]["positive_fraction"] * 100:.1f}%；匹配后 {stats["matched"]["positive_fraction"] * 100:.1f}%。实际生成结果另存，未强行修改掩膜去凑数。</p><p class="notice">新增支持区 = |干净形变相位| ≥ π/2。这是待验证的任务代理，尚不等于人工可见条纹边界。物理沉降标签继续保留；B、C 对全部像素监督，水体和低相干区域不再自动挖除。</p><h2>只用合成数据训练后的真实开发验证</h2>{result}{conclusion}<p>对照组各 {len(rows)} 个独立合成场景，{split_counts["train"]}/{split_counts["val"]}/{split_counts["test"]} 划分；训练预算见实验记录。模型按各自合成验证集选择。面积和频率统计只来自指定开发标注，评价日期与其分离；独立保留测试不在本流程中运行。该实验不能证明跨矿区泛化。</p></header><h2>固定前 20 景生成结果</h2><p>按编号展示，不按视觉效果筛选。B 与 A 使用完全相同的条纹图，只改变标签及监督有效区；C 调整目标出现概率和模拟画幅以改变像素面积。负样本仍有背景条纹。</p>"""
        + "".join(cards)
    )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>模拟器改进实验</title><style>body{margin:0;background:#101c29;color:#e6eef5;font:16px/1.65 system-ui}main{max-width:1380px;padding:24px;margin:auto}header,article{background:#192a3a;padding:22px;margin:18px 0;border-radius:16px;border:1px solid #375167}h1{font-size:28px}h2{font-size:20px}p{color:#c4d5e2}a{color:#70ddd0}.notice{border-left:4px solid #edba73;padding-left:15px}.panels{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;display:block}figcaption{font-size:14px;margin-bottom:7px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #375167;text-align:left}@media(max-width:800px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}main{padding:12px}header,article{padding:16px}}</style><main>BODY</main></html>"""
    (args.out / "index.html").write_text(page.replace("BODY", body), encoding="utf-8")
    return fingerprint(stats)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ["baseline", "matched", "profile", "out"]:
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--balanced-run", type=Path)
    p.add_argument("--evaluate", action="store_true")
    for n in ["labels", "source", "original", "supervision", "matched-run"]:
        p.add_argument("--" + n, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.evaluate:
        if any(
            getattr(args, n) is None
            for n in ["labels", "source", "original", "supervision", "matched_run"]
        ):
            p.error("Evaluation requires real data and all three trained runs")
        evaluate(args)
    write_report(args)
