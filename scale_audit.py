"""Frozen-model display-scale comparison against provisional annotations."""

import argparse
import html
import json
from pathlib import Path

import numpy as np

from annotation_audit import connected_regions, read_annotations, sha256


def scores(prediction, truth, scope):
    tp = int((prediction & truth & scope).sum())
    fp = int((prediction & ~truth & scope).sum())
    fn = int((~prediction & truth & scope).sum())

    def ratio(a, b):
        return a / b if b else None

    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        iou=ratio(tp, tp + fp + fn),
        precision=ratio(tp, tp + fp),
        recall=ratio(tp, tp + fn),
    )


def compare(labels, source, runs, output):
    if (output / "scale_comparison.json").exists():
        raise ValueError("Comparison already exists")
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    digest = sha256(source)
    truth, scope, conflicts, records, _ = read_annotations(labels)
    if digest != audit["source_sha256"] or records != audit["annotation_files"]:
        raise ValueError("Audit source or annotations changed")
    groups = connected_regions(truth)
    sizes = {"small": (0, 256), "medium": (256, 4096), "large": (4096, float("inf"))}
    results, checkpoint = [], None
    for run in runs:
        info = json.loads((run / "prediction.json").read_text(encoding="utf-8"))
        probability = np.load(run / "probability.npy", mmap_mode="r", allow_pickle=False)
        if info["source_sha256"] != digest or probability.shape != truth.shape:
            raise ValueError("Source or coordinate mismatch")
        if checkpoint is None:
            checkpoint = info["checkpoint_sha256"]
        if info["checkpoint_sha256"] != checkpoint or info["threshold"] != 0.5:
            raise ValueError("Comparison requires one frozen checkpoint and threshold 0.5")
        if info["input_mode"] != "phase" or info["phase_bins"] != 16 or len(info["scales"]) != 1:
            raise ValueError("Expected identical 16-bin phase preprocessing with one scale")
        scale = info["scales"][0]
        if scale in [r["scale"] for r in results]:
            raise ValueError("Duplicate comparison scale")
        prediction = probability >= 0.5
        bins = {}
        for name, (lower, upper) in sizes.items():
            chosen = [g for g in groups if lower <= g["area_px"] < upper]
            area, hits, covered = 0, 0, 0
            for group in chosen:
                hit = sum(int(prediction[y, left:right].sum()) for y, left, right in group["runs"])
                area += group["area_px"]
                hits += hit
                covered += hit >= 0.5 * group["area_px"]
            bins[name] = dict(
                regions=len(chosen),
                positive_pixels=area,
                covered_positive_pixels=hits,
                positive_pixel_recall=hits / area if area else None,
                regions_at_least_half_covered=covered,
            )
        results.append(
            dict(
                scale=scale,
                inference_batch_size=info.get("inference_batch_size", 1),
                prediction_metadata_sha256=sha256(run / "prediction.json"),
                saved_tile_scope=scores(prediction, truth, scope),
                excluding_overlap_conflicts=scores(prediction, truth, scope & ~conflicts),
                size_bins=bins,
            )
        )
    results.sort(key=lambda row: row["scale"])
    report = dict(
        schema="scale-audit-1",
        audit_id=audit["audit_id"],
        source_sha256=digest,
        checkpoint_sha256=checkpoint,
        threshold=0.5,
        training_performed=False,
        interpretation="Development-set annotation agreement, not verified real accuracy. Unmarked background and overlap conflicts require review. Size bins are pixel areas, not physical extents or mine instances.",
        results=results,
    )
    (output / "scale_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def pct(value):
        return "—" if value is None else f"{100 * value:.2f}%"

    rows = []
    for row in results:
        m = row["saved_tile_scope"]
        cells = [
            f"原图 ÷ {row['scale']}",
            pct(m["iou"]),
            pct(m["precision"]),
            pct(m["recall"]),
            pct(row["size_bins"]["small"]["positive_pixel_recall"]),
        ]
        rows.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
    section = (
        """<div id="scale-results"><h2>同一个模型，改变输入尺度</h2><p>阈值固定 0.5；没有重新训练。以下是已保存标注范围内的一致性诊断，不能当作真实精度。小区域定义为标注连通区域少于 256 像素。</p><div class="scroll"><table><tr><th>输入尺度</th><th>IoU</th><th>精确率</th><th>召回率</th><th>小区域像素召回率</th></tr>"""
        + "".join(rows)
        + "</table></div><p>详细记录另列剔除重叠标注差异区域后的结果；这同样不代表已完成标签核实。未按这些结果选择最终测试配置。</p></div>"
    )
    page = output / "index.html"
    content = page.read_text(encoding="utf-8")
    marker = '<div id="scale-results"></div>'
    if content.count(marker) != 1:
        raise ValueError("Expected an unmodified audit page")
    page.write_text(content.replace(marker, section), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.labels, args.source, args.run, args.out), ensure_ascii=False))
