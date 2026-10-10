"""Portable, local-only review page; review decisions never modify source labels."""

import base64
import html
import io
import json

import numpy as np
from PIL import Image


def image_uri(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def write_audit_report(output, image, truth, conflict, report):
    cards = []
    for i, row in enumerate(report["review_tiles"]):
        x, y, r, b = row["bbox"]
        crop = image.crop((x, y, r, b))
        layer = np.zeros((b - y, r - x, 4), dtype=np.uint8)
        layer[truth[y:b, x:r]] = [30, 255, 120, 160]
        layer[conflict[y:b, x:r]] = [255, 160, 25, 210]
        # Keep source pixels intact; CSS controls display size and overlay visibility.
        cards.append(f'''<article data-index="{i}"><h2>{i + 1:02d} · {html.escape(row["image"])}</h2>
<p>{row["shapes"]} 个标注片段 · 本切片正像素 {row["positive_pixels"]} · 重叠区差异 {row["overlap_disagreement_pixels"]} 像素</p>
<div class="frame"><img alt="原始切片" src="{image_uri(crop)}"><img class="mask" alt="标注叠加" src="{image_uri(Image.fromarray(layer))}"></div>
<label><input class="native-size" type="checkbox"> 按原始像素查看（可横向滚动）</label>
<label><input class="visible" type="checkbox" checked> 显示标注（绿：跨切片标注并集；橙：重叠处不一致）</label>
<label>检查结论 <select class="decision"><option value="pending">待核实</option><option value="complete">标注完整</option><option value="needs_correction">需要修正</option><option value="uncertain">无法确定</option></select></label>
<label><input class="background" type="checkbox"> 已完整核查未标注区域，没有遗漏目标</label>
<label>备注<textarea class="notes" placeholder="例如：漏标、边界不清、中心需要忽略……"></textarea></label></article>''')
    payload = json.dumps(
        {k: report[k] for k in ("audit_id", "source_sha256", "review_tiles")}, ensure_ascii=True
    ).replace("<", "\\u003c")
    summary = report["regions"]
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实条纹标注检查</title>
<style>body{margin:0;background:#101c29;color:#e6eef5;font:16px/1.65 system-ui,sans-serif}main{max-width:1150px;margin:auto;padding:24px}header,article{background:#192a3a;border:1px solid #375167;border-radius:16px;padding:22px;margin:18px 0}h1{font-size:28px}h2{font-size:18px;overflow-wrap:anywhere}p{color:#bcd0df}a{color:#70ddd0}button,select,textarea{font:inherit;border-radius:7px;padding:9px;background:#edf7fa;color:#152a3b;border:1px solid #7296ae}button{cursor:pointer;background:#70ddd0}label{display:block;margin-top:12px}textarea{display:block;box-sizing:border-box;width:100%}.frame{position:relative;max-width:100%;overflow:auto;background:#070c12}.frame img{width:100%;display:block;image-rendering:auto}.frame.native img{width:auto;max-width:none}.frame .mask[hidden]{display:none}.frame .mask{position:absolute;inset:0;pointer-events:none}input{accent-color:#70ddd0}.bar{position:sticky;top:0;z-index:3;background:#101c29ed;padding:12px 0}.notice{border-left:4px solid #efb766;padding-left:14px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #375167;text-align:left}.scroll{overflow:auto}</style>
<main><header><a href="/">返回模拟器</a><h1>真实条纹 · 标注与尺度检查</h1>
<p>目标：完整沉降条纹区域；相连盆地合并；保留确认的小目标。潜在影响区单独表达不确定性。</p>
<p class="notice">当前为待核实标注。未标注像素不自动等于可靠背景。橙色只说明相邻切片标注不一致，不自动判定哪张正确。中心不清楚时先记录，不能从 BMP 推定相干性或水体。</p>
<p>显示影像不等于校准相位。请记录导出软件与命令；数值相位、相干性、像元间距需单独提供。</p>
SUMMARY
<div id="scale-results"></div></header>
<div class="bar"><button id="download">下载本次检查记录</button> <span id="progress"></span><small> · 下载后保存；刷新页面不会自动保留选择</small></div>
CARDS
<p>检查记录只保存判断，不修改 LabelMe 文件。需要修正的边界和忽略区仍须在标注工具中处理；未核实样本不进入微调训练。</p></main>
<script>const audit=PAYLOAD;
const cards=[...document.querySelectorAll('article')];
function update(){document.getElementById('progress').textContent=cards.filter(c=>c.querySelector('.decision').value!=='pending').length+'/'+cards.length+' 已检查'}
for(const c of cards){c.querySelector('.native-size').onchange=e=>c.querySelector('.frame').classList.toggle('native',e.target.checked);c.querySelector('.visible').onchange=e=>c.querySelector('.mask').hidden=!e.target.checked;c.querySelector('.decision').onchange=update}
document.getElementById('download').onclick=()=>{const rows=cards.map((c,i)=>({file:audit.review_tiles[i].file,annotation_sha256:audit.review_tiles[i].annotation_sha256,decision:c.querySelector('.decision').value,background_checked:c.querySelector('.background').checked,notes:c.querySelector('.notes').value}));const blob=new Blob([JSON.stringify({schema:'label-review-1',audit_id:audit.audit_id,source_sha256:audit.source_sha256,created_at:new Date().toISOString(),reviews:rows},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='annotation-review-'+audit.audit_id.slice(0,12)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};update();</script></html>"""
    page = page.replace(
        "SUMMARY",
        f"<p>共 {len(report['annotation_files'])} 个已保存标注文件，合并得到 {summary['count']} 个连通区域（不等于矿区数量）。重叠标注差异：{report['overlap_disagreement_pixels']} 像素。以下 {len(cards)} 张按标注面积选取，不按模型好坏筛选。</p>",
    )
    page = page.replace("CARDS", "\n".join(cards)).replace("PAYLOAD", payload)
    (output / "index.html").write_text(page, encoding="utf-8")
