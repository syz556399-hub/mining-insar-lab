"""Portable, self-contained local viewer for a completed segmentation prediction."""

import base64
import html
import io
import json

import numpy as np
from PIL import Image


def image_url(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def write_report(output, image, probability, info, comparison=None, target=None, valid=None):
    preview = image.convert("RGB").copy()
    preview.thumbnail((1400, 1400))
    scores = Image.fromarray(np.clip(np.round(probability * 255), 0, 255).astype(np.uint8))
    scores = scores.resize(preview.size, Image.Resampling.BILINEAR)
    data = dict(image=image_url(preview), scores=image_url(scores), threshold=info["threshold"])
    scales = "、".join(str(n) for n in info["scales"])
    width, height = image.size
    name = html.escape(info["source_name"])
    source_kind = (
        "合成数值相位" if info["input_kind"] == "numeric_phase" else "索引色影像显示码代理"
    )
    if info["input_kind"] == "rgb":
        source_kind = "彩色影像"
    percent = info["predicted_area_percent"]
    comparison_html = ""
    synthetic_html = ""
    if target is not None:
        truth = Image.fromarray((np.asarray(target) >= 0.5).astype(np.uint8) * 255)
        truth = truth.resize(preview.size, Image.Resampling.NEAREST)
        panels = [
            f'<section class="panel"><h2>模拟器自动标签 · 白色为目标</h2><img src="{image_url(truth)}" alt="合成分割标签"></section>'
        ]
        if valid is not None:
            usable = Image.fromarray((np.asarray(valid) >= 0.5).astype(np.uint8) * 255)
            usable = usable.resize(preview.size, Image.Resampling.NEAREST)
            panels.append(
                f'<section class="panel"><h2>模拟有效区 · 黑色忽略</h2><img src="{image_url(usable)}" alt="模拟有效区"></section>'
            )
        synthetic_html = (
            '<h2>合成测试答案</h2><p>标签与有效区仅用于对照，没有作为预测输入。</p><div class="grid">'
            + "".join(panels)
            + "</div>"
        )
    if comparison is not None:
        rows = []
        for trial in comparison["trials"]:
            metrics = trial["saved_tile_scope"]
            cells = [html.escape(trial["name"])]
            cells.extend(
                f"{100 * metrics[key]:.2f}%" if metrics[key] is not None else "—"
                for key in ("foreground_iou", "precision", "recall")
            )
            rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in cells) + "</tr>")
        caption = html.escape(comparison["caption"])
        picture = image_url(comparison["image"])
        comparison_html = (
            '<section class="note"><b>已有标注范围内的开发对照</b><p>'
            + caption
            + "</p><table><thead><tr><th>实验</th><th>参考 IoU</th><th>精确率</th><th>召回率</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table><details><summary>查看六个固定区域的原图、标注与预测</summary>"
            + f'<img style="width:100%;height:auto" src="{picture}" alt="六个固定区域的分割对照"></details></section>'
        )
    content = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>矿区干涉影像分割</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101722;color:#e5edf8;font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1540px;margin:auto;padding:32px}h1{font-size:28px;line-height:1.2;margin:12px 0}p{color:#aebdd0}
.bar,.cards,.tools{display:flex;gap:16px;flex-wrap:wrap;align-items:center}.bar{justify-content:space-between}
a{color:#72ddca;text-decoration:none}.tag{border:1px solid #375467;border-radius:24px;padding:4px 12px;color:#96deca}
.card{background:#192331;border:1px solid #29384b;border-radius:12px;padding:14px 20px;flex:1;min-width:150px}.card b{display:block;font-size:23px}
.tools{padding:18px 0}input[type=range]{width:220px;accent-color:#60d7b6}button{background:#26374a;border:1px solid #40536c;color:#e5edf8;border-radius:8px;padding:8px 14px;cursor:pointer}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{border:1px solid #29384b;border-radius:12px;overflow:hidden;background:#080e16}.panel h2{font-size:16px;padding:12px 16px;margin:0;background:#192331}.panel img,.panel canvas{display:block;width:100%;height:auto}
.legend{color:#aebdd0;font-size:13px;margin:12px 0}.swatch{display:inline-block;width:12px;height:12px;border-radius:3px;margin:0 5px 0 16px}.note{background:#192331;padding:16px 20px;border-radius:12px;margin-top:22px;color:#bac8db}
details{margin-top:16px;color:#bac8db}summary{cursor:pointer}code{overflow-wrap:anywhere}.downloads{display:flex;gap:20px;flex-wrap:wrap;margin-top:16px}
table{width:100%;border-collapse:collapse;text-align:left}th,td{padding:10px 8px;border-bottom:1px solid #34445b}
@media(max-width:850px){main{padding:18px}.grid{grid-template-columns:1fr}h1{font-size:24px}}
</style><main>
<div class="bar"><a href="/">← 采掘干涉实验室</a>__LEARNING__<span class="tag">仅用模拟样本训练</span></div>
<h1>矿区干涉影像分割</h1><p>__NAME__ · __KIND__。查看模型圈出的沉降候选区域与边界。</p>
<div class="cards"><div class="card">原始尺寸<b>__SIZE__</b></div><div class="card">保存阈值<b>__THRESHOLD__</b></div>
<div class="card">预测区域占整幅图<b>__AREA__%</b></div><div class="card">融合观察尺度<b>__SCALES__</b></div></div>
<div class="tools"><label>预览阈值 <input id="threshold" type="range" min="0.05" max="0.95" step="0.01"><b id="value"></b></label>
<button id="mode">仅看轮廓</button><button id="reset">恢复保存阈值</button></div>
<div class="grid"><section class="panel"><h2>原始干涉影像</h2><img id="original" alt="原始干涉影像"></section>
<section class="panel"><h2 id="result-title">分割区域与轮廓</h2><canvas id="result" aria-label="分割预览"></canvas></section></div>
<div class="legend"><span class="swatch" style="background:#2cd2a3"></span>预测区域 <span class="swatch" style="background:#ffd65c"></span>区域边界</div>
__SYNTHETIC__
__COMPARISON__
<div class="note">这里显示的是模型的沉降候选区，尚不能作为可靠矿区监测结果。模拟监督标签仍是垂直沉降增量达到设定阈值的区域；真实人工圈选的边界含义尚未统一。滑块只改变缩放预览，不修改全分辨率保存结果或评价指标。</div>
<div class="downloads"><a data-artifact="mask.png" href="mask.png" download>下载二值分割图</a><a data-artifact="overlay.png" href="overlay.png" download>下载叠加图</a>
<a data-artifact="boundaries.png" href="boundaries.png" download>下载轮廓图</a><a data-artifact="probability.npy" href="probability.npy" download>下载原始分数数组</a></div>
<details><summary>结果说明</summary><p>多尺度窗口预测经重叠加权后，在原图坐标内等权融合。观察尺度只表示显示像素范围，没有确定米制像元大小。模型分数未经概率校准。显示码代理没有恢复标定物理相位，也不能由黑色区域判定水体或真实相干性。</p>
<p>训练、验证、测试先按完整合成场景划分，小块继承原场景划分。此页不显示未经核实的真实准确率。详细设置保存在 <code>prediction.json</code>。</p></details></main>
<script>
const data=__DATA__;const image=new Image(),scores=new Image();const slider=document.getElementById('threshold');
if(location.protocol==='http:'||location.protocol==='https:')document.querySelectorAll('[data-artifact]').forEach(a=>{a.href='/segmentation/artifact/'+a.dataset.artifact;});
const canvas=document.getElementById('result'),ctx=canvas.getContext('2d');let pixels,values,contours=false;
slider.value=data.threshold;document.getElementById('original').src=data.image;
function render(){if(!pixels||!values)return;const threshold=Number(slider.value);document.getElementById('value').textContent=threshold.toFixed(2);
const out=new ImageData(new Uint8ClampedArray(pixels.data),pixels.width,pixels.height),w=pixels.width,h=pixels.height;
const inside=(x,y)=>x>=0&&x<w&&y>=0&&y<h&&values[(y*w+x)*4]/255>=threshold;
for(let y=0;y<h;y++)for(let x=0;x<w;x++){if(!inside(x,y))continue;const i=(y*w+x)*4;
if(!contours){out.data[i]=out.data[i]*.55+44*.45;out.data[i+1]=out.data[i+1]*.55+210*.45;out.data[i+2]=out.data[i+2]*.55+163*.45;}
if(!inside(x-1,y)||!inside(x+1,y)||!inside(x,y-1)||!inside(x,y+1)){out.data[i]=255;out.data[i+1]=214;out.data[i+2]=92;}}
ctx.putImageData(out,0,0);}
Promise.all([new Promise(r=>{image.onload=r;image.src=data.image;}),new Promise(r=>{scores.onload=r;scores.src=data.scores;})]).then(()=>{
canvas.width=image.width;canvas.height=image.height;ctx.drawImage(image,0,0);pixels=ctx.getImageData(0,0,image.width,image.height);
ctx.drawImage(scores,0,0);values=ctx.getImageData(0,0,image.width,image.height).data;render();});
slider.addEventListener('input',render);document.getElementById('reset').onclick=()=>{slider.value=data.threshold;render();};
document.getElementById('mode').onclick=()=>{contours=!contours;document.getElementById('mode').textContent=contours?'显示区域填充':'仅看轮廓';
document.getElementById('result-title').textContent=contours?'预测区域轮廓':'分割区域与轮廓';render();};
</script></html>"""
    for key, value in dict(
        NAME=name,
        KIND=source_kind,
        SIZE=f"{width} × {height}",
        THRESHOLD=f"{info['threshold']:.2f}",
        AREA=f"{percent:.2f}",
        SCALES=scales,
        DATA=json.dumps(data),
        COMPARISON=comparison_html,
        SYNTHETIC=synthetic_html,
        LEARNING='<a data-artifact="learning.html" href="learning.html">查看它怎么学习 →</a>'
        if (output / "learning.html").is_file()
        else "",
    ).items():
        content = content.replace(f"__{key}__", value)
    (output / "index.html").write_text(content, encoding="utf-8")
