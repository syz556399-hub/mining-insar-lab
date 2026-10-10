"""Explain a completed training run using its own logs and measured first update."""

import base64
import html
import json


def write_learning_report(output, config, history, summary, first_step):
    """Write a standalone lesson; never derive performance from illustrative images."""
    counts = config["split_scene_counts"]
    patches = config.get("patch_training")
    sampling = config.get("patch_sampling", {"mode": "all"})
    sampling_name = {
        "all": "遍历全部图块",
        "uniform": "每景等额随机取样",
        "balanced": "每景等额、目标与背景均衡取样",
    }[sampling["mode"]]
    input_description = (
        "含噪观测相位的 sin/cos 两通道"
        if config["input_mode"] == "phase"
        else "含噪观测图的 RGB 三通道"
    )
    if config["input_mode"] == "phase" and config.get("phase_bins"):
        input_description += f"（先均匀量化为 {config['phase_bins']} 档）"
    if patches:
        patch_description = f"每个输入为 {patches['size']}×{patches['size']}；源场景观察范围倍数为 {patches['context_scales']}。"
    else:
        patch_description = "使用完整场景输入。"
    draws = sampling.get("draws_per_epoch", config["split_counts"]["train"])
    repeat = "，允许重复" if sampling["mode"] != "all" else ""
    if config.get("scene_sampling"):
        sampling_name = f"按目标存在性抽取完整训练场景，含目标概率 {config['scene_sampling']['positive_draw_probability']:.0%}"
        repeat = "，允许重复；验证和测试不重采样"
    images = []
    for filename, caption in (
        ("patch_examples.png", "固定源场景切块：左为观测输入，中为标签，右为有效区。"),
        (
            "test_predictions.png",
            "固定类别的完整合成测试场景：含噪输入、自动标签、预测叠加与模型分数；标签灰色区忽略。",
        ),
    ):
        path = output / filename
        if path.is_file():
            encoded = base64.b64encode(path.read_bytes()).decode()
            images.append(
                f'<details><summary>{caption}</summary><img class="example" src="data:image/png;base64,{encoded}" alt="{caption}"></details>'
            )
    dataset = config["dataset"]
    threshold_mm = dataset.get("target_threshold_mm")
    label = (
        f"两期垂直沉降增量 ≥ {threshold_mm:g} mm"
        if threshold_mm is not None
        else "两期垂直沉降增量达到模拟器设定阈值"
    )
    if config.get("target_mode") == "fringe":
        label = "干净形变相位绝对值达到支持区阈值（实验代理，不是已校准可见边界）；全像素监督"
    test = summary["synthetic_test"]

    def percent(value):
        return "—" if value is None else f"{100 * value:.2f}%"

    data = json.dumps(
        dict(history=history, best_epoch=summary["best_epoch"]), allow_nan=False
    ).replace("<", "\\u003c")
    trace = html.escape(json.dumps(first_step, ensure_ascii=False, indent=2))
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>这次模型怎样学习</title><style>
*{box-sizing:border-box}body{margin:0;background:#101722;color:#e5edf8;font:16px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1120px;margin:auto;padding:28px}h1{font-size:30px}h2{font-size:21px;margin-top:30px}p{color:#bccbdd}a{color:#72ddca}.cards,.flow{display:flex;gap:12px;flex-wrap:wrap}.card,.step,.box{background:#192331;border:1px solid #2c3d51;border-radius:12px;padding:16px}.card{flex:1;min-width:145px}.card b{display:block;font-size:26px}.step{flex:1;min-width:145px}.step b{display:block;color:#72ddca}.box{margin:18px 0}pre{background:#080e16;padding:18px;overflow:auto;border-radius:10px;font-size:14px;line-height:1.7}code{font-family:ui-monospace,monospace}.charts{display:grid;grid-template-columns:1fr 1fr;gap:16px}svg{width:100%;height:auto;background:#192331;border-radius:12px}.axis{fill:#adbed2;font-size:11px}.example{max-width:100%;height:auto;display:block;margin-top:15px}summary{cursor:pointer;color:#72ddca}details{margin:18px 0}input{width:230px;accent-color:#72ddca}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #34465c}@media(max-width:750px){main{padding:18px}.charts{grid-template-columns:1fr}}
</style><main><a id="back" href="index.html">← 返回分割结果</a><h1>这次模型怎样学习</h1>
<p>使用你自己的模拟器重新生成样本，从随机初始化开始训练。真实影像与 LabelMe 标注没有参与参数更新。</p>
<div class="cards"><div class="card">独立训练场景<b>__TRAIN__</b></div><div class="card">独立验证场景<b>__VAL__</b></div><div class="card">独立测试场景<b>__TEST__</b></div><div class="card">完成训练轮数<b>__EPOCHS__</b></div></div>
<h2>1 · 准备问题和答案</h2><p>模型的输入是__INPUT__。答案是模拟器计算的「__LABEL__」二值区域。含噪观测里包括形变、地形残差、大气、轨道背景与统计噪声；网络需要在这些干扰中预测目标区域。输入不含干净参考、沉降真值、标签或有效区。</p>
<p>__PATCH__先按完整源场景分训练、验证、测试，再切块；同一场景的图块不能跨集合。当前取样方式：__SAMPLING__，每轮使用 __DRAWS__ 个训练输入__REPEAT__。图块数不等于独立场景数。</p>
<div class="box">均衡模式仅在一个训练场景同时有含目标图块和纯背景图块时，以各 50% 的机会选择一类，然后在类内随机取块。全背景场景仍训练背景。它不是把目标像元比例强制变为 50%，也不使用验证、测试或真实标注取样。</div>
<h2>2 · 每一批都重复这五步</h2><div class="flow"><div class="step"><b>① 取一批样本</b>观测输入、目标标签、有效区</div><div class="step"><b>② 做预测</b>U-Net 为每个像元输出分数</div><div class="step"><b>③ 算误差</b>BCE + soft Dice，忽略无效像元</div><div class="step"><b>④ 反向传播</b>求误差对网络权重的梯度</div><div class="step"><b>⑤ 更新权重</b>Adam 按梯度调整卷积等参数</div></div>
<pre><code>optimizer.zero_grad(set_to_none=True)  # 清掉上一批的梯度
logits = model(inputs)                # 前向：预测每个像元
loss = masked_loss(logits, target, valid)
loss.backward()                      # 反向：误差怎样随权重变化
optimizer.step()                     # 更新：让参数沿优化方向改变</code></pre>
<p>这与你之前练习的 ŷ = w × x、loss.backward()、更新 w 是同一套过程。这里训练的是 __PARAMETERS__ 个网络参数，输出是一张分割图；采用 Adam，而不是手写单个 w 的梯度下降。网络不会直接学习概率积分法公式，也没有把毫米沉降量作为回归输出。</p>
<details><summary>查看真实记录的第一批学习过程</summary><p>这是实际第一批的张量尺寸、误差、梯度大小和权重变化量，没有额外做训练更新。变化量大于零可确认该批参数发生更新；它不证明真实泛化。</p><pre>__TRACE__</pre></details>
<h2>3 · 看它是否越学越好</h2><p>训练损失通常越低越好；验证 IoU 衡量未参与更新的完整合成场景上预测与标签的重合程度。两条曲线的含义不同，训练损失下降不保证验证表现持续上升。</p>
<div class="charts"><svg id="loss" viewBox="0 0 500 260" aria-label="训练损失曲线"></svg><svg id="iou" viewBox="0 0 500 260" aria-label="整景验证 IoU 曲线"></svg></div>
<div class="box"><label>查看第 <b id="epoch"></b> 轮 <input id="slider" type="range" min="1" max="__EPOCHS__" step="1"></label><p id="record"></p><p>实际保存第 __BEST__ 轮：整景合成验证 IoU 最高。阈值固定 __THRESHOLD__；预测窗口先拼回完整场景，每个有效源像元只计一次。没有用真实图的得分选择训练轮次。</p></div>
<h2>4 · 最后才做合成测试</h2><table><tr><th>目标区域 IoU</th><th>精确率</th><th>召回率</th></tr><tr><td>__IOU__</td><td>__PRECISION__</td><td>__RECALL__</td></tr></table>
<p>IoU = 正确目标像元 /（正确目标 + 误检 + 漏检）。精确率看圈出的区域有多少正确，召回率看真实目标找到了多少。以上是按有效像元累计的合成结果，不能当作真实矿区准确率。</p>
__IMAGES__<h2>5 · 你接着怎样读代码</h2><p>按顺序读 <code>data_io.py → patch_dataset.py → segmentation.py → train.py → predict.py</code>：先追踪一张图的输入、标签与有效区，再看网络前向、误差计算、反向传播及保存权重。单景预测使用学到的权重，不再调用 backward 或更新参数。</p>
<p>教学基础参考《动手学深度学习》的 <a href="https://zh.d2l.ai/chapter_linear-networks/linear-regression.html">训练与优化</a>、<a href="https://zh.d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html">语义分割与配对裁剪</a>。项目网络和模拟核心是本项目的独立实现。</p>
<p>目前监督目标仍是物理阈值区域。真实人工圈选边界含义未统一，显示 BMP 也未恢复标定相位；水体、DEM 与统计观测尚未由真实矿区校准。这些差异决定了合成成绩不能替代真实独立检验。</p></main><script>
const data=__DATA__,hist=data.history,slider=document.getElementById('slider');slider.value=data.best_epoch;
if(location.protocol==='http:'||location.protocol==='https:')document.getElementById('back').href='/segmentation';
function chart(id,key,title,color,percent=false){const v=hist.map(r=>r[key]),n=v.length,L=58,R=480,T=38,B=220,lo=0,hi=percent?1:Math.max(...v)*1.12;
const x=i=>L+(R-L)*i/Math.max(1,n-1),y=a=>B-(B-T)*(a-lo)/Math.max(1e-9,hi-lo);let out='<text x="58" y="23" fill="#e5edf8" font-size="15">'+title+'</text>';
for(let i=0;i<=4;i++){let a=hi*i/4;out+='<path d="M'+L+' '+y(a)+'H'+R+'" stroke="#34465c"/><text class="axis" x="8" y="'+(y(a)+4)+'">'+(percent?(a*100).toFixed(0)+'%':a.toFixed(2))+'</text>';}
const labels=[...new Set([0,Math.floor((n-1)/2),n-1])];labels.forEach(i=>out+='<text class="axis" x="'+x(i)+'" y="240">'+(i+1)+'</text>');
out+='<text class="axis" x="450" y="255">轮数</text><polyline points="'+v.map((a,i)=>x(i)+','+y(a)).join(' ')+'" fill="none" stroke="'+color+'" stroke-width="2.5"/>';
let i=Number(slider.value)-1;out+='<circle cx="'+x(i)+'" cy="'+y(v[i])+'" r="5" fill="#ffd65c"/>';document.getElementById(id).innerHTML=out;}
function draw(){const r=hist[Number(slider.value)-1];document.getElementById('epoch').textContent=r.epoch;document.getElementById('record').textContent='训练损失 '+r.train_loss.toFixed(4)+'；整景验证 IoU '+(r.val_foreground_iou*100).toFixed(2)+'%。';chart('loss','train_loss','训练损失','#72ddca');chart('iou','val_foreground_iou','整景合成验证 IoU','#79adff',true);}
slider.addEventListener('input',draw);draw();</script></html>"""
    values = dict(
        TRAIN=counts["train"],
        VAL=counts["val"],
        TEST=counts["test"],
        EPOCHS=len(history),
        INPUT=input_description,
        LABEL=label,
        PATCH=patch_description,
        SAMPLING=sampling_name,
        DRAWS=draws,
        REPEAT=repeat,
        PARAMETERS=config["parameters"],
        TRACE=trace,
        BEST=summary["best_epoch"],
        THRESHOLD=config["threshold"],
        IOU=percent(test["foreground_iou"]),
        PRECISION=percent(test["precision"]),
        RECALL=percent(test["recall"]),
        IMAGES="".join(images),
        DATA=data,
    )
    for key, value in values.items():
        page = page.replace(f"__{key}__", str(value))
    (output / "learning.html").write_text(page, encoding="utf-8")
