# 历史流程快照 · 2.0 发布前的旧路线

本页保留旧开发说明供对照，旧标签、界面和数据格式不代表新版。当前入口与安装方法以根目录 README 为准。文中的历史版本和训练结果不作为本次发布的新结果。

# Mining InSAR Lab · 矿区 InSAR 模拟器

A reproducible research prototype for **working-face subsidence + synthetic terrain/water + statistical complex interferograms**. Python 3.11+, NumPy and Pillow; PyTorch is optional. MIT licensed.

本项目用于矿区沉降合成数据、算法开发和学习。代码可运行、公式与单位可检查；当前地形与水体均为合成场景，尚无真实矿区校准或外部验证，不宣称完整 SAR 成像或真实数据泛化能力。

**目前优先完善模拟器和标签，不继续训练新网络。** 新的 [20 张审核工作台](SIMULATOR_REBOOT.md)
把沉降源、背景、统计观测、显示外观与待确认标签分开；可逐张修边界、记录意见，
确认后导出为混合训练集的合成部分。旧版训练实验保留作历史对照。

历史：独立的“先检测、后分析”起步实验包括清晰正样本、背景负样本与全分辨率检测热图网络。
运行方法、监督定义和限制见 [检测路线说明](DETECTION_FIRST.md)。这不是 DDNet/PUNet 复现，尚未实现相位解缠。

新增 [自己的矿区范围模型](MINE_LEARNING.md)：自己的模拟数据、从零训练的范围分割，
以及有/无干净形变相位辅助任务的固定对照。它直接预测范围，合成标签仍是待校准的相位支持区代理。

## Quick start / 安装与运行

在工程目录执行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python simulator_review.py build --run runs/first_review
python simulator_review.py serve --run runs/first_review --port 8781
```

浏览器访问 `http://127.0.0.1:8781/`。生成命令只运行一次；再次打开已有批次只运行 `serve`。
参考自己的索引 BMP 时，生成命令追加 `--bmp my_reference.bmp`。Windows 激活命令为 `.venv\Scripts\activate`。
服务只监听本机，端口被占用时可以改 `--port`。旧版界面仍可用 `python server.py --open` 打开在端口 8778，
macOS 的 `启动实验室.command` 也仍启动旧版。

20 张影像和边界全部确认后，可[批量扩充同一风格的样本](SIMULATOR_REBOOT.md#确认风格后的批量扩充)：

```sh
python simulator_batch.py build --review-run runs/first_review --out exports/batch_500 --count 500
python simulator_batch.py serve exports/batch_500 --port 8782
```

新样本另存，保留参数、固定场景划分和 LabelMe 标注。程序生成的标签标为待人工抽查，不自动继承模板的确认状态。

## Interactive controls / 使用

左上方可选择「简洁 / 高级」。简洁显示常用工作面、日期、地形和相干性参数；高级显示全部参数。模式切换保留当前参数，隐藏参数仍参与计算。

四图同时显示含噪条纹、干净参考、垂直位移与分割标签。环境区显示 DEM、水体、估计相干性与推进范围。点击俯视图移动当前工作面。默认是去地形的差分模式；高级设置可保留完整地形相位。

水体类型可选随机组合、随机河道、随机湖塘或两者并存。位置、角度、弯曲、宽度、湖塘数量和岸线随种子变化；随机组合允许无水体场景。水体仍为几何近似，未由 DEM 水系推导。

本机可选 BMP 外观对照不进入公共源码包或合成训练数据；没有该文件的安装会自动隐藏对照区。

## Reproducible commands / 无界面复现

```sh
python cli.py config --out my_parameters.json
python cli.py scene --config examples/default.json --out exports/scene.zip
python cli.py dataset --config examples/default.json --count 50 --out exports/demo
python validate_dataset.py exports/demo
python diagnostics.py --out exports/diagnostics.json
```

输出路径已存在时 CLI 不覆盖。单景 ZIP 包含数组、PNG 和参数。批量输出：

```text
train/val/test/
  images/    wrapped-phase visualization PNG
  masks/     vertical-increment binary label PNG
  arrays/    numerical fields + water/coherence/validity NPZ
  metadata/  normalized parameters + version/units/hashes JSON
manifest.csv
 dataset.json
preview.png
```

五类场景按场景分组划分，50 张为 40/5/5；其他数量按类别舍入。类别不保证真实标签重叠。组合不同批次前运行 `python validate_dataset.py DIR_A DIR_B` 审核跨集合重复。详见 [采样策略](SAMPLING.md)。

## Model / 科学定义

- 矩形工作面高斯影响积分，沿推进方向分段，一阶时间响应；分别计算两期并求差。
- 东 E、北 N、向下 D 为正；LOS 距离增加为正，形变相位 `−4πΔLOS/λ`。
- 合成 DEM 与独立误差，局部小基线地形相位近似；分层大气与平面波背景。
- 两期相关复高斯观测、独立重复观测平均、地图像素滑动窗口平均、样本相干性估计。
- 水体使用低相干统计模型；标签来自无噪声垂直位移阈值，有效区排除水体和低模型相干像元。

完整公式、经验部分和适用边界见 [MODEL.md](../MODEL.md)；引用基础与边界见 [REFERENCES.md](REFERENCES.md)。不把理论背景文献的验证结果当成本项目结果。

## Optional PyTorch input / 训练读取

2.6 增加可对照的场景取样、第一批学习记录和「它怎么学习」说明页；保留整景切块训练和分割查看页。安装训练依赖后，可以直接运行完整实验：

```sh
python -m pip install -r requirements-training.txt
python segment.py --out runs/my_segmentation
python server.py --open
```

模拟器生成 100 个 256×256 场景；按完整场景划分后，取 128×128 和 256×256 两种观察范围，统一为 128×128 输入训练。网页点击「影像分割」查看最后完成的实验。完整命令、真实 BMP 输入及评价边界见 [分割训练说明](TRAINING.md)。公共包不含训练权重或个人影像。

需要扩大独立场景数量并控制每轮取样预算时，可用 `python segment.py --out runs/retraining --count 400 --epochs 24 --patch-sampling balanced --patch-draws 3`。`uniform` 与 `balanced` 可在同一导出数据上做对照，只有后者按训练标签平衡同一源场景内的目标图块与背景图块。均衡取样是否改善性能需要验证，不能预先保证。

在单独安装 PyTorch 的环境中：

```python
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits
from data_io import PhaseDataset

loader = DataLoader(PhaseDataset('exports/demo', return_valid=True), batch_size=4)
x, y, valid = next(iter(loader))
# x: [B,2,H,W] = sin/cos(wrapped phase); y and valid: [B,1,H,W]
# logits = model(x)
# pixel_loss = binary_cross_entropy_with_logits(logits, y, reduction='none')
# loss = (pixel_loss * valid).sum() / valid.sum().clamp_min(1)
```

`return_valid=False` 保留双返回值接口。PNG 颜色仅是显示，不是物理量；真实 BMP 的颜色与相位映射未知时不能反演位移或相干性。

## Train and predict / 用模拟器训练深度学习

流程：模拟工作面与观测 → 自动生成图像、标签和有效区 → 训练轻量 U-Net → 留出合成场景检验 → 在真实干涉图上独立验证。

```sh
python -m pip install -r requirements-training.txt
python cli.py dataset --count 1000 --out exports/train_1000
python validate_dataset.py exports/train_1000
python train.py --data exports/train_1000 --out runs/rgb_baseline --input rgb --epochs 30
python predict.py --checkpoint runs/rgb_baseline/best.pt --image your_interferogram.bmp --out runs/real_prediction
```

第一版默认 RGB 输入，读取含噪图像；标签来自无噪声垂直沉降增量阈值。水体及低模型相干像元在损失和指标中忽略。按验证集目标类 IoU 保存最佳权重，训练结束后只对该权重评估合成测试集。结果包含 `best.pt`、`history.csv`、`summary.json` 和 `test_predictions.png`。

有数值相位时可用 `--input phase` 训练 sin/cos 两通道；对应预测使用 `--arrays path/to/arrays.npz`。2.4 新增 CLI `--profile sparse_mine` 稀疏场景抽样、相位原点/符号增强，以及显式选择的 `--palette-code` 索引色显示码适配。后者严格核验支持的调色板，仅提取颜色顺序代理，不恢复物理相位。真实 BMP 的配色、空间分辨率和标注口径差异仍需处理，读取成功不代表真实矿区有效。完整命令与边界见 [训练说明](TRAINING.md)。

## Checks and publication / 检查与发布准备

```sh
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
python -m unittest discover -v
python prepare_release.py
```

[复现说明](REPRODUCIBILITY.md) · [贡献说明](../CONTRIBUTING.md) · [版本记录](../CHANGELOG.md) · [MIT 许可](../LICENSE)

`prepare_release.py` 生成当前版本的 `dist/mining-insar-lab-2.6.0/` 和 ZIP，仅包含当前公开源码、合成配置、说明与测试，不包含个人 BMP、本机路径记录、大数据集、模型权重、旧试验或缓存。GitHub Actions 配置已提供，远端运行结果需上传后才能确认。

CITATION.cff 使用贡献者集体署名；正式发布前可换为核实后的作者名，并补入实际仓库 URL。暂无软件 DOI。使用本软件请记录版本、提交与参数，同时引用研究实际采用的理论。

## 真实标注与尺度诊断

新增只读 LabelMe 标注核查、重叠区域差异检查和固定模型尺度对照。使用边界及两路训练设计见 [真实迁移约定](REAL_TRANSFER.md)。2.8 新增用户确认最终标注后的真实数据微调对照；保留每个 LabelMe 文件的范围与对象记录。未确认标注仍使用只读核查流程。

## 模拟器任务对齐实验

2.9 保留物理沉降标签，另存实验性干净形变相位支持区标签，界面可切换查看。新增只从开发标注提取像素面积和出现频率、再独立生成合成场景的流程，以及三组纯合成训练对照。见 [实验协议](SIMULATOR_ALIGNMENT.md)。支持区尚未校准为人工可见条纹边界，不宣称生成效果已经改善真实识别。
