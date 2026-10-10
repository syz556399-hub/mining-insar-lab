# 用自己的模拟器训练矿区沉降区域分割

模拟器批量产生观测图像和自动标签，深度学习学习从含噪图像预测目标区域。当前是二值语义分割：相接的盆地不自动分配独立实例 ID，也不直接回归毫米沉降量。

## 可对照的重训与学习说明（2.6）

先导出一次数据，再在完全相同的数据、网络、训练轮数和每轮输入数量下对照取样方式：

```sh
python cli.py dataset --config examples/training_patches.json --profile sparse_mine --count 400 --out exports/patch_scenes
python validate_dataset.py exports/patch_scenes
python train.py --data exports/patch_scenes --out runs/uniform --input phase --phase-bins 16 --phase-augment --patch-size 128 --patch-stride 96 --context-scales 1 2 --patch-sampling uniform --patch-draws 3 --base 8 --epochs 24 --batch-size 8 --seed 20261012 --device cpu
python train.py --data exports/patch_scenes --out runs/balanced --input phase --phase-bins 16 --phase-augment --patch-size 128 --patch-stride 96 --context-scales 1 2 --patch-sampling balanced --patch-draws 3 --base 8 --epochs 24 --batch-size 8 --seed 20261012 --device cpu
```

400 个独立源场景先划分为 320/40/40，分别有 3200/400/400 个候选图块。`uniform` 与 `balanced` 都每轮对每个有效训练场景抽取 3 次，允许重复，因此每轮 960 个输入，不遍历全部 3200 个候选块。每轮随机取样由训练种子与轮次确定。

`uniform` 在一个源场景的可用图块中均匀取样。`balanced` 用**训练标签与有效区**建立含目标和纯背景两类图块；两类都存在时，以各 50% 的概率选类，再在类内均匀选块。单一类别场景仍使用其可用图块，全背景场景不会被丢弃。全无效图块不进入抽样，忽略像元中的目标不算有效目标。这个比例是图块类别的抽样概率，不是目标像元比例；也不保证所有数据集上都增加目标比例或提高性能。

验证与测试仍使用完整源场景及全部有效像元，不做标签引导的选块。同一导出上选取样方案，应根据合成验证指标，并记录试验次数；最终测试的比较结果是描述性对照，不能反过来继续调参数后称为独立测试。合成训练成功不保证真实图的迁移成功。

每次训练完成后增加：

- `first_step.json`：实际第一批的输入/标签/输出形状、误差、梯度 L2 和权重变化 L2；只是记录，不新增训练更新。
- `learning.html`：可独立打开的学习说明、实际设置、逐轮损失/验证 IoU 曲线、固定合成样本与合成测试结果。可拖动轮次滑块。

用该目录的 `best.pt` 预测时，学习页会复制到结果目录，并由分割查看页的「查看它怎么学习」打开。数值合成样本页另外显示自动标签和有效区；这些答案仅用于查看，不进入模型预测。

一批训练的代码流程是 `model(inputs) → masked_loss → loss.backward() → optimizer.step()`，相当于此前单参数梯度练习的扩展。Adam 更新网络参数；无效像元不参与误差。预测阶段只读取权重做前向，不再更新。教学概念参考《动手学深度学习》的 [训练与优化](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html) 和 [语义分割](https://zh.d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html)。

完整 runner 的 `--scales` 未指定时，使用 `--context-scales` 的值。四个尺度等权平均需要显式选择，不能预设其一定优于单尺度。单独运行 `predict.py` 保留旧版默认单尺度行为；所有实际尺度会记录到 `prediction.json`。

## 整景切块分割实验（2.5）

```sh
python segment.py --out runs/my_segmentation
```

该命令完成生成、训练、合成整景评价和固定合成测试样本预测。默认生成 100 个 256×256 的独立场景，使用 `sparse_mine` 抽样和 16 档相位量化。80/10/10 场景分别得到 800/100/100 个图块：每个场景有 9 个重叠的 128×128 局部图块、1 个完整 256×256 上下文图块；后者缩小为 128×128 网络输入。图块数量不等于独立场景数量。

输入、标签、有效区使用同一个裁剪坐标。较大观察范围的 sin/cos 输入采用圆周平均，二值标签与有效区采用最近邻采样；这不是新的 SAR 多视物理模型。训练时同步翻转输入、标签与有效区，相位原点和符号增强只改变输入。

训练验证和最终合成测试均先将重叠窗口加权拼回原场景，再等权融合各观察尺度；每个原场景有效像元只计一次。以整景验证目标类 IoU 选择模型，完成选择后才评价合成测试场景。训练日志中的 `val_loss` 为 null，因为此路径按拼接后的分割指标选择模型，没有将窗口损失当作整景损失。

结果目录：

```text
dataset/                  完整合成场景与原始分组清单
training/patches_train.csv 图块的源场景、分组及裁剪坐标
training/patches_val.csv
training/patches_test.csv
training/best.pt           合成整景验证选择的权重
training/summary.json      合成整景测试指标
training/test_predictions.png 固定类别的完整测试场景预测
prediction/mask.png        原图坐标中的二值分割图
prediction/overlay.png     区域填充与黄色边界
prediction/boundaries.png  只叠加边界
prediction/probability.npy 未阈值化的 float32 模型分数
prediction/index.html      可调预览阈值的查看页
run.json                  完整实验记录
```

更大的场景可以显式指定：

```sh
python segment.py --out runs/large_context --scene-size 512 --patch-size 256 --patch-stride 192 --context-scales 1 2
```

单独给已有合成数据切块训练：

```sh
python train.py --data exports/scenes_256 --out runs/patch_trial --input phase --phase-bins 16 --phase-augment --patch-size 128 --patch-stride 96 --context-scales 1 2 --base 8 --epochs 20
```

测试你的原始索引色 BMP：

```sh
python segment.py --out runs/my_real_trial --image original_indexed.bmp --palette-code --scales 1 2 4 8
```

这里仍然只用模拟场景训练。真实图只进入预测，程序不会读取真实标注用于拟合。1/2/4/8 等权融合是显式的实验设置，不保证比任何单尺度好；实际性能需要固定评价集验证。BMP 必须通过已有调色板核验，仍属于显示码代理。

完成后运行 `server.py`，打开 `/segmentation` 或点击「影像分割」。滑块改变的只是缩放预览，不重写保存掩膜或指标；全分辨率结果以 `prediction.json` 内阈值为准。二值区域为候选区，未经真实独立验证不能作为可靠监测结果。已有实验目录不会覆盖。

## 数据与输入

`cli.py dataset` 导出五类设计场景：无沉降、单工作面、分离工作面、重叠候选、错时推进。类别是场景设计，真实标签是否重叠以元数据为准。生成器允许 50–2000 张；数量不是独立物理机理数量。扩展批次前审计跨集合场景重复。

默认分辨率 256×256，80%/10%/10% 的分配按各类别舍入，先按整场景划分，再训练，不把同一场景的图块随机分到不同集合。

| 输入模式 | 模型看到的数据 | 可用于预测的输入 |
|---|---|---|
| `rgb`（默认） | 已导出的含噪 PNG 转 RGB，除以 255 | BMP/PNG/JPEG 彩图或模拟 NPZ |
| `phase` | 缠绕相位的 sin/cos，两通道 | 数值 NPZ；或显式核验的索引色显示码代理 |

输入不含干净参考、位移真值、分割标签或有效区；标签和有效区只用于监督与评估。

现有真实 BMP 采用的调色板与模拟器循环配色可能不同。RGB 模型可以读入 BMP，但需要匹配配色和像素尺度、处理显示缩放及裁边等差异，不能从未知配色宣称已恢复真实相位。像素大小应依据地理元数据确定，不能以圈选尺寸反推真实米制尺度。

## 自动监督

`mask = delta_down_m >= target_threshold_mm / 1000`，不是按彩色条纹人工圈选。`valid_mask` 为模拟器定义的有效区：排除水体和低模型相干区域。训练需要数组中显式保存有效区。

损失由逐像元 BCE 和批次 soft Dice 相加组成，两项都忽略无效像元。没有有效像元的批次跳过；整个集合无有效像元则报错。Dice 在批次内汇总，不是整景实例损失；日志 loss 按有效像元数加权。

真实 LabelMe 若只圈条纹中心，不能直接等同于上述物理阈值边界。真实评价需要先写清“可见异常区域”还是“沉降阈值区域”，不要把两种指标混用。

## 模型和设备

`SmallUNet` 是自行编写的三次下采样 U-Net 基线，用两层卷积、GroupNorm、ReLU、跳跃连接和双线性上采样，输出单通道 logits。默认通道基数 16，可用 `--base 8` 先做轻量检查。网络结构参考 [Ronneberger、Fischer、Brox（2015）的 U-Net 思想](https://arxiv.org/abs/1505.04597)，不使用原论文源码，也不宣称架构创新。

`--device auto` 依次检查 CUDA、MPS、CPU；`--device cpu` 可用于一致的基础验证。MPS 使用条件参考 [PyTorch 官方文档](https://docs.pytorch.org/docs/2.14/notes/mps.html)。设备支持和速度需要在实际环境中确认。

随机种子默认 42，记录 Python/NumPy/PyTorch 和模拟器版本；不同设备与依赖版本仍可能出现数值差异。基线暂不包含断点续训、多 GPU 或超参数搜索。相位模式支持下述可选原点和符号增强。

## 显示编码差异与稀疏场景

显示 BMP 可以把明暗强度与循环颜色同时编码进调色板。直接把这类 RGB 输入只见过明亮循环配色的模型，会产生输入分布差异。`phase_encoding.py` 提供明确选择的 `--palette-code` 适配：仅接受原始 P 模式、256 色、16 个明暗等级 × 16 个按方向排列的循环颜色，核验其 RGB 缩放关系和颜色循环后，取索引的低四位作为循环显示码。其他调色板或 RGB 拷贝报错。

显示码只表示颜色顺序。它没有恢复经过标定的物理相位，仍不知道物理原点、符号、尺度和可能的显示重采样；不能用它反演毫米位移。零亮度没有可见颜色，适配预测会将其输出置零，但不把黑色判为水体，也不把它当作测得的相干性有效区。公共源码不包含任何个人 BMP 或私有调色板。

为减少对固定配色的依赖，相位训练可以设置 `--phase-bins 16 --phase-augment`。观察相位先均匀量化为 16 档，再在训练集合中随机改变全图相位原点和正负符号；验证和测试不做随机增强。标签和有效区不参与输入，不随相位显示原点改变。这个设置用于显示代理迁移探索，不代替物理标定。

`--profile sparse_mine` 在原来的物理方程之上改变抽样范围：更低的目标密度、更宽的观测扰动、较少平均次数和更多偏离中心的工作面。所有范围写在 [抽样说明](SAMPLING.md)，属于实验设计，未由真实矿区校准。此选项为 CLI 批量导出设置，浏览器保持原来的简洁/高级参数布局。

```sh
python cli.py dataset --config examples/training_transfer.json --profile sparse_mine --count 400 --out exports/transfer_trial
python validate_dataset.py exports/transfer_trial
python train.py --data exports/transfer_trial --out runs/transfer_trial --input phase --phase-bins 16 --phase-augment --base 8 --epochs 20
python predict.py --checkpoint runs/transfer_trial/best.pt --image original_indexed.bmp --palette-code --out runs/transfer_prediction
```

固定预测阈值和真实对照区域，先比较原 RGB 模型与仅改输入编码的结果，再比较重训结果。后者同时改变数据量、场景抽样和输入训练设置，不能据此分别宣称各改动的因果贡献。若真实图已经用于改进设计，应归入开发对照集，另保留未参与调整的日期或矿区做验证。

## 运行

安装依赖后先从 50 张开始检查流程，再增加数据量。所有输出目录必须是新路径，防止覆盖。

```sh
python -m pip install -r requirements-training.txt
python cli.py dataset --config examples/training_quick.json --count 50 --out exports/first_training
python validate_dataset.py exports/first_training
python train.py --data exports/first_training --out runs/first_training --base 8 --epochs 5
```

完成后查看：

- `config.json`：模型、输入模式、训练参数、分组规模、数据清单和配置校验和。
- `history.csv`：每轮训练/验证 loss、验证目标类 IoU 和 F1。
- `best.pt`：验证目标类 IoU 最高的权重；相同分数取较早轮次。
- `summary.json`：最佳验证指标及完成选择后评估的合成测试指标。
- `test_predictions.png`：每个设计类别最多一例固定选取的合成测试预测；不挑选最好看的结果。

IoU/F1 等指标累计有效像元的 TP/FP/FN，属于目标类微平均；某个指标分母为零时输出 null。预测阈值预先固定为 0.5。没有报告由大量背景抬高的像元准确率。这里只评估合成测试集，不能代替真实独立测试。

## 预测

```sh
python predict.py --checkpoint runs/first_training/best.pt --image your_interferogram.bmp --out runs/prediction
python predict.py --checkpoint runs/first_training/best.pt --arrays exports/first_training/test/arrays/face_scene_00000.npz --out runs/synthetic_prediction
```

第二条为路径形式示例，具体场景名以导出的 manifest 为准。预测按训练画幅大小滑窗，默认步长为画幅的 75%，重叠窗口用带正值边缘的 Hann 权重平均概率，保持原始像素尺寸。输出概率数组、灰度概率图、二值掩膜和预览。GroupNorm 使用每个窗口内的统计量，窗口上下文会影响结果；大图滑窗预测与整图前向结果不保证相同。

真实 RGB 输入没有真实水体/相干性有效区，因此程序不自动推断“可评估区域”。显示码适配也只排除零亮度显示像元；需要外部有效区和完整、语义一致的标注才能做规范的真实指标。

可选 `--downsample 2`、`4` 或 `8` 让窗口观察原图上更大的范围，最后将概率恢复到原图坐标。相位双通道采用圆周均值，避免把 +π/−π 两侧错误地平均成零相位；RGB 采用显示空间平均。这只是改变显示像素上下文，会损失细目标和高频信息，不建立真实的米制像元间距。默认 1 保持原分辨率推理。若通过真实开发图选择缩小倍率，必须记录选择过程，并在另外保留的图上固定倍率评价。

## 研究评价

第一阶段先证明流程正常，检查标签、困难场景和漏检/误检。第二阶段固定真实评价集，对照纯合成训练与其他基线，报告目标类 IoU、F1、召回和误检，并区分日期外推与跨矿区外推。若用某些真实图调整合成参数或配色，这些图归入开发集，不能同时作为最终独立测试集。

大量合成图可以训练模型；只有真实独立评价才能判断训练是否对实际矿区有帮助。
