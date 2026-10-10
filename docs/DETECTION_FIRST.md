# 先检测、后分析：独立实现的起步实验

本实验参考 Wu、Wang 等（2022）将形变检测与解缠分开的任务思想，
不复现 DDNet/PUNet，不导入作者源代码或预训练权重，也不声称达到论文精度。
原有物理模拟器和分割训练接口保持可用。

## 当前完成

- 用本项目的工作面概率积分实现，生成 20 张清晰沉降正样本和 8 张背景负样本。
- 先叠加两工作面位移，投影为 LOS 相位，再加背景和观测扰动、统一缠绕。
- 净形变目标为 3–6 圈，生成后检查 2–8 圈及 1.5%–40% 相位支持面积；
  这些是清晰样本设计条件，不是矿区总体统计或物理校准结论。
- 正样本有单工作面和双工作面，相邻贡献不保证每张都出现两个可辨识峰。
- 保留合成地形；本清晰组不含水体，后续应以独立难度组加入失相干与水体。
- 独立编写全分辨率残差空洞卷积网络；输入数值缠绕相位的 sin/cos 两通道。

## 监督定义

检测热图是 abs(净形变相位) 除以本幅最大值，负样本全零。
它是相对强度回归目标，既不是校准概率，也不是 LabelMe 完整边界。
另行展示的 fringe_mask 仍为 abs(净形变相位) ≥ π/2 的实验范围标签。
没有自动修改真实人工标注，没有用相位阈值替代真实评估真值。

训练损失为 mean((预测−目标)² × (1+4×目标))。该权重是本项目设计，
不能描述为作者原始损失。模型选择仅依据合成验证集损失，
完成选择后评估合成测试集，并同时报告全零预测的误差，检查背景塌缩。
28 张图划分为 train/val/test = 16/6/6；各组物理场景标识不重复。
这只是流程试验，样本数、随机种子和场景复杂度不足以证明泛化。

## 运行

```sh
python detection_first.py generate --out runs/detection_first_demo
python detection_first.py train --out runs/detection_first_demo --epochs 6
```

结果目录包含数值数组、生成参数、四联图、清单、热图模型和独立 HTML 页面。
专用 detector.pt 与分割 best.pt 的输出含义不同，不能交给原分割预测器使用。

## 后续验收顺序

1. 检查清晰正样本的局部密集条纹、背景和目标范围。
2. 分难度增加噪声、背景方向、幅度显示差异、水体和多尺度目标。
3. 扩大按物理场景隔离的数据集；增加随机种子，保留相同数据的 U-Net 对照。
4. 核对 GAMMA BMP 显示编码与输入分布，在预留真实日期上评估位置召回和误报。
5. 完整范围分割须单独验收；检测可靠后再开发解缠，不能把颜色换算当作解缠。

## 第二轮：难度分层与旧模型对照

新增 detection_study.py，生成 144 张独立场景（96/24/24）。各划分均含清晰、
中等、强干扰三组，正负样本比例为 3:1。地形、水体、噪声均为合成；正负
样本使用相同干扰抽样规则。全场景旋转同时作用于输入和标签，并记录于参数文件。
本轮不把低相干区域从热图真值中自动抹除，强干扰组可包含不可见的目标。

网络保持全分辨率，扩大空洞卷积感受野；训练增广包括空间变换、相位符号和
起点变化、可选16档量化。测试固定16档，仅模拟显示码离散，未恢复GAMMA物理相位。
梯度匹配损失是本项目独立设计，不是作者论文配方。选择仅用合成验证集，
训练完成后在同一24张留出图上比较上一版模型与本版。共同改变结构、训练与数据，
不是能单独归因的消融实验。整个试验仍只有一个初始化种子。

定位命中使用整幅最高分位置和相对强度目标；不能当成完整边界IoU或多盆地实例召回。
负样本误报用固定0.25阈值、64像素最小区域统计。自动导出加24像素边缘的候选
裁块供后续分析，不自动当作最终标注。阈值、损失、训练轮数写入训练前的protocol.json。

```sh
python detection_study.py generate --out runs/detection_study_v2
python detection_study.py train --out runs/detection_study_v2 --baseline runs/detection_first_demo/detector.pt --epochs 16
```

### 检测应用入口

检测应用入口支持合成数值数组，或通过调色板校验的原始索引BMP；保留原图像素坐标，
以256像素滑窗和重叠加权生成热图并导出候选裁块。不接受把任意RGB截图反算物理相位。

```sh
python detect_scene.py --checkpoint runs/detection_study_v2/detector_v2.pt --arrays runs/detection_study_v2/test_000.npz --out runs/detection_windows
```

使用原始BMP时把 --arrays 换成 --image；显示码仍只是数值代理，输出没有真实精度保证。

### 作者资料核对与方法来源边界（2026-10-09）

- 作者公开模型 PUNet.py 展示了全分辨率、多膨胀率卷积和残差连接。
- 作者公开 train.py 中配置的是 phaseUnwrapping 数据任务及 MSE，提供 Adam、
  随机镜像和学习率调度。不能据此声称掌握其全部DDNet检测训练配方。
- 本项目采用自己的PIM、sin/cos输入、轻量网络与损失，不复制代码或权重，
  不把检测热图描述为相位解缠。
- https://github.com/Wu-Patrick/Deformation-Monitoring-Dev/blob/main/model/PUNet.py
- https://github.com/Wu-Patrick/Deformation-Monitoring-Dev/blob/main/train.py

## 冻结模型的真实影像诊断

`real_detection_eval.py` 接受原始索引 BMP、最终 LabelMe 切片目录及两版冻结权重。
先校验每张标注切片与原图像素完全一致，再在原始分辨率上按 256 像素窗口、
192 像素步长推理。固定高分阈值 0.25，候选区域最小 64 像素，框外扩 24 像素。
不根据本次结果修改阈值、模型或标注。

评价仅限保存了最终标注的切片；每份文件分别计数，重叠切片可重复覆盖同一地物。
原始 shape 记录全部保留，不能把记录数当作独立矿区数。报告包括高分像素精确率、
标注像素覆盖率、每条标注的覆盖比例及未接触标注的候选框数量。
归一化形变热图没有接受完整人工范围监督，因此这些是任务差距诊断，
不等同于完成了分割评价，也不能与合成数据的最高分位置命中率直接比较。
使用过的真实开发日期不能重新声称为独立测试集。显示码转换仍非物理相位恢复。
输出保留输入/权重摘要、全部切片对照、原尺寸分数数组和逐条统计。

## 参考文献

- Wu et al. (2022), Deep Learning for the Detection and Phase Unwrapping of
  Mining-Induced Deformation in Large-Scale Interferograms.
  https://doi.org/10.1109/TGRS.2021.3121907
- Wang, Zhang & Wu (2023), A Deep-Learning-Facilitated, Detection-First Strategy
  for Operationally Monitoring Localized Deformation with Large-Scale InSAR.
  https://doi.org/10.3390/rs15092310
