# 模拟器任务对齐实验

本实验检验模拟器改动的作用，不使用真实影像微调。物理位移与观测方程保持不变，新增标签字段使引擎标识更新为 `working-face-integral-2.3`。

## 保留真值，分开任务代理

- `mask`：垂直沉降增量达到 `target_threshold_mm`，原定义不变。
- `valid_mask`：原有水体/模型相干性筛选，原定义不变。
- `fringe_mask`：`abs(deformation_phase_rad) >= fringe_threshold_rad`；默认 π/2，为四分之一相位周期的显式设计约定。
- `fringe_valid_mask`：全 1，让包括水体、低模型相干区域在内的像素参与这一任务的监督。

支持区只由干净形变相位计算；不读取大气、地形噪声或随机观测图来定标签。它没有恢复被包裹相位的真实信息，也**不等于已经校准的人工可见条纹边界**。水体上是否仍能支持范围推断需要真实验证，不能因为全像素监督就宣称模型能看穿失相干。

## 像素统计驱动设计

从一个开发日期的最终 LabelMe 文件按完整 256×256 块提取：含目标比例、正块中的目标像素面积分布。原始文件保持不变，轮廓纹理不复制到合成图。未保存标注的切片不使用。重叠视图保留各自标签，因此统计代表所选标注视图，不代表整个矿区的真实发生率。

生成时独立抽取开采/无开采与期望目标面积，在原有物理模拟中最多两次调整场景边长（400–6000 m）以改变像素覆盖；无支持区时允许增加厚度但不超过 8 m。最终尺寸可能未达到目标，逐景记录请求值、实现值和参数，不变形输出图或手绘掩膜补数。这是**像素外观设计**，不构成实际像元间距、埋深或开采参数反演。

## 三组对照

| 组 | 合成图 | 标签及监督 |
|---|---|---|
| A | 原稀疏矿区配置 | 物理阈值及原 valid_mask |
| B | 与 A 完全相同 | 实验支持区及全像素监督 |
| C | 匹配开发样本目标频率、面积 | 与 B 相同 |

B 同时改变标签和有效区，不能把 A/B 差异单独归因于阈值。C 改变频率和面积，不能单独归因于其中之一。背景观测过程本轮没有另行拟合；后续应再分开做消融，而不是同时堆叠改动。

固定网络、随机初始化、相位编码、训练轮数、每批样本数、学习率及真实评价规则。A/B 共享合成图及划分是成对对照设计，不能当作两个独立数据集。每个模型只按各自合成验证集 IoU 选择；真实开发集只评价。跨日期同矿区结果不等于跨矿区泛化，保留测试日期不参与拟合或选模。

```sh
python cli.py dataset --config examples/simulator_alignment.json --profile sparse_mine --count 200 --out exports/alignment_original
python transfer_design.py fit --labels data/development_labels --out runs/pixel_profile.json
python transfer_design.py generate --baseline exports/alignment_original --profile runs/pixel_profile.json --out exports/alignment_matched
# 三组保持以下训练参数相同，只改变数据路径和 --target。
python train.py --data exports/alignment_original --out runs/arm_a --input phase --phase-bins 16 --phase-augment --target physical --epochs 12 --batch-size 8 --base 8 --lr .001 --seed 20261017 --device cpu --threads 2
python train.py --data exports/alignment_original --out runs/arm_b --input phase --phase-bins 16 --phase-augment --target fringe --epochs 12 --batch-size 8 --base 8 --lr .001 --seed 20261017 --device cpu --threads 2
python train.py --data exports/alignment_matched --out runs/arm_c --input phase --phase-bins 16 --phase-augment --target fringe --epochs 12 --batch-size 8 --base 8 --lr .001 --seed 20261017 --device cpu --threads 2
```

真实评价采用每张最终标注切片内部的原始像素块，填充不计分，重叠切片视图分别计入。该指标不得与旧整景去重指标直接比较。数值数据、参考标签统计和权重不进入公共源码包。

## 追加 D：生成频率与训练抽样分离

当 C 在固定预算内的合成验证几乎不检出目标时，追加 D。D 复用 C 全部生成数据与划分，仅训练时按 80% 概率抽取含目标场景、20% 抽取无目标场景；允许重复，每轮输入总数仍为 160。验证/测试不重采样。模型初始化与训练预算保持相同，追加决定依据合成验证退化，发生在真实开发评价之前。

```sh
python train.py --data exports/alignment_matched --out runs/arm_d --input phase --phase-bins 16 --phase-augment --target fringe --scene-positive-fraction .8 --epochs 12 --batch-size 8 --base 8 --lr .001 --seed 20261017 --device cpu --threads 2
```

C/D 的区别是训练抽样，不应归功于模拟器本身。真实使用场景稀疏，并不要求每个训练批次都同样稀疏。抽样信息、独立训练场景数与每轮有放回抽取次数分别记录。D 在相同 200 景数据中重复使用目标，因此不等于增加独立目标多样性。
