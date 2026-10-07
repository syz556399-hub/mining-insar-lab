# Mining InSAR Lab · 矿区 InSAR 模拟器

A reproducible research prototype for **working-face subsidence + synthetic terrain/water + statistical complex interferograms**. Python 3.11+, NumPy and Pillow; PyTorch is optional. MIT licensed.

本项目用于矿区沉降合成数据、算法开发和学习。代码可运行、公式与单位可检查；当前地形与水体均为合成场景，尚无真实矿区校准或外部验证，不宣称完整 SAR 成像或真实数据泛化能力。

## Quick start / 安装与运行

在工程目录执行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python server.py --open
```

浏览器访问 `http://127.0.0.1:8778/`。macOS 创建环境后可双击 `启动实验室.command`。Windows 激活命令为 `.venv\Scripts\activate`。端口被占用时使用 `python server.py --port 8779 --open`。服务只监听本机。

## Interactive controls / 使用

左上方可选择「简洁 / 高级」。简洁显示常用工作面、日期、地形和相干性参数；高级显示全部参数。模式切换保留当前参数，隐藏参数仍参与计算。

四图同时显示含噪条纹、干净参考、垂直位移与分割标签。环境区显示 DEM、水体、估计相干性与推进范围。点击俯视图移动当前工作面。默认是去地形的差分模式；高级设置可保留完整地形相位。

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

五类场景按场景分组划分，50 张为 40/5/5；其他数量按类别舍入。类别不保证真实标签重叠。组合不同批次前运行 `python validate_dataset.py DIR_A DIR_B` 审核跨集合重复。详见 [采样策略](docs/SAMPLING.md)。

## Model / 科学定义

- 矩形工作面高斯影响积分，沿推进方向分段，一阶时间响应；分别计算两期并求差。
- 东 E、北 N、向下 D 为正；LOS 距离增加为正，形变相位 `−4πΔLOS/λ`。
- 合成 DEM 与独立误差，局部小基线地形相位近似；分层大气与平面波背景。
- 两期相关复高斯观测、独立重复观测平均、地图像素滑动窗口平均、样本相干性估计。
- 水体使用低相干统计模型；标签来自无噪声垂直位移阈值，有效区排除水体和低模型相干像元。

完整公式、经验部分和适用边界见 [MODEL.md](MODEL.md)；引用基础与边界见 [REFERENCES.md](docs/REFERENCES.md)。不把理论背景文献的验证结果当成本项目结果。

## Optional PyTorch input / 训练读取

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

## Checks and publication / 检查与发布准备

```sh
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
python -m unittest discover -v
python prepare_release.py
```

[复现说明](docs/REPRODUCIBILITY.md) · [贡献说明](CONTRIBUTING.md) · [版本记录](CHANGELOG.md) · [MIT 许可](LICENSE)

`prepare_release.py` 生成 `dist/mining-insar-lab-2.1.0/` 和 ZIP，仅包含当前公开源码、合成配置、说明与测试，不包含个人 BMP、本机路径记录、大数据集、旧试验或缓存。GitHub Actions 配置已提供，远端运行结果需上传后才能确认。

CITATION.cff 使用贡献者集体署名；正式发布前可换为核实后的作者名，并补入实际仓库 URL。暂无软件 DOI。使用本软件请记录版本、提交与参数，同时引用研究实际采用的理论。
