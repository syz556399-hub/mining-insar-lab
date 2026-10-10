# Mining InSAR Lab 2.0

This release turns the independently implemented mining InSAR simulator into a review-to-dataset workflow. Start with 20 scenes, inspect and edit complete fringe-range labels, then expand the approved parameter family with fresh synthetic sources.

## Included

- Working-face influence integrals, time response and LOS phase; synthetic terrain, DEM error, atmosphere, water and correlated complex observations.
- Simple/advanced review controls, exact mask editing, old revisions and hash-bound approval.
- Batch generation, independent source-group train/validation/test splits, integrity validation and explicit resume.
- A paginated gallery with input, candidate outline, binary label and filters.
- PNG/NPZ/metadata/LabelMe exports and an optional PyTorch reader that requires explicit opt-in for draft labels.
- MIT license, scientific definitions, reproducibility notes and automated checks.

## Install

Python 3.11+; install requirements.txt. Follow the repository README to build and serve the first review batch. Once all 20 scenes are explicitly approved, use simulator_batch.py build with --count 500 (or another count), validate the output and open its gallery.

## Validation and limits

All 93 unit tests, local Ruff checks and a clean-package 20-scene smoke generation passed. A separate local 500-scene draft export was checked for array/image/label consistency and loading; it is excluded from the source download. No new model was trained for this release.

Labels begin as an uncalibrated clean-phase-support proxy and require review. DEM/water and parts of the observation model are synthetic/empirical. No complete SAR imaging, measured mine calibration, arbitrary real-image segmentation or synthetic-to-real improvement is claimed. See docs/SIMULATOR_REBOOT.md and VALIDATION.md.

## Version naming and contents

The rebuilt product is released as 2.0.0. Numerical engine identifiers retain their original replay meanings. Previous repository commits and the earlier 2.1–2.9 development notes are preserved.

The attached ZIP contains the allowlisted source, configuration, documentation, synthetic UI screenshot and tests, with SOURCE_MANIFEST.json and a separate ZIP SHA-256 file. It excludes private BMP/LabelMe sources, large generated datasets, weights, caches and local histories.

---

中文：2.0 包含新版 20 张审核工作台、独立生成核心、批量生成、分页查看和训练数据导出。先确认影像与完整条纹范围，再扩充样本；新批量标签仍为自动候选，不能当作全部人工确认或真实矿区精度证明。安装和运行步骤见 README。
