# Validation record · 2.1.0

Checked locally on 2026-10-07 using Python 3.11.17, NumPy 2.4.6 and Pillow 12.3.0 on macOS ARM64.

- 23 tests passed (`python -m unittest discover -v`). These test mathematical/statistical properties and public-bundle separation, not real-mine prediction.
- Ruff lint and format checks passed; the browser script passed Node syntax checking.
- A newly generated 50-scene batch passed all stored-array, total/component-label, validity/coherence, PNG, version/checksum, exact per-category replay and split checks: train 40, validation 5, test 5.
- Optional PyTorch reading returned `[4,2,256,256]` phase inputs and `[4,1,256,256]` label/validity tensors.
- Fixed-scene diagnostics are saved in [docs/numerical_diagnostics.json](docs/numerical_diagnostics.json). For the default single face at 128px, 1600m and days 40/52, 32 strips differ from a 128-strip numerical reference by relative L2 ≈0.00067545 (0.067545%). This is discretization evidence for that scenario, not a mine accuracy estimate.
- The complex-pair population correlation error at γ=0.7 is ≈0.00028336 under the fixed diagnostic. At γ=0 the mean sample magnitude with 32 ensembles is ≈0.15685, demonstrating finite-sample positive bias. With one ensemble and no spatial average it is 1, confirming the degeneracy stated in MODEL.md.

No field calibration, measured-water classification, real DEM registration, site-held-out network evaluation or external scientific peer review has been completed. Other environments may require justified numerical tolerances. GitHub Actions is configured for Python 3.11–3.13; its remote runs will only be available after repository upload.
