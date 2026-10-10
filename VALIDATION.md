# Simulator 2.0 release checks · 2026-10-10

Checked from the public allowlisted source snapshot on macOS ARM64 using Python 3.11.17, NumPy 2.4.6, Pillow 12.3.0 and optional PyTorch 2.14.1.

- All 93 unit tests passed, including independent source/observation properties, exact mask/LabelMe round trips, review authorization and hash checks, fresh grouped batch sampling, safe resume and source-package privacy boundaries.
- Ruff 0.15.0 lint and format checks passed (51 Python files).
- The public quick-start command generated all 20 new scenes with the built-in display profile, including two zero-label background scenes. These smoke-test scenes remain unreviewed; test execution does not authorize human approval.
- The separate local 500-scene batch completed as 400/50/50 synthetic train/validation/test sources. Saved numerical fields, PNG labels and exact LabelMe mask round trips were checked; all 500 optional PyTorch records loaded at native resolution. Three fixed scenes replayed their numerical fields exactly. The batch contains 50 background negatives, 61 synthetic-water cases and 15 phase-sampling warnings, all retained. This dataset and its private reference records are excluded from the source release.
- The 500 new automatic labels remain drafts. These checks establish implementation and export consistency, not approval of every visible boundary or real-mine accuracy. No new model was trained for this release.
- GitHub CI runs separately on the uploaded release commit. Its current result is available in the repository Actions page; local checks do not substitute for that result.

## Archived development checks

# Current development checks · 2026-10-09

- 72 tests passed with the optional PyTorch environment, including signed auxiliary-target alignment,
  preserved negative truth, finite gradients and dataset tampering/cross-split reuse rejection.
- An import-blocked optional-dependency check passed 37 numerical tests and skipped 35 learning tests.
  An earlier empty lint environment lacked NumPy; it was not used as a valid package environment.
- Ruff 0.15.0 lint and formatting checks passed locally.
- The independent mining-range pilot generated 192 scenes in a 128/32/32 split, with 192 unique
  physical group identifiers. All file checksums and declared support labels were checked;
  one saved scene from each split replayed exactly across all seven saved arrays.
- Two matched synthetic-only training arms compare range segmentation with/without an auxiliary
  clean-deformation phase loss. Protocol, selection, complete test predictions and real development
  diagnosis are stored with each local run. See [method and limitations](docs/MINE_LEARNING.md).
- The synthetic range remains a phase-support proxy, not a validated manual fringe boundary.
  Neither synthetic scores nor same-mine development results establish field/generalization accuracy.
  Remote CI for these changes has not run.

# Historical validation record · 2.4.0

Checked locally on 2026-10-08 using Python 3.11.17, NumPy 2.4.6, Pillow 12.3.0 and PyTorch 2.14.1 on macOS ARM64.

- 36 tests passed, including display brightness invariance, unsupported-palette rejection, circular quantization, sampling reproducibility and circular downsampling across the phase wrap boundary.
- Ruff 0.15.0 lint and formatting checks passed.
- A 400-scene `sparse_mine` export at 128×128 passed numerical, label, validity, PNG, provenance, exact per-category replay and split checks: train 320, validation 40, test 40; no cross-split repeated geometry/time groups.
- Actual mean foreground fraction in that export was 1.0517%, versus 6.0205% in the earlier 50-scene standard demonstration. These are observed sample statistics, not field occurrence estimates.
- A two-channel SmallUNet with base width 8 (122,393 parameters), 16-bin phase display quantization and training-only phase origin/sign augmentation trained for 20 epochs on CPU. Epoch 15 was selected by synthetic validation foreground IoU 0.251120.
- The once-evaluated, harder synthetic test set returned foreground IoU 0.312554, F1 0.476253, precision 0.491962 and recall 0.461517 over 644,775 valid pixels. It differs from the old demonstration dataset; the two test scores do not isolate the effect of any model change.
- Private local real-image development comparisons tested palette adaptation, synthetic retraining and display context factors 1/2/4/8 at the fixed 0.5 threshold. A setting chosen on the first date was kept fixed on a second date. Agreement improved but remained low; incomplete-background review, annotation semantics and missing physical phase/pixel-scale calibration prevent a field-accuracy claim. Real annotations were not used to fit weights.
- The display adapter extracts an ordered color-code proxy; zero brightness is excluded from its predictions only. This is not measured coherence/water validity or phase recovery.
- No independent mine/site-held-out evaluation, field calibration or external scientific review has been completed. Remote CI has not run for this local version.

## Historical checks · 2.3.0

Checked locally on 2026-10-08 using Python 3.11.17, NumPy 2.4.6, Pillow 12.3.0 and PyTorch 2.14.1 on macOS ARM64.

- 31 tests passed, including masked-loss/metric invariance, zero ignored-pixel gradients, observed-input separation, checkpoint round-trip and tiled edge coverage.
- Ruff 0.15.0 lint and formatting checks passed.
- A new 50-scene 128×128 export passed numerical, label, validity, PNG, provenance, per-category exact replay and split checks: train 40, validation 5, test 5.
- An RGB SmallUNet with base width 8 (122,465 parameters) trained for 5 epochs on CPU; selected epoch 5 by validation foreground IoU (0.813680).
- The once-evaluated synthetic test split returned foreground IoU 0.848364, F1 0.917962, precision 0.938833 and recall 0.897998 over 79,459 valid pixels. These are five small synthetic scenes, not independent real-mine evidence or a stable performance estimate.
- The saved checkpoint reloaded successfully for NPZ and PNG prediction. RGB preprocessed from the same two sources produced identical probability arrays.
- MPS/CUDA branches are available but were not exercised in the local CPU demonstration. GitHub training checks are configured, but the new remote job has not run.

## Historical checks · 2.2.0

Checked locally on 2026-10-08 using Python 3.11.17, NumPy 2.4.6 and Pillow 12.3.0 on macOS ARM64.

- 26 tests passed, including seeded water diversity, explicit modes, zero river width, metadata geometry reconstruction and unchanged mining truth/DEM under water changes.
- Ruff 0.15.0 lint and formatting checks passed.
- 20 review scenes passed numerical, mask, PNG and provenance checks; all 20 replayed exactly.
- The paired 2.1/2.2 review scenes have identical mining truth, DEM, DEM error and noiseless reference phase. Only water geometry and dependent observations/validity change. This batch has no train/validation/test split.
- The new review batch contains 7 river, 5 ponds, 5 mixed and 3 dry scenes, without image selection or resampling. These counts test scene diversity, not regional water frequencies.
- Updated fixed numerical diagnostics are in docs/numerical_diagnostics.json.

The new water generator remains geometric and has not been derived from DEM drainage or validated against measured shorelines. No field calibration or real-mine network evaluation has been completed.

## Historical checks · 2.1.0

Checked locally on 2026-10-07 using Python 3.11.17, NumPy 2.4.6 and Pillow 12.3.0 on macOS ARM64.

- 23 tests passed (`python -m unittest discover -v`). These test mathematical/statistical properties and public-bundle separation, not real-mine prediction.
- Ruff lint and format checks passed; the browser script passed Node syntax checking.
- A newly generated 50-scene batch passed all stored-array, total/component-label, validity/coherence, PNG, version/checksum, exact per-category replay and split checks: train 40, validation 5, test 5.
- Optional PyTorch reading returned `[4,2,256,256]` phase inputs and `[4,1,256,256]` label/validity tensors.
- Fixed-scene diagnostics are saved in [docs/numerical_diagnostics.json](docs/numerical_diagnostics.json). For the default single face at 128px, 1600m and days 40/52, 32 strips differ from a 128-strip numerical reference by relative L2 ≈0.00067545 (0.067545%). This is discretization evidence for that scenario, not a mine accuracy estimate.
- The complex-pair population correlation error at γ=0.7 is ≈0.00028336 under the fixed diagnostic. At γ=0 the mean sample magnitude with 32 ensembles is ≈0.15685, demonstrating finite-sample positive bias. With one ensemble and no spatial average it is 1, confirming the degeneracy stated in MODEL.md.

No field calibration, measured-water classification, real DEM registration, site-held-out network evaluation or external scientific peer review has been completed. Other environments may require justified numerical tolerances. GitHub Actions is configured for Python 3.11–3.13; its remote runs will only be available after repository upload.
