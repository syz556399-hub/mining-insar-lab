# Changelog

## 2.0.0 — 2026-10-10 — Simulator 2.0 public release

- Establish the rebuilt simulator as the public 2.0 product release; preserve earlier repository history and independent numerical engine identifiers.
- Make the 20-scene review desk the documented default entry point and macOS launcher. Add an optional local browser opening flag.
- Publish the independently implemented source/background/complex-observation model, exact label editing and hash-bound human review, and scalable batch generation with validation and explicit resume.
- Include the paginated batch viewer, source-group splits, LabelMe/array/PNG exports and an optional draft-aware PyTorch reader. New batch labels remain automatic candidates.
- Document installation, label semantics, training-data integration, synthetic DEM/water and the limits of physical/real-data claims. Preserve legacy experiments separately.
- Publish source and configuration only; exclude private real references, annotation records, generated datasets and model weights.

## Internal development — simulator review engine 3.2

- Align the default source and mild coordinate augmentation with the approved sample family. Retain an experimental domain with one broad bend, slow taper and shallow waist. Keep prior engines and source requests reproducible. No real contour or image patch enters synthetic samples.
- Add independently implemented Gaussian influence integration over a varying-width, offset mined domain, with 48 non-overlapping strips, saved node/strip geometry and explicit approximation limits. Preserve full saved 3.0 requests and approved sample files.
- Replace the overly circular S13/S16/S19 and exclude the original tiny S18. All 20 current images and exact masks are now explicitly accepted; prior versions remain available. No new network training or performance claim.
- Add bulk sampling around accepted configurations with fresh seeds, moderate parameter ranges, independent synthetic water occurrence, fixed scenario-stratified source splits, exact raster/LabelMe labels and a paginated local viewer. Automatic new labels remain drafts and require explicit opt-in for experiment loading.
- Fix float32 cyclic color lookup endpoint overflow without changing numerical phase or labels. Add explicit plan-bound generation resume with unchanged-file hashes and source receipts.
- Retain exact reviewed raster masks, hash-bound approval, prior versions, BMP display statistics and matched-brightness alternate views. Export reviewed samples separately from drafts.
- Verify constant-width domain equivalence with the original rectangle integration, source units, deterministic generation, exact LabelMe export and preservation of accepted sample hashes.

## Archived development — 2026-10-09 — detection-first pilot

- Add frozen native-resolution real-image comparison against exact final LabelMe records, source alignment checks, unchanged-input hashes and complete per-tile reports. No threshold tuning or retraining.

- Extend with a 144-scene, three-difficulty synthetic detection comparison, independent held-out geometry, phase/spatial augmentation and larger-context regression.
- Add fixed-threshold candidate analysis windows and crop export, explicitly separate from final segmentation and physical instance counting.
- Record sample checksums and the pre-training protocol; compare both models on every held-out scene and retain negative false-alarm results.

- Add independently implemented full-resolution residual/dilated heatmap detector and conditional clear-positive working-face samples.
- Separate 20 positive previews from 8 background negatives; export reproducible physical requests and disjoint scene splits.
- Keep normalized deformation regression separate from experimental support masks and final real annotations.
- Show every held-out synthetic prediction at a fixed intensity scale and compare regression loss with an all-zero baseline. No real-data performance or phase-unwrapping claim.

## 2.9.0 — 2026-10-09 — simulator task alignment experiments

- Preserve physical masks and add separate clean-deformation phase support proxies and all-pixel task supervision. No observed noise enters the proxy label.
- Fit pixel-area and occurrence statistics from development annotations; record bounded simulator extent adjustments and achieved, rather than assumed, target sizes. No real image texture is copied into synthetic training scenes.
- Add explicit physical/fringe training targets and equal-budget synthetic-only comparisons; retain model selection on synthetic validation splits.
- Add physical/proxy label views and a simulator experiment page with fixed first-20 generated examples.
- Update array-schema engine version to 2.3; displacement and observation equations remain unchanged.

## 2.8.0 — 2026-10-09 — final-label transfer baseline

- Preserve exact per-file final LabelMe masks and original object records, with full source-tile pixel alignment checks. Do not equate connected regions with annotation objects.
- Add native-resolution real-label fine-tuning from a synthetic checkpoint, fixed baseline comparison, acquisition-date separation checks, validation selection and training traces.
- Report per-tile development metrics explicitly, including overlapping views; retain an untouched independent test outside this workflow.
- Add fixed-area-sample before/after comparison pages; retain final labels at zero display brightness and do not force model output to background there.
- Keep original label files unchanged. Semantic segmentation does not implement instance counting.

## 2.7.0 — 2026-10-09 — annotation and scale audit

- Add read-only LabelMe mask reconstruction, source-tile alignment checks, overlap disagreement flags and connected-region size diagnostics without small-region filtering.
- Add fixed-selection local review pages and downloadable, source-bound review records; pending labels never automatically enter training.
- Add frozen-checkpoint scale comparisons, provisional-scope metrics and size-group coverage diagnostics.
- Add optional batched inference while retaining single-tile defaults and spatial fusion semantics.
- Document the target definition, uncertainty and synthetic-only versus future real-fine-tuning protocol. Physical simulation and synthetic label definitions remain unchanged.

## 2.6.0 — 2026-10-08 — controlled retraining and learning explanation

- Add epoch-seeded, equal-budget training draws per original scene: uniform or training-mask-balanced, without changing held-out sampling.
- Exclude all-invalid patches and ignore invalid positive pixels when assigning training patch pools; retain complete negative scenes.
- Record the actual first valid training batch's shapes, loss, gradient norm and parameter change without introducing extra updates.
- Generate a standalone learning page from run logs, with interactive training/validation curves, actual settings and fixed synthetic examples.
- Expose learning pages alongside local predictions and add synthetic target/validity panels for inspection.
- Default the complete runner's prediction context to its training contexts; four-scale averaging remains explicitly selectable and is not presumed superior.
- Add a 256-pixel configuration for larger independent-scene experiments. Physical equations and label definitions remain unchanged.

## 2.5.0 — 2026-10-08 — scene-patch segmentation workflow

- Add aligned, overlapping patches with several context sizes; preserve each original scene's split and export crop manifests.
- Select and test patch-trained models after whole-scene overlap fusion, counting each valid source pixel once.
- Add explicit equal-weight multiscale prediction and full-resolution mask, overlay and boundary outputs; retain legacy single-scale checkpoint support.
- Add a complete synthetic generation/training/prediction runner and a local segmentation viewer with preview-only threshold controls.
- Add checks for crop alignment, split inheritance, overlapping scene evaluation, multiscale edge coverage and mask boundaries.
- Preserve physical simulation equations and the vertical-increment label definition. Pixel-context scales do not establish physical calibration or real-image accuracy.

## 2.4.0 — 2026-10-08 — display encoding and transfer experiments

- Add a validated, explicitly selected adapter for original indexed 16×16 cyclic-color/brightness display tables; display codes are not calibrated physical phase.
- Add optional phase quantization and training-only random phase origin/sign augmentation; preserve labels and observed-input separation.
- Add a documented sparse-mine sampling profile with wider nuisance ranges; preserve standard sampling and physical equations.
- Record phase encoding and sampling profile in training/prediction provenance; reject incompatible input flags and unverified palettes.
- Exclude zero-brightness display pixels from adapted predictions without claiming water/coherence validity.
- Add adapter rejection, brightness invariance, wrapped quantization and profile reproducibility checks. Real-image agreement remains exploratory, not validated field accuracy.
- Add explicitly selected larger display context with circular phase averaging and output probabilities restored to original pixel coordinates; no geographic pixel-spacing claim.

## 2.3.0 — 2026-10-08 — synthetic segmentation baseline

- Add an independently implemented compact U-Net, RGB/phase input modes, masked BCE + Dice loss, training, and tiled prediction.
- Preserve the default two-channel PhaseDataset interface; add RGB input and strict validity checks for training.
- Select checkpoints using validation foreground IoU; evaluate synthetic test scenes only after training and selection.
- Record dataset manifest/config checksums, runtime, splits, metrics, checkpoint and held-out prediction examples.
- Add optional PyTorch dependencies and tests for ignored pixels, observed-input separation, checkpoint replay and edge coverage.
- Simulation engine and equations remain at working-face-integral-2.2. A small demonstration is not real-mine validation.

## 2.2.0 — 2026-10-08 — randomized water scenes

- Replace the fixed river and ellipse lake with seeded random river, irregular pond, mixed and dry scenes.
- Randomize river position, angle, meanders and width variation; randomize pond count, centers, axes and harmonic shorelines.
- Add `water_mode` selection and record full sampled geometry in `metadata.water_scene` for audit and exact replay.
- Preserve independent terrain/DEM-error streams and mining truth when water settings change.
- The water generator remains a geometric approximation, without DEM-derived drainage or calibrated water occurrence probabilities. Existing 2.1 exports remain untouched.

## 2.1.0 — 2026-10-07 — publication preparation

- Isolate terrain, water and DEM-error random streams; changing water no longer resamples DEM errors.
- Normalize floating-point configuration values; record schema, engine, runtime versions, units and request checksum.
- Record a geometry/time group fingerprint independent of noise and label threshold; add cross-dataset split audit.
- Make the model-coherence validity threshold explicit and configurable.
- Gate exact replay on the recorded runtime and engine. Saved numerical checks remain available for older exports.
- Add portable CLI/launcher, parameter documentation, scientific diagnostics, MIT license, citation metadata and CI configuration.
- Create a public source bundle that excludes personal BMP assets, absolute local paths, generated datasets, caches and old experiments.

Stochastic outputs differ from 2.0 because the independent random streams changed. Original 2.0 exports remain unchanged.

## 2.0.0 — local prototype

Synthetic DEM, geometric river/lake masks, terrain residual phase, correlated complex observations, sliding spatial averaging and estimated coherence. These components are synthetic/statistical approximations, not calibrated SAR imaging.

## 1.0.0 — local prototype

Rectangular working-face influence integral, segmented extraction time and exponential response, horizontal-motion approximation and LOS conversion.
# Unreleased · 2026-10-09

- Add independently implemented range segmentation with a clean-deformation phase auxiliary head.
- Add a matched two-arm synthetic-only experiment, split/content checks and frozen final-LabelMe diagnosis.
- Retain the synthetic support-proxy and display-code calibration limits explicitly.
- Skip optional learning tests when PyTorch is not installed; run them in the CPU training CI job.
