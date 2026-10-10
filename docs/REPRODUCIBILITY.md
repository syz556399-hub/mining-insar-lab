# Simulator 2.0 reproducibility

The product release is 2.0.0; numerical engine IDs remain independent and are recorded per sample. The rebuilt workflow is defined in SIMULATOR_REBOOT.md. Start with the built-in display profile or record a local development-only indexed BMP profile; do not fit it on held-out real evaluation images.

Keep each review run and its exact masks/approval receipts. Batch exports include the immutable sampling plan, accepted template hashes, seed and configuration for each new scene, source group/split, source-code hashes, numerical arrays and label hashes. Resume events record the reason and source receipt; they never silently replace completed samples. New labels remain drafts even when generated from approved templates. Dependency versions can change random draws, so preserve the runtime and code snapshot for exact local replay.

Use simulator_batch.py validate on the complete batch to check stored arrays, file hashes, native image/label alignment, group/split separation and candidate semantics. Validation does not approve visual boundaries. Final training masks may differ from initial candidate arrays after human correction. Group all subsequent crops/variants under the original source; keep real evaluation independent by site/date.

The following details describe the retained legacy engine and its separate export/validation contract, rather than the rebuilt batch format.

# Reproducibility

Use Python 3.11 with `requirements-lock.txt` for the locally checked numerical dependency versions. The broader `requirements.txt` supports fresh installation but does not promise identical random draws across dependency versions. Torch is optional and does not affect simulation.

Each export records normalized parameters, engine ID, schema version, Python/NumPy/Pillow/software versions, request SHA-256, geometry/time group ID, array shapes/dtypes/units and the validity rule. Exact replay is attempted only with matching recorded runtime and engine; a mismatch produces a clear error. `--no-replay` still verifies saved arrays and labels. Cross-platform libm/FFT differences may prevent bitwise identity even at matching versions; report the platform and use scientifically justified tolerances when comparing across machines.

Random streams are keyed by the seed and separate fixed identifiers: atmospheric plane waves 701; legacy circular observations 907; dataset sampling 1201; terrain 2301; water geometry 2302; DEM error 2303; complex observations 2407. These identifiers are part of the current engine contract.

Tests distinguish numerical correctness from field validity. They cover analytical rectangle limits, superposition, LOS signs, chronological monotonicity, rotation, segmented-time convergence, zero baseline, perfect DEM removal, water/noise independence, population complex correlation and sample-coherence bias. No real-mine RMSE, detection accuracy or generalization result has been established.

A defensible research evaluation should report: site and instrument geometry; fitted subsidence/time parameters; independent leveling/GNSS or validated InSAR observations; realistic train/validation/test site/time groups; sensitivity to label threshold, DEM error, coherence and window; synthetic-to-real performance with uncertainty; baseline and ablation comparisons. The repository currently supplies a testable synthetic model, not those completed experiments.
