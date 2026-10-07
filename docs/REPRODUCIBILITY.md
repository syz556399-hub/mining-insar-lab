# Reproducibility

Use Python 3.11 with `requirements-lock.txt` for the locally checked numerical dependency versions. The broader `requirements.txt` supports fresh installation but does not promise identical random draws across dependency versions. Torch is optional and does not affect simulation.

Each export records normalized parameters, engine ID, schema version, Python/NumPy/Pillow/software versions, request SHA-256, geometry/time group ID, array shapes/dtypes/units and the validity rule. Exact replay is attempted only with matching recorded runtime and engine; a mismatch produces a clear error. `--no-replay` still verifies saved arrays and labels. Cross-platform libm/FFT differences may prevent bitwise identity even at matching versions; report the platform and use scientifically justified tolerances when comparing across machines.

Random streams are keyed by the seed and separate fixed identifiers: atmospheric plane waves 701; legacy circular observations 907; dataset sampling 1201; terrain 2301; water geometry 2302; DEM error 2303; complex observations 2407. These identifiers are part of the current engine contract.

Tests distinguish numerical correctness from field validity. They cover analytical rectangle limits, superposition, LOS signs, chronological monotonicity, rotation, segmented-time convergence, zero baseline, perfect DEM removal, water/noise independence, population complex correlation and sample-coherence bias. No real-mine RMSE, detection accuracy or generalization result has been established.

A defensible research evaluation should report: site and instrument geometry; fitted subsidence/time parameters; independent leveling/GNSS or validated InSAR observations; realistic train/validation/test site/time groups; sensitivity to label threshold, DEM error, coherence and window; synthetic-to-real performance with uncertainty; baseline and ablation comparisons. The repository currently supplies a testable synthetic model, not those completed experiments.
