# Source publication · Simulator 2.0

The public product release is **Mining InSAR Lab 2.0**, package version **2.0.0**, tagged `v2.0.0` in <https://github.com/syz556399-hub/mining-insar-lab>. This release rebuilds the simulator workflow rather than continuing the previous learning experiments. Earlier 2.1–2.9 development identifiers remain in archived documentation and old records; the previous repository commit is preserved. Algorithm engine IDs (including `mine-simulator-reboot-3.2` and the legacy `working-face-integral-2.3`) are independent of the product release number and retain their numerical replay meanings.

`prepare_release.py` produces a source snapshot from an explicit allowlist: current code, configuration examples, MIT license, citation metadata, scientific specifications, tests and CI. It excludes personal BMP/LabelMe records, generated datasets, model weights, old exploratory snapshots, local virtual environments and caches. Use the generated folder as the repository root, not the full desktop working folder. The locally generated 500-scene dataset is not part of the source release.

The reproducible public entry point starts from the built-in display profile. A user may provide their own compatible indexed BMP locally. The public package does not include the developer's private display reference, reviewed images, annotations or approval receipts; each installation starts a new review. Do not manufacture approvals to bypass the batch-generation gate.

The collective contributor entry in CITATION.cff is not an assertion of any named person's authorship. There is no software DOI or companion validation paper. The MIT notice is retained. Scientific principles still require appropriate attribution; see REFERENCES.md.

A suitable description is: “An independently implemented mining InSAR simulator with reviewable fringe-range labels and reproducible synthetic dataset generation.” Claims of real-mine accuracy or improved synthetic-to-real generalization require independent evidence. The current label is a reviewed or draft phase-support proxy, not a calibrated field boundary.

Local checks and historical experiments are distinguished in VALIDATION.md. GitHub Actions status must be checked on the release commit; earlier remote checks are not evidence for this source snapshot.
