# Source publication preparation

The public snapshot produced by `prepare_release.py` contains current code, configuration examples, MIT license, citation metadata, scientific specifications, tests and CI. It excludes personal BMP/reference records, generated datasets, old exploratory snapshots, local virtual environments and caches. Use the generated folder as the repository root; do not upload the full desktop working folder containing local data.

Before an actual release, maintainers should use verified contributor names in the copyright and CITATION.cff, add the real repository URL, record the release commit/tag and retain all MIT notices. The collective contributor entry is not an assertion of a named person's authorship. There is no software DOI or companion validation paper.

A suitable initial public description is: “A reproducible research prototype for synthetic mining subsidence and statistical InSAR observations, with explicit assumptions and numerical tests.” Claims of real-mine accuracy or improved synthetic-to-real generalization require independent experimental evidence.

Current version is 2.1.0. The package is prepared locally; no GitHub repository has been created or uploaded by this workflow.
