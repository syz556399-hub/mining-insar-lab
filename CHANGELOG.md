# Changelog

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
