# Dataset sampling and split policy

All listed random draws are uniform unless otherwise stated. `draw_sample` is the executable source of this policy. Each scene uses a stream keyed by the base seed, scene index and stream identifier 1201. The observation seed is a sampled uint32.

| Parameter | Distribution |
|---|---|
| First observation day | 25–70 days |
| Observation interval | 6–18 days |
| Atmospheric equivalent LOS standard deviation | 0.5–4 mm |
| East-west orbit ramp | −5–5 cycles across scene |
| Initial land coherence | 0.70–0.98 |
| Water coherence | 0–0.08 |
| Synthetic relief | 30–350 m |
| DEM-error standard deviation | 0–12 m |
| Low-coherence patch enabled | Bernoulli probability 0.25 |
| Panel length / width / depth | 220–540 / 100–220 / 160–360 m |
| Panel bearing | −180–180° |
| Advance speed | 4–10 m/day |
| Seam thickness / subsidence factor | 1–3 m / 0.4–0.8 |
| Response time | 50–130 days |
| Extraction start day | 0–15 days; second staggered face: first observation day + (−5 to 3) |
| Panel midpoint | Each coordinate within ±0.12×extent |
| Separation | 0.07–0.14×extent; separated category 0.28–0.40×extent |

Legacy circular-noise spread is also drawn at 0.15–0.9 rad, but has no effect when the default complex-observation model is enabled. Frame, extent, radar geometry, spatial window, independent observation count, label/validity thresholds and the remaining settings are inherited from the normalized base request.

Categories cycle equally through negative, single, separated, overlapping and staggered. The last two names describe parameter construction, not guaranteed threshold overlap. Use actual `overlap_pixels` and masks for analysis.

Within each category, shuffled scene indices are divided into test/validation/train with `max(1, round(category_count*0.1))` held out to each of test and validation. For 50 scenes this gives 40/5/5. For other counts, rounding changes the exact ratio.

`group_id` hashes full panel parameters and frame/observation-time/segment settings while excluding noise and threshold settings. It conservatively groups changed noise/label versions of the same geometry. Single exports and batch metadata carry this identity. The assignment is local to each export; generating another dataset with a different count can change split assignment. Before combining datasets, run `validate_dataset.py DIR_A DIR_B` to check cross-split group repeats. The fingerprint detects exact geometry/time reuse, not spatially similar scenarios or real-image tile leakage.

Use an independently held-out real mine/site/time interval for real-world evaluation. Pixels from the same real interferogram and overlapping tiles require their own grouping policy. Current data provide synthetic labels only.
