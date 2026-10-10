# Parameters and units

Generated from current dataclasses/defaults and parser ranges. Allowed ranges are numerical design bounds, not empirically calibrated mine ranges.

| Group | Key | Unit | Default | Accepted range |
|---|---|---|---|---|
| Settings | `size` | pixel | `256` | 128 / 256 / 512 |
| Settings | `extent_m` | m | `1600.0` | (400, 6000) |
| Settings | `seed` | 1 | `20261007` | (0, 4294967295) |
| Settings | `day_before` | day | `40.0` | (-365, 2000) |
| Settings | `day_after` | day | `52.0` | (-365, 2000) |
| Settings | `segments` | 1 | `32` | (8, 128) |
| Settings | `wavelength_m` | m | `0.055` | (0.02, 0.25) |
| Settings | `incidence_deg` | degree | `33.0` | (0, 80) |
| Settings | `radar_azimuth_deg` | degree | `90.0` | (0, 360) |
| Settings | `horizontal_factor` | 1 | `0.2` | (0, 0.5) |
| Settings | `target_threshold_mm` | mm | `5.0` | (0.1, 100) |
| Settings | `atmosphere_mm` | mm LOS standard deviation | `1.5` | (0, 20) |
| Settings | `atmosphere_scale_m` | m | `600.0` | (50, 6000) |
| Settings | `orbit_cycles` | cycles/scene | `3.0` | (-30, 30) |
| Settings | `phase_spread_rad` | rad (legacy only) | `0.45` | (0, 3) |
| Settings | `looks` | independent realization count | `4` | (1, 32) |
| Settings | `disturbed_patch` | switch | `False` | boolean |
| Settings | `complex_observation` | switch | `True` | boolean |
| Settings | `terrain_enabled` | switch | `True` | boolean |
| Settings | `water_enabled` | switch | `True` | boolean |
| Settings | `water_mode` | category | `random` | random / river / ponds / mixed |
| Settings | `raw_topography` | switch | `False` | boolean |
| Settings | `relief_m` | m (max-min) | `180.0` | (0, 1500) |
| Settings | `river_width_m` | m (reference width; actual width varies) | `65.0` | (0, 500) |
| Settings | `dem_error_m` | m (standard deviation) | `5.0` | (0, 100) |
| Settings | `baseline_m` | m | `100.0` | (-1000, 1000) |
| Settings | `slant_range_m` | m | `850000.0` | (500000, 1500000) |
| Settings | `height_atmosphere_mm` | mm LOS / 100m elevation | `1.0` | (-20, 20) |
| Settings | `land_coherence` | 1 | `0.92` | (0, 1) |
| Settings | `water_coherence` | 1 | `0.03` | (0, 1) |
| Settings | `decorrelation_days` | day (0 disables) | `180.0` | (0, 1000000) |
| Settings | `slope_coherence_loss` | 1 | `0.3` | (0, 5) |
| Settings | `spatial_window` | map pixels (odd square width) | `3` | 1 / 3 / 5 / 7 / 9 |
| Settings | `valid_coherence_threshold` | 1 | `0.2` | (0, 1) |
| Face | `enabled` | switch | `True` | boolean |
| Face | `east_m` | m east | `-110.0` | (-6000, 6000) |
| Face | `north_m` | m north | `0.0` | (-6000, 6000) |
| Face | `length_m` | m | `440.0` | (20, 2000) |
| Face | `width_m` | m | `160.0` | (20, 1000) |
| Face | `bearing_deg` | degree, CCW from East | `20.0` | (-180, 180) |
| Face | `depth_m` | m | `240.0` | (50, 1200) |
| Face | `thickness_m` | m | `2.2` | (0.1, 8) |
| Face | `subsidence_factor` | 1 | `0.65` | (0.05, 1) |
| Face | `influence_tangent` | 1 | `2.0` | (0.5, 5) |
| Face | `advance_m_day` | m/day | `7.0` | (0.2, 30) |
| Face | `start_day` | day | `0.0` | (-365, 2000) |
| Face | `response_days` | day | `80.0` | (1, 500) |

Additional constraints: day_after ≥ day_before; 1–2 faces; nonzero baseline requires incidence ≥1°; booleans cannot replace numbers; seed/looks/segments/size/window require integer types. Unknown fields and NaN/Infinity are rejected.

Canonical JSON stores physical floats as float values; UI mm values are converted to m before generation. A hidden advanced field remains active at its current value.