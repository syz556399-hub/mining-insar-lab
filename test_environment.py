import unittest

import numpy as np

from mine_model import default_request, simulate, wrap
from scene_environment import local_mean


class EnvironmentTests(unittest.TestCase):
    def base(self):
        r = default_request()
        r["settings"].update(size=128)
        return r

    def test_zero_baseline(self):
        r = self.base()
        r["settings"]["baseline_m"] = 0
        a, _ = simulate(r)
        self.assertFalse(a["topographic_phase_rad"].any())
        self.assertFalse(a["residual_topographic_phase_rad"].any())

    def test_perfect_dem_removal(self):
        r = self.base()
        r["settings"].update(dem_error_m=0, raw_topography=False)
        a, _ = simulate(r)
        self.assertTrue(a["topographic_phase_rad"].any())
        self.assertFalse(a["applied_topographic_phase_rad"].any())
        r["settings"]["raw_topography"] = True
        b, _ = simulate(r)
        np.testing.assert_array_equal(
            b["applied_topographic_phase_rad"], b["topographic_phase_rad"]
        )
        np.testing.assert_array_equal(a["delta_down_m"], b["delta_down_m"])

    def test_water_changes_quality_not_truth(self):
        r = self.base()
        a, _ = simulate(r)
        water = a["water_mask"].astype(bool)
        self.assertTrue(water.any())
        self.assertFalse(a["valid_mask"][water].any())
        self.assertLess(
            a["estimated_coherence"][water].mean(), a["estimated_coherence"][~water].mean()
        )
        r["settings"]["water_enabled"] = False
        b, _ = simulate(r)
        np.testing.assert_array_equal(a["mask"], b["mask"])
        np.testing.assert_array_equal(a["delta_down_m"], b["delta_down_m"])

    def test_ideal_complex_phase(self):
        r = self.base()
        r["settings"].update(
            land_coherence=1,
            water_enabled=False,
            slope_coherence_loss=0,
            decorrelation_days=0,
            spatial_window=1,
        )
        a, _ = simulate(r)
        np.testing.assert_allclose(
            wrap(a["wrapped_phase_rad"] - wrap(a["reference_phase_rad"])), 0, atol=5e-6
        )
        np.testing.assert_allclose(a["estimated_coherence"], 1, atol=1e-6)

    def test_spatial_filter(self):
        x = np.arange(36).reshape(6, 6).astype(float)
        y = local_mean(x, 3)
        p = np.pad(x, 1, mode="reflect")
        expected = np.array([[p[i : i + 3, j : j + 3].mean() for j in range(6)] for i in range(6)])
        np.testing.assert_allclose(y, expected)

    def test_single_sample_coherence_bias(self):
        r = self.base()
        r["settings"].update(looks=1, spatial_window=1, land_coherence=0, water_coherence=0)
        a, m = simulate(r)
        np.testing.assert_allclose(a["estimated_coherence"], 1, atol=1e-6)
        self.assertTrue(any("恒为 1" in x for x in m["warnings"]))

    def test_repeatable_environment(self):
        r = self.base()
        a, _ = simulate(r)
        b, _ = simulate(r)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])


if __name__ == "__main__":
    unittest.main(verbosity=2)
