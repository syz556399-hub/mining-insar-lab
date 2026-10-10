import unittest

import numpy as np

from mine_model import Settings, default_request, grid, parse_request, simulate, wrap
from scene_environment import local_mean, rasterize_water, water_layout


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

    def test_random_water_varies_across_seeds(self):
        # Regression: a fixed river + fixed ellipse must not pass this check.
        masks, modes, counts = set(), set(), set()
        for seed in range(40):
            s = Settings(size=128, seed=seed)
            east, north = grid(s)
            layout = water_layout(s)
            mask = rasterize_water(layout, east, north, s.extent_m)
            modes.add(layout["sampled_mode"])
            counts.add(len(layout["ponds"]))
            if mask.any():
                masks.add(mask.tobytes())
            if layout["sampled_mode"] == "dry":
                self.assertFalse(mask.any())
        self.assertEqual(modes, {"dry", "river", "ponds", "mixed"})
        self.assertGreater(len(masks), 25)
        self.assertGreater(len(counts), 2)

    def test_explicit_water_modes_and_zero_river_width(self):
        for mode in ("river", "ponds", "mixed"):
            s = Settings(size=128, water_mode=mode)
            layout = water_layout(s)
            self.assertEqual(layout["sampled_mode"], mode)
            self.assertEqual(bool(layout["rivers"]), mode in ("river", "mixed"))
            self.assertEqual(bool(layout["ponds"]), mode in ("ponds", "mixed"))
            east, north = grid(s)
            self.assertTrue(rasterize_water(layout, east, north, s.extent_m).any())
        s = Settings(size=128, water_mode="river", river_width_m=0)
        east, north = grid(s)
        self.assertFalse(rasterize_water(water_layout(s), east, north, s.extent_m).any())

    def test_water_mode_validation_and_recording(self):
        request = self.base()
        for invalid in (False, 1, None, "unknown"):
            request["settings"]["water_mode"] = invalid
            with self.assertRaises(ValueError):
                parse_request(request)
        request["settings"]["water_mode"] = "mixed"
        a, metadata = simulate(request)
        self.assertEqual(metadata["water_scene"]["sampled_mode"], "mixed")
        s, _ = parse_request(request)
        east, north = grid(s)
        np.testing.assert_array_equal(
            a["water_mask"], rasterize_water(metadata["water_scene"], east, north, s.extent_m)
        )
        request["settings"]["water_mode"] = "river"
        b, _ = simulate(request)
        for field in ("delta_down_m", "mask", "dem_m", "dem_error_m"):
            np.testing.assert_array_equal(a[field], b[field])


if __name__ == "__main__":
    unittest.main(verbosity=2)
