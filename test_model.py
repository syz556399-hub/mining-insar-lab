import copy
import io
import unittest
import zipfile

import numpy as np

from data_io import single_archive
from mine_model import (
    Face,
    Settings,
    default_request,
    grid,
    panel_epochs,
    parse_request,
    simulate,
    strip_integral,
    wrap,
)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.r = default_request()
        self.r["settings"].update(
            size=128,
            complex_observation=False,
            terrain_enabled=False,
            water_enabled=False,
            baseline_m=0,
        )

    def test_zero(self):
        for f in self.r["faces"]:
            f["enabled"] = False
        a, _ = simulate(self.r)
        self.assertFalse(a["delta_down_m"].any())
        self.assertFalse(a["mask"].any())
        self.r = default_request()
        self.r["settings"].update(size=128, day_after=40)
        a, _ = simulate(self.r)
        self.assertFalse(a["delta_los_m"].any())

    def test_superposition(self):
        a, _ = simulate(self.r)
        singles = []
        for i in range(2):
            r = copy.deepcopy(self.r)
            r["faces"][1 - i]["enabled"] = False
            singles.append(simulate(r)[0])
        for k in ["delta_down_m", "delta_east_m", "delta_north_m", "delta_los_m"]:
            np.testing.assert_allclose(a[k], singles[0][k] + singles[1][k], atol=4e-8)
        self.r["settings"]["incidence_deg"] = 0
        a, _ = simulate(self.r)
        np.testing.assert_array_equal(a["delta_los_m"], a["delta_down_m"])

    def test_rectangle(self):
        f = Face(bearing_deg=0, response_days=1)
        _, a = panel_epochs(f, 128, 1600, 32, 0, 2000)
        e, n = grid(Settings(size=128))
        r = f.depth_m / f.influence_tangent
        expected = (
            f.thickness_m
            * f.subsidence_factor
            * strip_integral(e - f.east_m, -f.length_m / 2, f.length_m / 2, r)
            * strip_integral(n - f.north_m, -f.width_m / 2, f.width_m / 2, r)
        )
        np.testing.assert_allclose(a, expected, atol=1e-14)

    def test_time_convergence(self):
        changes = []
        for segments in [32, 64, 128]:
            a, b = panel_epochs(Face(), 128, 1600, segments, 40, 52)
            changes.append(b - a)
        errors = [
            np.linalg.norm(x - changes[-1]) / np.linalg.norm(changes[-1]) for x in changes[:2]
        ]
        self.assertLess(errors[0], 0.02)
        self.assertLess(errors[1], errors[0])

    def test_seed_and_truth(self):
        a, _ = simulate(self.r)
        b, _ = simulate(self.r)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        self.r["settings"].update(seed=321, atmosphere_mm=10, phase_spread_rad=1)
        b, _ = simulate(self.r)
        for k in ["mask", "delta_down_m", "delta_los_m"]:
            np.testing.assert_array_equal(a[k], b[k])
        self.assertFalse(np.array_equal(a["wrapped_phase_rad"], b["wrapped_phase_rad"]))

    def test_clean_phase(self):
        self.r["settings"].update(
            atmosphere_mm=0, orbit_cycles=0, phase_spread_rad=0, disturbed_patch=False
        )
        a, _ = simulate(self.r)
        np.testing.assert_allclose(
            wrap(a["wrapped_phase_rad"] - wrap(a["deformation_phase_rad"])), 0, atol=3e-6
        )
        np.testing.assert_array_equal(a["mask"], a["delta_down_m"] >= 0.005)

    def test_validation(self):
        for k, v in [
            ("day_after", -100),
            ("looks", 2.5),
            ("wavelength_m", 0),
            ("seed", True),
            ("extent_m", float("nan")),
        ]:
            r = copy.deepcopy(self.r)
            r["settings"][k] = v
            with self.assertRaises(ValueError):
                parse_request(r)

    def test_archive(self):
        with zipfile.ZipFile(io.BytesIO(single_archive(self.r))) as z:
            self.assertTrue({"arrays.npz", "metadata.json", "mask.png"} <= set(z.namelist()))
            self.assertIsNone(z.testzip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
