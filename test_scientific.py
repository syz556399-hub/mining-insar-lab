"""Analytical/statistical properties, distinct from real-world model validation."""

import copy
import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from mine_model import Face, Settings, default_request, grid, panel_epochs, simulate
from provenance import fingerprint, group_identity
from scene_environment import environment, observe
from validate_dataset import audit_splits


class ScientificProperties(unittest.TestCase):
    def test_water_toggle_preserves_terrain_and_dem_error(self):
        s = Settings(size=128)
        e, n = grid(s)
        a = environment(s, e, n)
        b = environment(replace(s, water_enabled=False), e, n)
        for key in (
            "dem_m",
            "dem_error_m",
            "topographic_phase_rad",
            "residual_topographic_phase_rad",
        ):
            np.testing.assert_array_equal(a[key], b[key])

    def test_positive_baseline_sign_and_scaling(self):
        s = Settings(size=128)
        e, n = grid(s)
        a = environment(s, e, n)
        b = environment(replace(s, baseline_m=-s.baseline_m), e, n)
        c = environment(replace(s, baseline_m=2 * s.baseline_m), e, n)
        np.testing.assert_allclose(a["topographic_phase_rad"], -b["topographic_phase_rad"])
        np.testing.assert_allclose(c["topographic_phase_rad"], 2 * a["topographic_phase_rad"])

    def test_rotation_and_time_monotonicity(self):
        f = Face(east_m=0, north_m=0, bearing_deg=0)
        before, after = panel_epochs(f, 128, 1600, 64, 40, 52)
        _, later = panel_epochs(f, 128, 1600, 64, 52, 100)
        self.assertTrue((after >= before).all())
        self.assertTrue((later >= after).all())
        _, rotated = panel_epochs(replace(f, bearing_deg=90), 128, 1600, 64, 40, 52)
        np.testing.assert_allclose(rotated, np.rot90(after), atol=2e-14)

    def test_complex_pair_population_correlation(self):
        s = Settings(size=128, looks=32, spatial_window=1, disturbed_patch=False)
        e, n = grid(s)
        shape = e.shape
        gamma, phi = 0.7, 0.8
        env = {"model_coherence": np.full(shape, gamma), "backscatter_power": np.ones(shape)}
        cross, _, _, p1, p2 = observe(np.full(shape, phi), s, env, e, n)
        rho = cross.mean() / np.sqrt(p1.mean() * p2.mean())
        self.assertLess(abs(rho - gamma * np.exp(1j * phi)), 0.01)

    def test_identity_tracks_geometry_not_noise(self):
        r = default_request()
        changed = copy.deepcopy(r)
        changed["settings"]["seed"] += 1
        changed["settings"]["target_threshold_mm"] += 1
        self.assertEqual(group_identity(r), group_identity(changed))
        changed["faces"][0]["depth_m"] += 1
        self.assertNotEqual(group_identity(r), group_identity(changed))
        self.assertNotEqual(fingerprint(r), fingerprint(changed))

    def test_cross_dataset_leakage_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            roots = []
            for index, split in enumerate(("train", "test")):
                root = Path(folder) / str(index)
                root.mkdir()
                roots.append(root)
                with (root / "manifest.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["group_id", "split"])
                    writer.writeheader()
                    writer.writerow({"group_id": "same_geometry", "split": split})
            with self.assertRaises(ValueError):
                audit_splits(roots)

    def test_provenance_and_configurable_validity(self):
        r = default_request()
        r["settings"].update(size=128, valid_coherence_threshold=0.8)
        arrays, metadata = simulate(r)
        np.testing.assert_array_equal(
            arrays["valid_mask"], (arrays["water_mask"] == 0) & (arrays["model_coherence"] >= 0.8)
        )
        self.assertEqual(metadata["request_sha256"], fingerprint(metadata["request"]))
        self.assertEqual(metadata["array_schema"]["delta_los_m"]["unit"], "m")
        self.assertEqual(metadata["array_schema"]["reference_phase_rad"]["unit"], "rad")
        json.dumps(metadata, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
