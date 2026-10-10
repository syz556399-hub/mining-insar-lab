import unittest

import numpy as np

from mine_model import default_request, simulate
from task_labels import fringe_support
from transfer_design import matched_scene


class TaskLabelTests(unittest.TestCase):
    def test_support_is_sign_invariant_and_not_wrapped(self):
        phase = np.array([[0.0, np.pi / 2, 4 * np.pi, -4 * np.pi]], np.float32)
        np.testing.assert_array_equal(fringe_support(phase), [[0, 1, 1, 1]])
        np.testing.assert_array_equal(fringe_support(phase), fringe_support(-phase))
        self.assertFalse(fringe_support(np.zeros((3, 3))).any())

    def test_new_label_keeps_physical_truth_and_noise_independence(self):
        req = default_request()
        req["settings"]["size"] = 128
        a, _ = simulate(req)
        req["settings"]["atmosphere_mm"] = 15
        req["settings"]["water_enabled"] = False
        b, _ = simulate(req)
        np.testing.assert_array_equal(a["fringe_mask"], b["fringe_mask"])
        np.testing.assert_array_equal(a["mask"], a["delta_down_m"] >= 0.005)
        self.assertTrue(a["fringe_valid_mask"].all())
        req["settings"]["fringe_threshold_rad"] = 3.0
        c, _ = simulate(req)
        np.testing.assert_array_equal(b["mask"], c["mask"])
        self.assertTrue((c["fringe_mask"] <= b["fringe_mask"]).all())

    def test_matched_generation_is_repeatable_and_negative_has_no_target(self):
        req = default_request()
        req["settings"]["size"] = 128
        profile = dict(
            size=128,
            positive_probability=0.0,
            positive_area_samples_px=[300],
            profile_sha256="test",
        )
        a, m = matched_scene(req, profile, 0)
        b, n = matched_scene(req, profile, 0)
        self.assertFalse(a["fringe_mask"].any())
        self.assertFalse(a["mask"].any())
        np.testing.assert_array_equal(a["wrapped_phase_rad"], b["wrapped_phase_rad"])
        self.assertEqual(m["request"], n["request"])
        profile["positive_probability"] = 1.0
        a, m = matched_scene(req, profile, 1)
        self.assertEqual(m["pixel_design"]["requested_support_area_px"], 300)
        self.assertTrue(400 <= m["request"]["settings"]["extent_m"] <= 6000)
        np.testing.assert_array_equal(a["fringe_mask"], fringe_support(a["deformation_phase_rad"]))


class SamplingTests(unittest.TestCase):
    def test_training_scene_sampler_is_seeded_and_leaves_rng_untouched(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("Install requirements-training.txt for sampler checks")

        from train import balanced_scene_sampler

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for i in range(10):
                mask = np.full((8, 8), i == 0, np.uint8)
                np.savez(root / f"{i}.npz", fringe_mask=mask, fringe_valid_mask=np.ones_like(mask))
                rows.append(dict(arrays=f"{i}.npz", split="train"))
            data = SimpleNamespace(root=root, rows=rows, target_mode="fringe")
            torch.manual_seed(42)
            state = torch.get_rng_state().clone()
            sampler, info = balanced_scene_sampler(data, 0.8, 42)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            sample = list(sampler)
            other, _ = balanced_scene_sampler(data, 0.8, 42)
            self.assertEqual(sample, list(other))
            self.assertEqual(len(sample), 10)
            weights = sampler.weights.numpy()
            self.assertAlmostEqual(weights[0] / weights.sum(), 0.8)
            self.assertEqual(info["positive_training_scenes"], 1)
            rows[0]["split"] = "val"
            with self.assertRaises(ValueError):
                balanced_scene_sampler(data, 0.8, 42)


if __name__ == "__main__":
    unittest.main()
