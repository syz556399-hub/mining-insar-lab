import unittest

import numpy as np

from mine_model import simulate

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from detection_first import FringeDetector, clear_request, detection_target, make_scene


@unittest.skipIf(torch is None, "Install requirements-training.txt for detection checks")
class DetectionFirstTests(unittest.TestCase):
    def test_target_zero_sign_scale_and_bad_input(self):
        phase = np.array([[0, -2, 4]], dtype=np.float32)
        np.testing.assert_allclose(detection_target(phase), [[0, 0.5, 1]])
        np.testing.assert_array_equal(detection_target(phase), detection_target(-phase * 3))
        self.assertFalse(detection_target(np.zeros((4, 4))).any())
        with self.assertRaises(ValueError):
            detection_target(np.array([[np.nan]]))

    def test_full_resolution_backward_on_odd_image(self):
        torch.set_num_threads(2)
        model = FringeDetector()
        x = torch.randn(2, 2, 33, 47)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 1, 33, 47))
        y.square().mean().backward()
        self.assertTrue(
            all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        )

    def test_scene_truth_independent_of_background_and_negative_zero(self):
        request = clear_request(20261019, 0)
        request["settings"]["size"] = 128
        a, _ = simulate(request)
        request["settings"]["orbit_cycles"] = -5
        request["settings"]["atmosphere_mm"] = 10
        b, _ = simulate(request)
        np.testing.assert_array_equal(
            detection_target(a["deformation_phase_rad"]),
            detection_target(b["deformation_phase_rad"]),
        )
        self.assertFalse(np.array_equal(a["wrapped_phase_rad"], b["wrapped_phase_rad"]))
        negative, _ = make_scene(20261039, 20, True)
        self.assertFalse(negative["detection_target"].any())
        self.assertGreater(float(negative["wrapped_phase_rad"].std()), 0.1)


if __name__ == "__main__":
    unittest.main()
