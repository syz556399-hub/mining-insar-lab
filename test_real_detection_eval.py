import base64
import io
import unittest

import numpy as np
from PIL import Image

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from real_detection_eval import object_coverage, threshold_counts
    from segmentation import tile_starts, tiled_probability


@unittest.skipIf(torch is None, "Install requirements-training.txt for detection checks")
class RealDetectionEvaluationTests(unittest.TestCase):
    def test_fixed_threshold_counts_include_zero_brightness_truth(self):
        score = np.array([[0.25, 0.9], [0.0, 0.1]])
        truth = np.array([[True, False], [True, False]])
        self.assertEqual(threshold_counts(score, truth), [1, 1, 1, 1])

    def test_original_small_and_overlapping_shape_records_are_preserved(self):
        buffer = io.BytesIO()
        Image.fromarray(np.ones((1, 1), np.uint8)).save(buffer, format="PNG")
        shape = dict(
            label="1",
            shape_type="mask",
            points=[[1, 1], [1, 1]],
            mask=base64.b64encode(buffer.getvalue()).decode(),
        )
        doc = dict(imageWidth=3, imageHeight=3, shapes=[shape, shape])
        score = np.zeros((3, 3), np.float32)
        score[1, 1] = 0.25
        self.assertEqual(object_coverage(score, doc), [1.0, 1.0])

    def test_progress_preserves_native_prediction_and_counts_edge_windows(self):
        class Identity(torch.nn.Module):
            def forward(self, x):
                return x[:, :1]

        channels = np.random.default_rng(1).normal(size=(2, 21, 29)).astype(np.float32)
        kwargs = dict(tile_size=16, stride=12, batch_size=2)
        progress = []
        a = tiled_probability(Identity(), channels, torch.device("cpu"), **kwargs)
        b = tiled_probability(
            Identity(), channels, torch.device("cpu"), progress=progress.append, **kwargs
        )
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a, 1 / (1 + np.exp(-channels[0])), atol=1e-6)
        self.assertEqual(progress[-1], len(tile_starts(21, 16, 12)) * len(tile_starts(29, 16, 12)))
        self.assertEqual(progress, sorted(set(progress)))


if __name__ == "__main__":
    unittest.main()
