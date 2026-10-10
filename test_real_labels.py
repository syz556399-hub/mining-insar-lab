import base64
import io
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from annotation_audit import decode_annotation_mask

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from torch.utils.data import DataLoader

    from real_labels import RealCrops, disjoint_dates
    from real_report import block_probability
    from segmentation import SmallUNet, metrics_from_counts
    from train import run_epoch


@unittest.skipIf(torch is None, "Install requirements-training.txt for real-label checks")
class RealLabelTests(unittest.TestCase):
    def test_exact_mask_keeps_hole_and_tiny_object(self):
        pixels = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="PNG")
        shape = dict(
            label="1",
            shape_type="mask",
            points=[[0, 0], [2, 2]],
            mask=base64.b64encode(stream.getvalue()).decode(),
        )
        doc = dict(imageWidth=3, imageHeight=3, shapes=[shape])
        np.testing.assert_array_equal(decode_annotation_mask(doc), pixels.astype(bool))
        self.assertFalse(decode_annotation_mask(doc)[1, 1])

    def test_validation_preserves_all_pixels_once_and_masks_padding(self):
        target = np.zeros((13, 17), bool)
        target[12, 16] = True
        channels = np.stack([target, target]).astype(np.float32)
        scene = SimpleNamespace(tiles=[dict(target=target, channels=channels)])
        ds = RealCrops(scene, size=8, training=False)
        total = positive = 0
        for i in range(len(ds)):
            x, y, v = ds[i]
            total += v.sum().item()
            positive += (y * v).sum().item()
            np.testing.assert_array_equal((x[0].numpy() * v[0].numpy()) > 0, y[0].numpy() > 0)
        self.assertEqual(total, 13 * 17)
        self.assertEqual(positive, 1)
        train = RealCrops(scene, size=8, draws=4)
        self.assertGreater(train[0][1].sum().item(), 0)
        for a, b in zip(train[0], train[0]):
            np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_report_uses_the_same_validation_blocks(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        target = np.random.default_rng(11).random((13, 17)) > 0.7
        scene = SimpleNamespace(
            tiles=[dict(target=target, channels=np.stack([target, target]).astype(np.float32))]
        )
        data = RealCrops(scene, size=8, training=False)
        model = SmallUNet(input_channels=2, base=4)
        measured = run_epoch(model, DataLoader(data, batch_size=4), "cpu", 0.5)
        prediction = block_probability(model, scene.tiles[0]["channels"], 8, 4) >= 0.5
        expected = metrics_from_counts(
            [
                int((prediction & target).sum()),
                int((prediction & ~target).sum()),
                int((~prediction & target).sum()),
                int((~prediction & ~target).sum()),
            ]
        )
        for key, value in expected.items():
            self.assertEqual(measured[key], value)

    def test_date_and_source_leakage_rejected(self):
        a = dict(source_sha256="a", source_name="20250101_20250113.diff.bmp")
        b = dict(source_sha256="b", source_name="20250113_20250125.diff.bmp")
        with self.assertRaisesRegex(ValueError, "share"):
            disjoint_dates(a, b)
        b["source_name"] = "20250301_20250313.diff.bmp"
        disjoint_dates(a, b)
        b["source_sha256"] = "a"
        with self.assertRaisesRegex(ValueError, "identical"):
            disjoint_dates(a, b)


if __name__ == "__main__":
    unittest.main()
