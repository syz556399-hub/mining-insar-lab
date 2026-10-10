import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from provenance import fingerprint

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from detection_first import FringeDetector
    from detection_study import (
        DILATIONS,
        augment,
        candidates,
        file_hash,
        load_dataset,
        regression_loss,
    )


@unittest.skipIf(torch is None, "Install requirements-training.txt for detection checks")
class DetectionStudyTests(unittest.TestCase):
    def test_tiled_score_adapter_does_not_apply_sigmoid_twice(self):
        from detect_scene import ScoreLogits
        from segmentation import tiled_probability

        class FixedScore(torch.nn.Module):
            def forward(self, x):
                return torch.full((len(x), 1, *x.shape[-2:]), 0.2)

        result = tiled_probability(
            ScoreLogits(FixedScore()),
            np.zeros((2, 21, 29), np.float32),
            torch.device("cpu"),
            tile_size=16,
            stride=12,
            batch_size=2,
        )
        np.testing.assert_allclose(result, 0.2, atol=1e-6)

    def test_augmentation_deterministic_and_preserves_target_values(self):
        phase = np.arange(64, dtype=np.float32).reshape(8, 8) / 20
        target = np.zeros((8, 8), np.float32)
        target[2:4, 5:7] = 1
        a, y = augment(phase, target, np.random.default_rng(91))
        b, z = augment(phase, target, np.random.default_rng(91))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(y, z)
        self.assertEqual(float(y.sum()), 4)
        self.assertEqual(float(target.sum()), 4)
        np.testing.assert_allclose(np.linalg.norm(a, axis=0), 1, atol=1e-6)

    def test_candidate_padding_bounds_and_background(self):
        score = np.zeros((30, 40), np.float32)
        self.assertEqual(candidates(score), [])
        score[0:8, 0:8] = 0.8
        score[29, 39] = 0.9
        result = candidates(score)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["bbox"], [0, 0, 32, 30])
        self.assertEqual(result[0]["support_pixels"], 64)

    def test_extended_model_and_target_gradient_loss(self):
        torch.set_num_threads(2)
        model = FringeDetector(width=12, dilations=DILATIONS)
        x = torch.randn(1, 2, 35, 41)
        target = torch.rand(1, 1, 35, 41)
        prediction = model(x)
        self.assertEqual(prediction.shape, target.shape)
        self.assertEqual(float(regression_loss(target, target)), 0)
        loss = regression_loss(prediction, target)
        loss.backward()
        self.assertTrue(
            all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        )

    def test_dataset_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for split in ("train", "val", "test"):
                path = root / f"{split}.npz"
                np.savez(
                    path,
                    wrapped_phase_rad=np.zeros((8, 8), np.float32),
                    detection_target=np.zeros((8, 8), np.float32),
                )
                rows.append(dict(name=split, split=split, group_id=split, sha256=file_hash(path)))
            manifest = dict(samples=rows)
            manifest["fingerprint"] = fingerprint(manifest)
            (root / "manifest.json").write_text(json.dumps(manifest))
            _, partitions = load_dataset(root)
            self.assertEqual([len(p) for p in partitions.values()], [1, 1, 1])
            (root / "test.npz").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_dataset(root)


if __name__ == "__main__":
    unittest.main()
