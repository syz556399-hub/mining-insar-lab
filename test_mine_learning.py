import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from task_labels import fringe_support

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from annotation_audit import sha256
    from mine_learning import (
        PHASE_SCALE,
        RangePhaseNet,
        augment,
        learning_loss,
        load_data,
        request_for,
    )
    from provenance import fingerprint


@unittest.skipIf(torch is None, "Install requirements-training.txt for mining learning checks")
class MineLearningTests(unittest.TestCase):
    def test_split_reuse_and_modified_arrays_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for split in ("train", "val", "test"):
                path = root / (split + ".npz")
                clean = np.zeros((8, 8), np.float32)
                np.savez(
                    path,
                    wrapped_phase_rad=clean,
                    deformation_phase_rad=clean,
                    fringe_mask=fringe_support(clean),
                )
                metadata = root / (split + ".json")
                metadata.write_text("{}")
                rows.append(
                    dict(
                        name=split,
                        split=split,
                        group_id=split,
                        sha256=sha256(path),
                        metadata_sha256=sha256(metadata),
                    )
                )
            manifest = dict(samples=rows, splits=dict(train=1, val=1, test=1))

            def write():
                payload = dict(manifest, fingerprint=fingerprint(manifest))
                (root / "manifest.json").write_text(json.dumps(payload))

            write()
            _, data = load_data(root)
            self.assertEqual([len(part) for part in data.values()], [1, 1, 1])
            rows[2]["group_id"] = "train"
            write()
            with self.assertRaisesRegex(ValueError, "Repeated physical scene"):
                load_data(root)
            rows[2]["group_id"] = "test"
            write()
            (root / "test.npz").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_data(root)

    def test_signed_auxiliary_stays_aligned_with_range_after_augmentation(self):
        clean = np.zeros((16, 16), np.float32)
        clean[2:8, 7:13] = -8 * np.pi
        target = fringe_support(clean).astype(np.float32)
        observed = np.angle(np.exp(1j * clean)).astype(np.float32)
        for seed in range(8):
            x, y, cycles = augment(observed, target, clean, np.random.default_rng(seed))
            np.testing.assert_array_equal(y[0], fringe_support(cycles[0] * PHASE_SCALE))
            self.assertAlmostEqual(float(np.abs(cycles).max()), 4.0, places=5)
            self.assertEqual(int(y.sum()), int(target.sum()))
            np.testing.assert_allclose(np.linalg.norm(x, axis=0), 1, atol=1e-6)

    def test_noise_cannot_create_ground_truth_in_negative_design(self):
        request, design = request_for(71, 3)
        self.assertEqual(design["category"], "negative")
        self.assertFalse(any(f["enabled"] for f in request["faces"]))
        self.assertNotEqual(request["settings"]["orbit_cycles"], 0)
        clean = np.zeros((12, 12), np.float32)
        observed = np.random.default_rng(7).uniform(-np.pi, np.pi, clean.shape).astype(np.float32)
        _, target, cycles = augment(
            observed, fringe_support(clean), clean, np.random.default_rng(7)
        )
        self.assertEqual(float(target.sum()), 0)
        self.assertEqual(float(np.abs(cycles).sum()), 0)

    def test_auxiliary_weight_changes_loss_and_reaches_phase_parameters(self):
        torch.set_num_threads(2)
        torch.manual_seed(1)
        model = RangePhaseNet()
        logits, phase = model(torch.randn(2, 2, 19, 23))
        target = torch.zeros_like(logits)
        target[:, :, 4:12, 8:16] = 1
        truth = -3 * target
        mask_only = learning_loss(logits, phase, target, truth, 0)
        joint = learning_loss(logits, phase, target, truth, 0.2)
        self.assertGreater(float(joint.detach()), float(mask_only.detach()))
        joint.backward()
        self.assertTrue(torch.isfinite(model.phase_head.weight.grad).all())
        self.assertGreater(float(model.phase_head.weight.grad.abs().sum()), 0)
        self.assertEqual(logits.shape, target.shape)


if __name__ == "__main__":
    unittest.main()
