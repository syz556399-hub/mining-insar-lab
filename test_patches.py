import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from data_io import PhaseDataset
    from patch_dataset import ScenePatchDataset, ScenePatchSampler
    from segmentation import SmallUNet, mask_boundary, multiscale_probability
    from train import evaluate_scenes, run_epoch


@unittest.skipIf(torch is None, "Install requirements-training.txt for patch checks")
class PatchTests(unittest.TestCase):
    def test_first_update_trace_does_not_change_training_updates(self):
        torch.manual_seed(16)
        first, second = SmallUNet(2, 4), SmallUNet(2, 4)
        second.load_state_dict(first.state_dict())
        batch = (torch.rand(2, 2, 16, 16), torch.ones(2, 1, 16, 16), torch.ones(2, 1, 16, 16))
        first_optimizer = torch.optim.Adam(first.parameters(), lr=0.001)
        second_optimizer = torch.optim.Adam(second.parameters(), lr=0.001)
        trace = {}
        run_epoch(first, [batch, batch], torch.device("cpu"), 0.5, first_optimizer, trace)
        run_epoch(second, [batch, batch], torch.device("cpu"), 0.5, second_optimizer)
        for before, after in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertEqual(trace["input_shape"], [2, 2, 16, 16])
        self.assertEqual(trace["output_shape"], [2, 1, 16, 16])
        self.assertGreater(trace["gradient_l2"], 0)
        self.assertGreater(trace["parameter_change_l2"], 0)

    def test_scene_sampler_equal_budget_and_epoch_reproducibility(self):
        class Patches:
            augment = True
            rows = [dict(scene_id=scene, split="train") for scene in ("a",) * 9 + ("b",)]

            def __getitem__(self, index):
                return torch.zeros(2, 8, 8), torch.zeros(1, 8, 8), torch.ones(1, 8, 8)

        patches = Patches()
        sampler = ScenePatchSampler(patches, 30, seed=7)
        self.assertTrue(patches.augment)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        self.assertEqual(len(first), 60)
        self.assertEqual(sum(patches.rows[i]["scene_id"] == "a" for i in first), 30)
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))
        sampler.set_epoch(0)
        self.assertEqual(first, list(sampler))

    def test_balanced_sampler_uses_training_targets_and_excludes_invalid(self):
        class Patches:
            augment = False
            rows = [dict(scene_id="a", split="train") for _ in range(10)]

            def __getitem__(self, index):
                target, valid = torch.zeros(1, 8, 8), torch.ones(1, 8, 8)
                if index in (0, 1):
                    target[:, 0, 0] = 1
                if index == 1:
                    valid[:, 0, 0] = 0  # Ignored positive does not create a positive patch.
                if index == 9:
                    valid[:] = 0
                return torch.zeros(2, 8, 8), target, valid

        sampler = ScenePatchSampler(Patches(), 1000, balanced=True, seed=2)
        draws = list(sampler)
        self.assertEqual(sampler.groups["a"]["positive"], [0])
        self.assertNotIn(9, draws)
        self.assertTrue(450 < draws.count(0) < 550)
        self.assertEqual(sampler.summary()["draws_per_epoch"], 1000)

    def test_scene_sampler_rejects_held_out_splits(self):
        class Patches:
            rows = [dict(scene_id="a", split="val")]

        with self.assertRaisesRegex(ValueError, "training split"):
            ScenePatchSampler(Patches())

    def test_crop_coordinates_alignment_and_split_inheritance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            y, x = np.indices((25, 25))
            phase = (x + 2 * y).astype(np.float32) / 10
            target = ((x >= 8) & (y >= 7)).astype(np.uint8)
            valid = (x < 21).astype(np.uint8)
            np.savez(root / "scene.npz", wrapped_phase_rad=phase, mask=target, valid_mask=valid)
            with (root / "manifest.csv").open("w", newline="") as stream:
                fields = ["scene_id", "group_id", "split", "category", "arrays"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    dict(
                        scene_id="scene",
                        group_id="same-geometry",
                        split="test",
                        category="single",
                        arrays="scene.npz",
                    )
                )
            scenes = PhaseDataset(root, "test", return_valid=True, require_valid=True)
            patches = ScenePatchDataset(scenes, 25, 8, 6, (1, 2))
            coverage = np.zeros((25, 25), dtype=bool)
            for i, row in enumerate(patches.rows):
                self.assertEqual(row["split"], "test")
                self.assertEqual(row["group_id"], "same-geometry")
                channels, actual_target, actual_valid = patches[i]
                self.assertEqual(channels.shape, (2, 8, 8))
                if row["context_scale"] == 1:
                    a, b, c, d = (row[k] for k in ("top", "bottom", "left", "right"))
                    np.testing.assert_array_equal(actual_target[0].numpy(), target[a:b, c:d])
                    np.testing.assert_array_equal(actual_valid[0].numpy(), valid[a:b, c:d])
                    np.testing.assert_allclose(channels[0], np.sin(phase[a:b, c:d]), atol=1e-6)
                    coverage[a:b, c:d] = True
            self.assertTrue(coverage.all())
            self.assertEqual(len(PhaseDataset(root, "train")), 0)

    def test_multiscale_fusion_covers_odd_edges_and_preserves_constant_score(self):
        class Constant(torch.nn.Module):
            def forward(self, x):
                return x[:, :1] * 0 + 0.7

        channels = np.ones((2, 45, 53), dtype=np.float32)
        result = multiscale_probability(
            Constant(), channels, torch.device("cpu"), 16, 11, (1, 2, 4)
        )
        self.assertEqual(result.shape, (45, 53))
        np.testing.assert_allclose(result, torch.tensor(0.7).sigmoid().item(), atol=1e-6)
        with self.assertRaises(ValueError):
            multiscale_probability(Constant(), channels, torch.device("cpu"), scales=(1, 1))

    def test_whole_scene_evaluation_counts_overlaps_once_and_excludes_invalid(self):
        class Constant(torch.nn.Module):
            def forward(self, x):
                return x[:, :1] * 0 + 1

        channels = torch.ones(2, 25, 29)
        target = torch.ones(1, 25, 29)
        valid = torch.ones(1, 25, 29)
        valid[:, :2] = 0
        result = evaluate_scenes(
            Constant(), [(channels, target, valid)], torch.device("cpu"), 0.5, 16, (1, 2)
        )
        self.assertEqual(result["valid_pixels"], 23 * 29)
        self.assertEqual(result["tp"], 23 * 29)
        self.assertEqual(result["foreground_iou"], 1)

    def test_boundary_keeps_image_edge_and_holes(self):
        mask = np.ones((5, 5), dtype=bool)
        mask[2, 2] = False
        boundary = mask_boundary(mask)
        self.assertTrue(boundary[0].all())
        self.assertTrue(boundary[-1].all())
        self.assertFalse(boundary[2, 2])
        self.assertTrue(boundary[1, 2])


if __name__ == "__main__":
    unittest.main()
