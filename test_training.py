import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from data_io import PhaseDataset, phase_rgb

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from segmentation import (
        SmallUNet,
        confusion_counts,
        downsample_channels,
        masked_loss,
        tiled_probability,
    )


@unittest.skipIf(torch is None, "Install requirements-training.txt for PyTorch checks")
class TrainingTests(unittest.TestCase):
    def test_batched_tiles_match_single_tiles_on_odd_edges(self):
        torch.manual_seed(7)
        model = SmallUNet(input_channels=2, base=4).eval()
        channels = np.random.default_rng(7).normal(size=(2, 49, 55)).astype(np.float32)
        single = tiled_probability(model, channels, torch.device("cpu"), 24, 17, 1)
        batched = tiled_probability(model, channels, torch.device("cpu"), 24, 17, 4)
        np.testing.assert_allclose(single, batched, atol=2e-6, rtol=2e-6)
        with self.assertRaises(ValueError):
            tiled_probability(model, channels, torch.device("cpu"), batch_size=0)

    def test_circular_downsampling_preserves_wrap_boundary_and_missing_vectors(self):
        angles = np.array(
            [[np.pi - 0.1, -np.pi + 0.1], [np.pi - 0.1, -np.pi + 0.1]], dtype=np.float32
        )
        channels = np.stack([np.sin(angles), np.cos(angles)])
        result = downsample_channels(channels, 2, phase_mode=True)
        np.testing.assert_allclose(result[:, 0, 0], [0, -1], atol=1e-6)
        missing = downsample_channels(np.zeros((2, 5, 7), dtype=np.float32), 2, phase_mode=True)
        self.assertEqual(missing.shape, (2, 3, 4))
        self.assertFalse(missing.any())

    def test_invalid_pixels_do_not_change_loss_metrics_or_gradients(self):
        logits = torch.tensor([[[[0.2, -0.5], [1.0, 2.0]]]], requires_grad=True)
        target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        valid = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        loss = masked_loss(logits, target, valid)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[valid == 0], torch.zeros(2)))
        changed_logits = logits.detach().clone()
        changed_logits[valid == 0] = 100
        changed_target = target.clone()
        changed_target[valid == 0] = 1
        torch.testing.assert_close(
            loss.detach(), masked_loss(changed_logits, changed_target, valid)
        )
        self.assertEqual(
            confusion_counts(logits, target, valid),
            confusion_counts(changed_logits, changed_target, valid),
        )

    def test_all_invalid_has_zero_loss_and_zero_gradient(self):
        logits = torch.randn(1, 1, 4, 4, requires_grad=True)
        loss = masked_loss(logits, torch.ones_like(logits), torch.zeros_like(logits))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_loader_uses_observed_image_not_ground_truth_as_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase = np.zeros((16, 16), dtype=np.float32)
            image = phase_rgb(phase)
            Image.fromarray(image).save(root / "observed.png")
            np.savez(
                root / "a.npz",
                wrapped_phase_rad=phase,
                mask=np.zeros_like(phase),
                valid_mask=np.ones_like(phase),
            )
            with (root / "manifest.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["split", "image", "arrays"])
                writer.writeheader()
                writer.writerow(dict(split="train", image="observed.png", arrays="a.npz"))
            dataset = PhaseDataset(root, return_valid=True, input_mode="rgb", require_valid=True)
            before, target, _ = dataset[0]
            np.savez(
                root / "a.npz",
                wrapped_phase_rad=phase,
                mask=np.ones_like(phase),
                valid_mask=np.ones_like(phase),
            )
            after, changed, _ = dataset[0]
            torch.testing.assert_close(before, after)
            self.assertEqual(target.sum().item(), 0)
            self.assertEqual(changed.sum().item(), 256)
            torch.testing.assert_close(
                before, torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1) / 255)
            )
            phase_input, _ = PhaseDataset(root)[0]
            self.assertEqual(phase_input.shape, (2, 16, 16))

    def test_tiled_prediction_covers_odd_edges_and_small_images(self):
        class Constant(torch.nn.Module):
            def forward(self, x):
                return x[:, :1] * 0 + 0.7

        for shape in [(3, 45, 53), (3, 5, 7)]:
            probability = tiled_probability(
                Constant(),
                np.ones(shape, dtype=np.float32),
                torch.device("cpu"),
                tile_size=24,
                stride=17,
            )
            self.assertEqual(probability.shape, shape[1:])
            np.testing.assert_allclose(probability, torch.tensor(0.7).sigmoid().item(), atol=1e-6)

    def test_checkpoint_round_trip_and_odd_unet_output_size(self):
        torch.manual_seed(5)
        model = SmallUNet(input_channels=3, base=4).eval()
        x = torch.rand(1, 3, 27, 35)
        with torch.no_grad():
            before = model(x)
        self.assertEqual(before.shape, (1, 1, 27, 35))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            torch.save(dict(state_dict=model.state_dict()), path)
            restored = SmallUNet(input_channels=3, base=4).eval()
            restored.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
            with torch.no_grad():
                torch.testing.assert_close(before, restored(x), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
