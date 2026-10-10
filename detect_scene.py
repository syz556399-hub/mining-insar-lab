"""Apply the detection heatmap model and export candidate analysis crops."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

from data_io import phase_rgb
from detection_first import FringeDetector
from detection_study import candidates, file_hash
from phase_encoding import decode_display_palette, quantized_phase
from segmentation import tiled_probability


class ScoreLogits(nn.Module):
    """Adapt a score network to the existing logit-based overlap inference API."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.logit(self.model(x).clamp(1e-7, 1 - 1e-7))


def detect(checkpoint, output, arrays=None, image=None):
    if (arrays is None) == (image is None):
        raise ValueError("Specify exactly one numeric scene or original indexed BMP")
    if output.exists():
        raise ValueError("Choose a new output directory")
    torch.set_num_threads(2)
    cp = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = cp.get("model_config", dict(width=cp.get("width", 8)))
    model = FringeDetector(**config)
    model.load_state_dict(cp["state_dict"])
    model.eval()
    if arrays is not None:
        with np.load(arrays, allow_pickle=False) as data:
            phase = quantized_phase(data["wrapped_phase_rad"], 16)
        visible = np.ones(phase.shape, bool)
        display = Image.fromarray(phase_rgb(phase))
        source = arrays
        adapter = dict(kind="numeric phase; 16-bin input")
    else:
        with Image.open(image) as original:
            phase, visible, adapter = decode_display_palette(original)
            display = original.convert("RGB")
        source = image
    x = np.stack([np.sin(phase), np.cos(phase)]).astype(np.float32)
    x[:, ~visible] = 0
    score = tiled_probability(
        ScoreLogits(model), x, torch.device("cpu"), tile_size=256, stride=192, batch_size=2
    )
    windows = candidates(score)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "score.npy", score)
    Image.fromarray(np.round(score * 255).astype(np.uint8)).save(output / "score.png")
    overlay = display.copy()
    draw = ImageDraw.Draw(overlay)
    for i, window in enumerate(windows):
        draw.rectangle(window["bbox"], outline="#fff080", width=2)
        display.crop(window["bbox"]).save(output / f"crop_{i:03d}.png")
    overlay.save(output / "windows.png")
    result = dict(
        source_name=source.name,
        source_sha256=file_hash(source),
        checkpoint_sha256=file_hash(checkpoint),
        image_size=list(display.size),
        windows=windows,
        input_adapter=adapter,
        window_rule="score >=0.25, area >=64 pixels, padding=24; fixed experimental settings",
        coordinates="original image pixels; left/top inclusive, right/bottom exclusive",
        scope="candidate windows only, not final labels, instance counts or calibrated probabilities",
        inference="native 256 pixel tiles, stride192, tapered mean; no image resizing",
    )
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--arrays", type=Path)
    source.add_argument("--image", type=Path)
    args = parser.parse_args()
    print(json.dumps(detect(args.checkpoint, args.out, args.arrays, args.image), indent=2))
