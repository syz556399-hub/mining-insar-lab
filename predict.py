"""Predict a saved simulator NPZ or an RGB/BMP image using a trained checkpoint."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data_io import phase_rgb
from phase_encoding import decode_display_palette, quantized_phase
from segmentation import (
    SmallUNet,
    multiscale_probability,
    prediction_preview,
    segmentation_overlay,
    select_device,
)
from segmentation_report import write_report


def predict(args):
    if args.out.exists():
        raise ValueError("Output exists; use a new prediction directory")
    if args.palette_code and args.arrays:
        raise ValueError("--palette-code applies only to an original indexed --image")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_format") != 1:
        raise ValueError("Unsupported checkpoint format")
    mode = checkpoint["input_mode"]
    target = valid = None
    display_info = None
    display_valid = None
    if args.arrays:
        with np.load(args.arrays, allow_pickle=False) as data:
            phase = quantized_phase(data["wrapped_phase_rad"], checkpoint.get("phase_bins", 0))
            image = Image.fromarray(phase_rgb(phase))
            target_key = (
                "fringe_mask"
                if checkpoint["configuration"].get("target_mode") == "fringe"
                else "mask"
            )
            valid_key = "fringe_valid_mask" if target_key == "fringe_mask" else "valid_mask"
            if target_key in data:
                target = data[target_key].copy()
            if valid_key in data:
                valid = data[valid_key].copy()
        channels = (
            np.stack([np.sin(phase), np.cos(phase)]).astype(np.float32)
            if mode == "phase"
            else np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255
        )
    else:
        if args.palette_code:
            with Image.open(args.image) as source:
                phase, display_valid, display_info = decode_display_palette(source)
                image = source.convert("RGB")
            channels = (
                np.stack([np.sin(phase), np.cos(phase)]).astype(np.float32)
                if mode == "phase"
                else np.asarray(Image.fromarray(phase_rgb(phase)), dtype=np.float32).transpose(
                    2, 0, 1
                )
                / 255
            )
            channels[:, ~display_valid] = 0
        elif mode != "rgb":
            raise ValueError(
                "A phase model requires numeric --arrays or an explicitly verified --palette-code image"
            )
        else:
            with Image.open(args.image) as source:
                image = source.convert("RGB")
            channels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255
    threshold = checkpoint["threshold"] if args.threshold is None else args.threshold
    if not 0 < threshold < 1:
        raise ValueError("Require 0 < threshold < 1")
    tile = checkpoint["tile_size"]
    stride = max(1, int(tile * 0.75)) if args.stride is None else args.stride
    model = SmallUNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    original_shape = channels.shape[1:]
    scales = getattr(args, "scales", None)
    if scales and args.downsample != 1:
        raise ValueError("Choose --scales or --downsample, not both")
    scales = tuple(scales or [args.downsample])
    probability = multiscale_probability(
        model,
        channels,
        device,
        tile,
        stride,
        scales,
        progress=lambda scale: print(json.dumps(dict(completed_scale=scale)), flush=True),
        batch_size=getattr(args, "inference_batch_size", 1),
    )
    suppress_display_zero = checkpoint["configuration"].get("display_zero_policy") != "predict"
    if display_valid is not None and suppress_display_zero:
        probability[~display_valid] = 0
    args.out.mkdir(parents=True, exist_ok=False)
    np.save(args.out / "probability.npy", probability, allow_pickle=False)
    Image.fromarray((probability >= threshold).astype(np.uint8) * 255).save(args.out / "mask.png")
    Image.fromarray(np.round(probability * 255).astype(np.uint8)).save(args.out / "probability.png")
    prediction_preview(image, probability, target, valid, threshold).save(args.out / "preview.png")
    segmentation_overlay(image, probability, threshold).save(args.out / "overlay.png")
    segmentation_overlay(image, probability, threshold, contours_only=True).save(
        args.out / "boundaries.png"
    )
    info = dict(
        source_path=str((args.arrays or args.image).resolve()),
        source_name=(args.arrays or args.image).name,
        input_kind="numeric_phase"
        if args.arrays
        else ("display_proxy" if args.palette_code else "rgb"),
        source_sha256=hashlib.sha256((args.arrays or args.image).read_bytes()).hexdigest(),
        checkpoint_path=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        input_mode=mode,
        phase_bins=checkpoint.get("phase_bins", 0),
        display_adapter=display_info,
        device=str(device),
        cpu_threads=args.threads,
        inference_batch_size=getattr(args, "inference_batch_size", 1),
        shape=list(original_shape),
        threshold=threshold,
        tile_size=tile,
        stride=stride,
        downsample=args.downsample,
        scales=list(scales),
        scale_fusion="equal-weight mean of source-aligned per-scale scores; no size filtering",
        predicted_area_percent=float(np.mean(probability >= threshold) * 100),
        scale_interpretation="Display context only; not a geographic pixel-scale calibration",
        checkpoint_epoch=checkpoint["epoch"],
        checkpoint_dataset=checkpoint["configuration"]["dataset"],
        interpretation="Raw model prediction. Training on synthetic data does not establish real-image accuracy.",
        validity=(
            "Zero-brightness display pixels are suppressed; no measured coherence/water validity is inferred."
            if display_valid is not None and suppress_display_zero
            else (
                "All numeric scene pixels are predicted; exported valid_mask is reserved for evaluation."
                if args.arrays
                else "All image pixels are predicted; no true coherence/water validity is inferred."
            )
        ),
    )
    (args.out / "prediction.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    lesson = args.checkpoint.parent / "learning.html"
    if lesson.is_file():
        shutil.copyfile(lesson, args.out / "learning.html")
    write_report(args.out, image, probability, info, target=target, valid=valid)
    print(json.dumps(info), flush=True)
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--arrays", type=Path)
    source.add_argument("--image", type=Path)
    parser.add_argument(
        "--palette-code",
        action="store_true",
        help="Explicitly use verified 16×16 cyclic palette display-code proxy",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--downsample", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        choices=[1, 2, 4, 8],
        help="Fuse several display context scales, e.g. 1 2 4 8",
    )
    args = parser.parse_args()
    try:
        predict(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
