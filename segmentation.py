"""An independently implemented compact U-Net and masked segmentation utilities."""

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F


def select_device(name="auto"):
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; choose auto or cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; choose auto or cpu")
    return torch.device(name)


class ConvBlock(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(4, outgoing),
            nn.ReLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(4, outgoing),
            nn.ReLU(),
        )


class SmallUNet(nn.Module):
    """Three downsampling stages; outputs one logit per input pixel."""

    def __init__(self, input_channels=3, base=16):
        super().__init__()
        if input_channels not in (2, 3) or base < 4 or base % 4:
            raise ValueError("Use 2/3 input channels and a base width divisible by 4")
        self.enc1 = ConvBlock(input_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.bridge = ConvBlock(base * 4, base * 8)
        self.dec3 = ConvBlock(base * 12, base * 4)
        self.dec2 = ConvBlock(base * 6, base * 2)
        self.dec1 = ConvBlock(base * 3, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.enc3(F.max_pool2d(b, 2))
        x = self.bridge(F.max_pool2d(c, 2))
        for skip, block in [(c, self.dec3), (b, self.dec2), (a, self.dec1)]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)


def masked_loss(logits, target, valid):
    """BCE plus batch soft Dice, with invalid pixels excluded from both terms."""
    if logits.shape != target.shape or valid.shape != target.shape:
        raise ValueError("Loss tensors must have identical dimensions")
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid).sum()
    bce = bce / valid.sum().clamp_min(1)
    probability = logits.sigmoid() * valid
    truth = target * valid
    dice = 1 - (2 * (probability * truth).sum() + 1) / (probability.sum() + truth.sum() + 1)
    return bce + dice


def confusion_counts(logits, target, valid, threshold=0.5):
    prediction = logits.sigmoid() >= threshold
    truth, usable = target >= 0.5, valid >= 0.5
    return [
        int((prediction & truth & usable).sum().item()),
        int((prediction & ~truth & usable).sum().item()),
        int((~prediction & truth & usable).sum().item()),
        int((~prediction & ~truth & usable).sum().item()),
    ]


def metrics_from_counts(counts):
    tp, fp, fn, tn = counts

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        foreground_iou=ratio(tp, tp + fp + fn),
        f1=ratio(2 * tp, 2 * tp + fp + fn),
        precision=ratio(tp, tp + fp),
        recall=ratio(tp, tp + fn),
        valid_pixels=tp + fp + fn + tn,
    )


def tile_starts(length, size, stride):
    starts = list(range(0, max(length - size, 0) + 1, stride))
    if starts[-1] != max(length - size, 0):
        starts.append(max(length - size, 0))
    return starts


def downsample_channels(channels, factor, phase_mode=False):
    """Average at a larger display scale; use circular means for phase channels.

    This changes pixel context only, never establishes a geographic pixel spacing.
    Zero phase vectors remain missing; phase vectors with cancellation remain zero.
    """
    if type(factor) is not int or factor < 1:
        raise ValueError("downsample must be a positive integer")
    height, width = channels.shape[1:]
    if factor == 1:
        return channels
    shape = (max(1, (width + factor - 1) // factor), max(1, (height + factor - 1) // factor))
    reduced = np.stack(
        [
            np.asarray(Image.fromarray(c).resize(shape, Image.Resampling.BOX), dtype=np.float32)
            for c in channels
        ]
    )
    if phase_mode:
        if channels.shape[0] != 2:
            raise ValueError("Circular means require two phase channels")
        magnitude = np.hypot(reduced[0], reduced[1])
        np.divide(reduced, magnitude[None], out=reduced, where=magnitude[None] > 1e-6)
        reduced[:, magnitude <= 1e-6] = 0
    return reduced


@torch.inference_mode()
def tiled_probability(
    model, channels, device, tile_size=256, stride=192, batch_size=1, progress=None
):
    """Overlap-weighted prediction; retains original pixel size and covers edge tiles."""
    if channels.ndim != 3 or not np.isfinite(channels).all():
        raise ValueError("Input must be a finite C,H,W array")
    if tile_size < 8 or not 1 <= stride <= tile_size:
        raise ValueError("Require tile_size >= 8 and 1 <= stride <= tile_size")
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError("Inference batch size must be an integer in 1..128")
    _, height, width = channels.shape
    if height < 1 or width < 1:
        raise ValueError("Image is empty")
    result = np.zeros((height, width), np.float32)
    weights = np.zeros_like(result)
    taper = np.maximum(np.hanning(tile_size).astype(np.float32), 0.05)
    window = taper[:, None] * taper[None, :]
    model.eval()
    pending = []
    completed = 0

    def flush():
        nonlocal completed
        tensor = torch.from_numpy(np.stack([item[0] for item in pending])).to(device)
        probabilities = model(tensor).sigmoid()[:, 0].cpu().numpy()
        for probability, (_, y, x, h, w) in zip(probabilities, pending):
            result[y : y + h, x : x + w] += probability[:h, :w] * window[:h, :w]
            weights[y : y + h, x : x + w] += window[:h, :w]
        completed += len(pending)
        pending.clear()
        if progress is not None:
            progress(completed)

    for y in tile_starts(height, tile_size, stride):
        for x in tile_starts(width, tile_size, stride):
            crop = channels[:, y : y + tile_size, x : x + tile_size]
            h, w = crop.shape[-2:]
            crop = np.pad(crop, ((0, 0), (0, tile_size - h), (0, tile_size - w)), mode="edge")
            pending.append((np.ascontiguousarray(crop, dtype=np.float32), y, x, h, w))
            if len(pending) == batch_size:
                flush()
    if pending:
        flush()
    return result / weights


def multiscale_probability(
    model, channels, device, tile_size=256, stride=192, scales=(1,), progress=None, batch_size=1
):
    """Equal-weight mean after each scale's tiled prediction returns to source pixels.

    No thresholds, target labels or component-size filters enter this fusion.
    Display context scales are not geographic or physical-scale calibration.
    """
    if not scales or len(set(scales)) != len(scales):
        raise ValueError("Prediction scales must be nonempty and unique")
    if any(type(scale) is not int or scale < 1 for scale in scales):
        raise ValueError("Prediction scales must be positive integers")
    shape = channels.shape[-2:]
    fused = np.zeros(shape, dtype=np.float32)
    for scale in scales:
        reduced = downsample_channels(channels, scale, channels.shape[0] == 2)
        probability = tiled_probability(model, reduced, device, tile_size, stride, batch_size)
        if scale != 1:
            probability = np.asarray(
                Image.fromarray(probability).resize(shape[::-1], Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
        fused += probability / len(scales)
        if progress is not None:
            progress(scale)
    return fused


def mask_boundary(mask):
    """One-pixel inner boundary, including objects touching the image edge."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, constant_values=False)
    interior = mask & padded[:-2, 1:-1] & padded[2:, 1:-1]
    interior &= padded[1:-1, :-2] & padded[1:-1, 2:]
    return mask & ~interior


def segmentation_overlay(image, probability, threshold=0.5, contours_only=False):
    pixels = np.asarray(image.convert("RGB")).copy()
    mask = probability >= threshold
    if pixels.shape[:2] != mask.shape:
        raise ValueError("Image and prediction coordinates differ")
    if not contours_only:
        pixels[mask] = (pixels[mask] * 0.55 + np.array([44, 210, 163]) * 0.45).astype(np.uint8)
    pixels[mask_boundary(mask)] = [255, 214, 92]
    return Image.fromarray(pixels)


def prediction_preview(image, probability, target=None, valid=None, threshold=0.5):
    """Observed image, truth when available, and raw model prediction."""
    image = image.convert("RGB")
    panels = [("INPUT", image)]
    if target is not None:
        truth = np.repeat((target[..., None] > 0).astype(np.uint8) * 255, 3, axis=2)
        if valid is not None:
            truth[valid < 0.5] = [70, 70, 100]
        panels.append(("LABEL (GRAY: IGNORED)", Image.fromarray(truth)))
    overlay = np.asarray(image).copy()
    predicted = probability >= threshold
    overlay[predicted] = (0.5 * overlay[predicted] + np.array([30, 125, 45])).astype(np.uint8)
    panels.extend(
        [
            ("PREDICTION", Image.fromarray(overlay)),
            (
                "PROBABILITY",
                Image.fromarray(np.round(probability * 255).astype(np.uint8)).convert("RGB"),
            ),
        ]
    )
    side = 256
    board = Image.new("RGB", (side * len(panels), side + 34), "#101722")
    draw = ImageDraw.Draw(board)
    for i, (name, panel) in enumerate(panels):
        draw.text((i * side + 8, 10), name, fill="white")
        board.paste(panel.resize((side, side)), (i * side, 34))
    return board
