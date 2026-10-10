"""Train a compact segmenter exclusively on exported simulator scenes."""

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler

from data_io import PhaseDataset
from learning_report import write_learning_report
from patch_dataset import ScenePatchDataset, ScenePatchSampler
from provenance import runtime_versions
from segmentation import (
    SmallUNet,
    confusion_counts,
    masked_loss,
    metrics_from_counts,
    multiscale_probability,
    prediction_preview,
    select_device,
)
from validate_dataset import audit_splits, rows_at


def run_epoch(model, loader, device, threshold, optimizer=None, step_trace=None):
    training = optimizer is not None
    model.train(training)
    counts = np.zeros(4, dtype=np.int64)
    total_loss, total_valid = 0.0, 0
    with torch.set_grad_enabled(training):
        for inputs, target, valid in loader:
            inputs, target, valid = (value.to(device) for value in (inputs, target, valid))
            usable = int(valid.sum().item())
            if not usable:
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = masked_loss(logits, target, valid)
            if not torch.isfinite(loss).item():
                raise ValueError("Non-finite training loss")
            if training:
                tracing = step_trace is not None and not step_trace
                before = [p.detach().clone() for p in model.parameters()] if tracing else None
                loss.backward()
                if tracing:
                    gradient_norm = (
                        sum(
                            p.grad.detach().square().sum().item()
                            for p in model.parameters()
                            if p.grad is not None
                        )
                        ** 0.5
                    )
                optimizer.step()
                if tracing:
                    step_trace.update(
                        input_shape=list(inputs.shape),
                        target_shape=list(target.shape),
                        output_shape=list(logits.shape),
                        valid_pixels=usable,
                        positive_valid_pixels=int(((target >= 0.5) & (valid >= 0.5)).sum().item()),
                        loss_before_update=float(loss.item()),
                        gradient_l2=gradient_norm,
                        parameter_change_l2=sum(
                            (p.detach() - old).square().sum().item()
                            for p, old in zip(model.parameters(), before)
                        )
                        ** 0.5,
                        optimizer=type(optimizer).__name__,
                        learning_rate=optimizer.param_groups[0]["lr"],
                        scope="first valid training batch only; no extra update or held-out input",
                    )
            total_loss += loss.item() * usable
            total_valid += usable
            counts += confusion_counts(logits.detach(), target, valid, threshold)
    if not total_valid:
        raise ValueError("This split has no usable pixels")
    return dict(loss=total_loss / total_valid, **metrics_from_counts([int(n) for n in counts]))


def inspect_export(root):
    info = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    rows = rows_at(root)
    if info.get("cancelled") or info["count"] != len(rows):
        raise ValueError("Use a complete export, not a cancelled dataset")
    audit_splits([root])
    dimensions = set()
    thresholds = set()
    for row in rows:
        for key in ["image", "mask", "arrays", "metadata"]:
            if not (root / row[key]).is_file():
                raise ValueError(f"Missing exported file: {row[key]}")
        metadata = json.loads((root / row["metadata"]).read_text(encoding="utf-8"))
        dimensions.add(metadata["request"]["settings"]["size"])
        thresholds.add(metadata["request"]["settings"]["target_threshold_mm"])
    if len(dimensions) != 1:
        raise ValueError("This baseline requires equal-sized scenes within a dataset")
    return dict(
        engine=info["engine"],
        simulator_versions=info["software_versions"],
        label_rule=info["label_rule"],
        size=dimensions.pop(),
        manifest_sha256=hashlib.sha256((root / "manifest.csv").read_bytes()).hexdigest(),
        dataset_json_sha256=hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest(),
        sampling_profile=info.get("sampling_profile", "standard"),
        target_threshold_mm=thresholds.pop() if len(thresholds) == 1 else None,
    )


def evaluate_scenes(model, dataset, device, threshold, tile_size, scales):
    """Each original scene pixel is counted once after overlapping windows fuse."""
    counts = np.zeros(4, dtype=np.int64)
    for index in range(len(dataset)):
        channels, target, valid = dataset[index]
        probability = multiscale_probability(
            model, channels.numpy(), device, tile_size, max(1, tile_size * 3 // 4), scales
        )
        prediction, truth, usable = (
            probability >= threshold,
            target[0].numpy() >= 0.5,
            valid[0].numpy() >= 0.5,
        )
        counts += [
            np.sum(prediction & truth & usable),
            np.sum(prediction & ~truth & usable),
            np.sum(~prediction & truth & usable),
            np.sum(~prediction & ~truth & usable),
        ]
    result = metrics_from_counts([int(n) for n in counts])
    if not result["valid_pixels"]:
        raise ValueError("This scene split has no usable pixels")
    return dict(loss=None, **result)


def save_test_preview(model, dataset, root, output, device, threshold, tile_size=None, scales=(1,)):
    # Deterministic selection of up to one held-out scene per design category.
    selected, seen = [], set()
    model.eval()
    with torch.inference_mode():
        for index, row in enumerate(dataset.rows):
            if row["category"] in seen:
                continue
            seen.add(row["category"])
            inputs, target, valid = dataset[index]
            probability = (
                multiscale_probability(
                    model, inputs.numpy(), device, tile_size, max(1, tile_size * 3 // 4), scales
                )
                if tile_size is not None
                else model(inputs[None].to(device)).sigmoid()[0, 0].cpu().numpy()
            )
            with Image.open(root / row["image"]) as image:
                panel = prediction_preview(
                    image, probability, target[0].numpy(), valid[0].numpy(), threshold
                )
            selected.append(panel)
    board = Image.new("RGB", (selected[0].width, sum(p.height for p in selected)), "#101722")
    offset = 0
    for panel in selected:
        board.paste(panel, (0, offset))
        offset += panel.height
    board.save(output / "test_predictions.png")


def balanced_scene_sampler(data, positive_fraction, seed):
    """Training-only sampling; inspection must not consume augmentation RNG."""
    if not 0 < positive_fraction < 1 or any(r["split"] != "train" for r in data.rows):
        raise ValueError("Balancing requires training rows and probability in (0,1)")
    flags = []
    for row in data.rows:
        with np.load(data.root / row["arrays"], allow_pickle=False) as arrays:
            key = "fringe_mask" if data.target_mode == "fringe" else "mask"
            valid_key = "fringe_valid_mask" if data.target_mode == "fringe" else "valid_mask"
            flags.append(bool(np.any((arrays[key] > 0) & (arrays[valid_key] > 0))))
    positives = sum(flags)
    negatives = len(flags) - positives
    if not positives or not negatives:
        raise ValueError("Scene balancing requires both positive and negative training scenes")
    weights = [
        positive_fraction / positives if flag else (1 - positive_fraction) / negatives
        for flag in flags
    ]
    sampler = WeightedRandomSampler(
        weights,
        len(flags),
        replacement=True,
        generator=torch.Generator().manual_seed(seed + 1000),
    )
    scene_sampling = dict(
        positive_draw_probability=positive_fraction,
        positive_training_scenes=positives,
        negative_training_scenes=negatives,
        draws_per_epoch=len(flags),
        replacement=True,
        scope="training labels only; validation and test unchanged",
    )
    return sampler, scene_sampling


def train(args):
    if args.out.exists():
        raise ValueError("Output exists; use a new run directory")
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1:
        raise ValueError("epochs, batch-size and threads must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0 or not 0 < args.threshold < 1:
        raise ValueError("Require lr > 0 and 0 < threshold < 1")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    provenance = inspect_export(args.data)
    patch_size = getattr(args, "patch_size", 0)
    patch_stride = getattr(args, "patch_stride", 0) or max(1, patch_size * 3 // 4)
    context_scales = tuple(getattr(args, "context_scales", [1]))
    datasets = {
        split: PhaseDataset(
            args.data,
            split,
            True,
            args.input,
            require_valid=True,
            phase_bins=args.phase_bins,
            target_mode=getattr(args, "target", "physical"),
            augment_phase=args.phase_augment and split == "train" and not patch_size,
        )
        for split in ["train", "val", "test"]
    }
    if any(len(data) == 0 for data in datasets.values()):
        raise ValueError("Require nonempty train, val and test splits")
    scenes = datasets.copy()
    if patch_size:
        datasets = {
            split: ScenePatchDataset(
                data,
                provenance["size"],
                patch_size,
                patch_stride,
                context_scales,
                augment=split == "train" and args.phase_augment,
            )
            for split, data in scenes.items()
        }
    elif context_scales != (1,):
        raise ValueError("Context scales require --patch-size")
    sampling = getattr(args, "patch_sampling", "all")
    sampler = None
    if sampling != "all":
        if not patch_size or sampling not in ("uniform", "balanced"):
            raise ValueError("Scene sampling requires patch training and uniform/balanced mode")
        sampler = ScenePatchSampler(
            datasets["train"],
            getattr(args, "patch_draws", 3),
            balanced=sampling == "balanced",
            seed=args.seed,
        )
    positive_fraction = getattr(args, "scene_positive_fraction", 0.0)
    scene_sampling = None
    if positive_fraction:
        if patch_size or sampler is not None or not 0 < positive_fraction < 1:
            raise ValueError("Scene balancing requires whole scenes and a fraction in (0,1)")
        sampler, scene_sampling = balanced_scene_sampler(
            datasets["train"], positive_fraction, args.seed
        )
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=split == "train" and sampler is None,
            sampler=sampler if split == "train" else None,
            num_workers=0,
            generator=generator if split == "train" else None,
        )
        for split, data in datasets.items()
    }
    model_config = dict(input_channels=3 if args.input == "rgb" else 2, base=args.base)
    model = SmallUNet(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=False)
    configuration = dict(
        input_mode=args.input,
        phase_bins=args.phase_bins,
        phase_augmentation=args.phase_augment,
        normalization="RGB / 255" if args.input == "rgb" else "sin/cos phase",
        model="SmallUNet",
        model_config=model_config,
        parameters=sum(p.numel() for p in model.parameters()),
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=str(device),
        cpu_threads=args.threads,
        threshold=args.threshold,
        selection="maximum validation foreground IoU; earliest epoch on ties",
        supervision=(
            "experimental clean-phase support; all pixels supervised"
            if getattr(args, "target", "physical") == "fringe"
            else "vertical-increment binary mask; loss and metrics exclude valid_mask=0"
        ),
        target_mode=getattr(args, "target", "physical"),
        runtime={**runtime_versions(), "torch": str(torch.__version__)},
        dataset=provenance,
        split_counts={s: len(d) for s, d in datasets.items()},
        split_scene_counts={s: len(d) for s, d in scenes.items()},
        patch_sampling=sampler.summary()
        if isinstance(sampler, ScenePatchSampler)
        else dict(mode="all"),
        scene_sampling=scene_sampling,
        patch_training=(
            dict(
                size=patch_size,
                stride=patch_stride,
                context_scales=list(context_scales),
                split_rule="all patches inherit original scene split",
                evaluation="whole scene; each valid source pixel counted once",
                augmentation="aligned spatial flips and optional phase origin/sign"
                if args.phase_augment
                else "none",
            )
            if patch_size
            else None
        ),
        limitations="Synthetic evaluation only; fixed seeds do not guarantee bitwise device reproducibility",
    )
    (args.out / "config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    if patch_size:
        for split, data in datasets.items():
            data.save_manifest(args.out / f"patches_{split}.csv")
        datasets["train"].save_examples(args.out)
    print(
        json.dumps(
            dict(
                device=str(device),
                input=args.input,
                samples=configuration["split_counts"],
                parameters=configuration["parameters"],
            )
        ),
        flush=True,
    )
    started = time.monotonic()
    history = []
    first_step = {}
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        if isinstance(sampler, ScenePatchSampler):
            sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model,
            loaders["train"],
            device,
            args.threshold,
            optimizer,
            first_step if epoch == 1 else None,
        )
        if epoch == 1:
            (args.out / "first_step.json").write_text(
                json.dumps(first_step, indent=2), encoding="utf-8"
            )
        val_metrics = (
            evaluate_scenes(
                model, scenes["val"], device, args.threshold, patch_size, context_scales
            )
            if patch_size
            else run_epoch(model, loaders["val"], device, args.threshold)
        )
        if val_metrics["foreground_iou"] is None:
            raise ValueError("Validation requires a foreground target or prediction")
        record = dict(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            val_loss=val_metrics["loss"],
            val_foreground_iou=val_metrics["foreground_iou"],
            val_f1=val_metrics["f1"],
        )
        history.append(record)
        with (args.out / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(history)
        if val_metrics["foreground_iou"] > best_score:
            best_score = val_metrics["foreground_iou"]
            best_epoch = epoch
            best_metrics = val_metrics
            torch.save(
                dict(
                    checkpoint_format=1,
                    model_config=model_config,
                    state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    input_mode=args.input,
                    phase_bins=args.phase_bins,
                    tile_size=patch_size or provenance["size"],
                    context_scales=list(context_scales),
                    threshold=args.threshold,
                    epoch=epoch,
                    val_metrics=val_metrics,
                    configuration=configuration,
                ),
                args.out / "best.pt",
            )
        print(json.dumps(record), flush=True)
    # Test scenes are evaluated only after all model selection is complete.
    checkpoint = torch.load(args.out / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = (
        evaluate_scenes(model, scenes["test"], device, args.threshold, patch_size, context_scales)
        if patch_size
        else run_epoch(model, loaders["test"], device, args.threshold)
    )
    save_test_preview(
        model,
        scenes["test"],
        args.data,
        args.out,
        device,
        args.threshold,
        patch_size or None,
        context_scales,
    )
    summary = dict(
        configuration=configuration,
        best_epoch=best_epoch,
        best_validation=best_metrics,
        synthetic_test=test_metrics,
        elapsed_seconds=time.monotonic() - started,
        note="Small synthetic trials demonstrate the pipeline; they do not establish real-mine accuracy",
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_learning_report(args.out, configuration, history, summary, first_step)
    print(
        json.dumps(dict(best_epoch=best_epoch, synthetic_test=test_metrics, out=str(args.out))),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--scene-positive-fraction",
        type=float,
        default=0.0,
        help="Optional training-only positive scene draw probability; 0 keeps uniform sampling",
    )
    parser.add_argument("--target", choices=["physical", "fringe"], default="physical")
    parser.add_argument("--input", choices=["rgb", "phase"], default="rgb")
    parser.add_argument("--phase-bins", type=int, default=0)
    parser.add_argument(
        "--patch-sampling",
        choices=["all", "uniform", "balanced"],
        default="all",
        help="All patches, equal draws per scene, or training-mask-balanced scene draws",
    )
    parser.add_argument(
        "--patch-draws",
        type=int,
        default=3,
        help="Draws per original training scene per epoch in uniform/balanced mode",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=0,
        help="Crop larger scenes into aligned patches; 0 uses whole scenes",
    )
    parser.add_argument(
        "--patch-stride",
        type=int,
        default=0,
        help="Patch stride at scale 1; 0 uses 75%% of patch size",
    )
    parser.add_argument(
        "--context-scales",
        type=int,
        nargs="+",
        default=[1],
        help="Source context multipliers for patch training, e.g. 1 2",
    )
    parser.add_argument(
        "--phase-augment", action="store_true", help="Random phase origin and sign during training"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base", type=int, choices=[8, 16, 32, 64], default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        train(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
