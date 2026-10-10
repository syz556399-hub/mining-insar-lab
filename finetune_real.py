"""Synthetic-pretrained versus real-final-label fine-tuning: development comparison."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from annotation_audit import sha256
from real_labels import FinalLabelScene, RealCrops, disjoint_dates
from segmentation import SmallUNet, select_device
from train import run_epoch


def finetune(args):
    if args.out.exists():
        raise ValueError("Output exists; use a new run directory")
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or not 0 < args.lr < 1:
        raise ValueError("Invalid training configuration")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["input_mode"] != "phase" or checkpoint.get("phase_bins") != 16:
        raise ValueError("Require a 16-bin phase checkpoint")
    scenes = {
        name: FinalLabelScene(labels, source)
        for name, labels, source in [
            ("train", args.train_labels, args.train_source),
            ("val", args.val_labels, args.val_source),
        ]
    }
    disjoint_dates(scenes["train"].provenance, scenes["val"].provenance)
    datasets = {
        s: RealCrops(scene, args.size, args.draws, args.seed, training=s == "train")
        for s, scene in scenes.items()
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        s: DataLoader(
            d,
            batch_size=args.batch_size,
            shuffle=s == "train",
            generator=generator if s == "train" else None,
        )
        for s, d in datasets.items()
    }
    model = SmallUNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True)
    config = dict(
        experiment="synthetic-pretrained versus real-final-label fine-tuning",
        init_checkpoint_sha256=sha256(args.checkpoint),
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=str(device),
        threads=args.threads,
        crop_size=args.size,
        context_scales=[1],
        native_pixels=True,
        training_draws_per_tile=args.draws,
        sampling="alternating positive-centered and uniform crops; aligned spatial flips",
        selection="maximum development validation pixel IoU; baseline eligible; earliest epoch on ties",
        threshold=0.5,
        display_zero_policy="predict",
        evaluation="nonoverlapping native blocks within saved annotation tiles; padding excluded; overlapping source tiles count as separate views",
        claims="same-mine cross-date DEVELOPMENT comparison; not independent or cross-mine testing",
        splits={s: scene.provenance for s, scene in scenes.items()},
        dataset={"label_rule": "user-confirmed final per-file LabelMe masks", "size": args.size},
        limitations="semantic segmentation only; source object records retained; no trained instance-count head",
    )
    (args.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    started = time.monotonic()
    baseline = run_epoch(model, loaders["val"], device, 0.5)
    best_score, best_epoch, best_metrics = baseline["foreground_iou"], 0, baseline
    if best_score is None:
        raise ValueError("Validation has no foreground target or prediction")

    def save(epoch, metrics):
        torch.save(
            dict(
                checkpoint_format=1,
                model_config=checkpoint["model_config"],
                state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                input_mode="phase",
                phase_bins=16,
                tile_size=args.size,
                context_scales=[1],
                threshold=0.5,
                epoch=epoch,
                val_metrics=metrics,
                configuration=config,
            ),
            args.out / "best.pt",
        )

    save(0, baseline)
    print(json.dumps({"baseline": baseline}), flush=True)
    history, trace = [], {}
    for epoch in range(1, args.epochs + 1):
        datasets["train"].epoch = epoch
        metrics = run_epoch(
            model, loaders["train"], device, 0.5, optimizer, trace if epoch == 1 else None
        )
        validation = run_epoch(model, loaders["val"], device, 0.5)
        row = dict(
            epoch=epoch,
            train_loss=metrics["loss"],
            val_loss=validation["loss"],
            val_iou=validation["foreground_iou"],
            val_precision=validation["precision"],
            val_recall=validation["recall"],
        )
        history.append(row)
        with (args.out / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        if validation["foreground_iou"] is not None and validation["foreground_iou"] > best_score:
            best_score, best_epoch, best_metrics = validation["foreground_iou"], epoch, validation
            save(epoch, validation)
        print(json.dumps(row), flush=True)
    summary = dict(
        baseline=baseline,
        best_finetuned_or_baseline=best_metrics,
        best_epoch=best_epoch,
        elapsed_seconds=time.monotonic() - started,
        independent_test_performed=False,
        training_objects=scenes["train"].provenance["annotation_objects"],
        validation_objects=scenes["val"].provenance["annotation_objects"],
        interpretation=config["claims"],
    )
    (args.out / "first_step.json").write_text(json.dumps(trace, indent=2))
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["checkpoint", "train-labels", "train-source", "val-labels", "val-source", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    for name, default in [
        ("epochs", 6),
        ("size", 256),
        ("draws", 4),
        ("batch-size", 8),
        ("seed", 42),
        ("threads", 2),
    ]:
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu")
    finetune(p.parse_args())
