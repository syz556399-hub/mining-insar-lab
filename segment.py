"""Run synthetic scene generation, aligned patch training and image segmentation."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from data_io import export_dataset
from mine_model import parse_request, request_dict
from predict import predict
from train import train


def run(args):
    if args.out.exists():
        raise ValueError("Output exists; choose a new experiment directory")
    if args.patch_size * max(args.context_scales) > args.scene_size:
        raise ValueError("Generated scenes must fit the largest training context")
    # Validate the original image before spending time generating/training.
    if args.image is not None:
        from PIL import Image

        from phase_encoding import decode_display_palette

        with Image.open(args.image) as image:
            if not args.palette_code:
                raise ValueError("This phase workflow needs numeric arrays or --palette-code")
            decode_display_palette(image)
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(
        state="running",
        supervision="synthetic vertical subsidence increment mask",
        real_annotations_used_for_training=False,
        steps=[],
    )

    def record(step, result):
        report["steps"].append(dict(step=step, result=result))
        (args.out / "run.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(dict(step=step, state="complete"), ensure_ascii=False), flush=True)

    base = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    base.setdefault("settings", {}).update(size=args.scene_size, seed=args.seed)
    settings, faces = parse_request(base)
    data = args.out / "dataset"
    result = export_dataset(
        data,
        args.count,
        request_dict(settings, faces),
        profile=args.profile,
        progress=lambda done, total: (
            print(json.dumps(dict(generated=done, total=total)), flush=True)
            if done % 10 == 0 or done == total
            else None
        ),
    )
    record("generate", result)
    training = args.out / "training"
    summary = train(
        SimpleNamespace(
            data=data,
            out=training,
            input="phase",
            phase_bins=args.phase_bins,
            phase_augment=True,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            context_scales=args.context_scales,
            patch_sampling=getattr(args, "patch_sampling", "all"),
            patch_draws=getattr(args, "patch_draws", 3),
            epochs=args.epochs,
            batch_size=args.batch_size,
            base=8,
            lr=0.001,
            seed=args.seed,
            threshold=0.5,
            device=args.device,
            threads=args.threads,
        )
    )
    record(
        "train",
        dict(
            best_epoch=summary["best_epoch"],
            synthetic_test=summary["synthetic_test"],
            output=str(training.resolve()),
        ),
    )
    # Without a supplied real image, demonstrate one fixed held-out numeric scene.
    if args.image is None:
        import csv

        with (data / "manifest.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        selected = next(
            row for row in rows if row["split"] == "test" and row["category"] == "single"
        )
        arrays = data / selected["arrays"]
    else:
        arrays = None
    prediction = args.out / "prediction"
    result = predict(
        SimpleNamespace(
            checkpoint=training / "best.pt",
            arrays=arrays,
            image=args.image,
            palette_code=args.palette_code,
            out=prediction,
            device=args.device,
            threshold=0.5,
            stride=None,
            threads=args.threads,
            downsample=1,
            scales=(args.scales or args.context_scales) if args.image else args.context_scales,
        )
    )
    record("predict", result)
    report["state"] = "complete"
    (args.out / "run.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    root = Path(__file__).resolve().parent
    if prediction.resolve().is_relative_to((root / "runs").resolve()):
        pointer = dict(prediction=prediction.resolve().relative_to(root.resolve()).as_posix())
        (root / "runs/segmentation_latest.json").write_text(json.dumps(pointer), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--scene-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patch-stride", type=int, default=96)
    parser.add_argument("--patch-sampling", choices=["all", "uniform", "balanced"], default="all")
    parser.add_argument("--patch-draws", type=int, default=3)
    parser.add_argument(
        "--context-scales", type=int, nargs="+", choices=[1, 2, 4, 8], default=[1, 2]
    )
    parser.add_argument("--profile", choices=["standard", "sparse_mine"], default="sparse_mine")
    parser.add_argument("--phase-bins", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--palette-code", action="store_true")
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        choices=[1, 2, 4, 8],
        help="Explicit prediction context; by default use the training context scales",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    try:
        run(parser.parse_args())
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
