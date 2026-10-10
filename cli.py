"""Portable command-line access to reproducible simulation and dataset generation."""

import argparse
import json
from pathlib import Path

from data_io import export_dataset, single_archive
from mine_model import default_request, parse_request, request_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    config = sub.add_parser("config", help="Write the normalized default parameter file")
    config.add_argument("--out", type=Path, required=True)
    scene = sub.add_parser("scene", help="Export one scene as ZIP")
    scene.add_argument("--config", type=Path)
    scene.add_argument("--out", type=Path, required=True)
    dataset = sub.add_parser("dataset", help="Generate stratified independent scenes")
    dataset.add_argument("--config", type=Path)
    dataset.add_argument("--count", type=int, default=50)
    dataset.add_argument("--profile", choices=["standard", "sparse_mine"], default="standard")
    dataset.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    request = (
        json.loads(args.config.read_text(encoding="utf-8"))
        if getattr(args, "config", None)
        else default_request()
    )
    settings, faces = parse_request(request)
    request = request_dict(settings, faces)
    if args.out.exists():
        parser.error("Output exists; choose a new path")
    if args.command == "config":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    elif args.command == "scene":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(single_archive(request))
    else:
        export_dataset(
            args.out,
            args.count,
            request,
            lambda n, total: print(f"{n}/{total}", flush=True) if n % 10 == 0 else None,
            profile=args.profile,
        )
    print(args.out)


if __name__ == "__main__":
    main()
