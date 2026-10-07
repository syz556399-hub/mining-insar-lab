"""Create a public source snapshot with an explicit allowlist; never upload it."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from provenance import VERSION

ROOT_FILES = (
    "README.md",
    "MODEL.md",
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "VALIDATION.md",
    "requirements.txt",
    "requirements-lock.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    ".gitignore",
    "mine_model.py",
    "scene_environment.py",
    "provenance.py",
    "data_io.py",
    "server.py",
    "cli.py",
    "validate_dataset.py",
    "diagnostics.py",
    "prepare_release.py",
    "test_model.py",
    "test_environment.py",
    "test_scientific.py",
    "test_release.py",
    "启动实验室.command",
)
WEB_FILES = ("web/index.html", "web/app.js", "web/style.css")


def public_files(root):
    selected = [root / name for name in (*ROOT_FILES, *WEB_FILES)]
    for folder in ("docs", "examples", ".github"):
        selected.extend(p for p in (root / folder).rglob("*") if p.is_file())
    for path in selected:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or symlinked public file: {path.relative_to(root)}")
    return sorted(set(selected))


def prepare(root, output):
    name = f"mining-insar-lab-{VERSION}"
    target = output / name
    archive = output / (name + ".zip")
    if target.exists() or archive.exists():
        raise ValueError("Release output exists; choose a new --output directory")
    files = public_files(root)
    forbidden = ("/Users/", "/Volumes/", "/private/tmp/", "20250112_20250124.diff.bmp")
    for path in files:
        if path.suffix.lower() in (
            ".py",
            ".js",
            ".html",
            ".md",
            ".json",
            ".cff",
            ".toml",
            ".yml",
            ".command",
        ):
            text = path.read_text(encoding="utf-8")
            # The release checker itself necessarily contains its banned-marker rules.
            if path.name != "prepare_release.py" and any(marker in text for marker in forbidden):
                raise ValueError(f"Local path/private reference in {path.relative_to(root)}")
    records = []
    for path in files:
        relative = path.relative_to(root)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        content = destination.read_bytes()
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
    (target / "SOURCE_MANIFEST.json").write_text(
        json.dumps({"version": VERSION, "files": records}, indent=2), encoding="utf-8"
    )
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                stream.write(path, (Path(name) / path.relative_to(target)).as_posix())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return {
        "folder": str(target),
        "zip": str(archive),
        "sha256": digest,
        "public_files": len(records),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "dist")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(Path(__file__).resolve().parent, args.output), ensure_ascii=False, indent=2
        )
    )
