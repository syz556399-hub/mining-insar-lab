"""Exact final LabelMe supervision; preserve per-file objects and mask pixels."""

import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from annotation_audit import decode_annotation_mask, read_annotations, sha256
from phase_encoding import decode_display_palette


class FinalLabelScene:
    def __init__(self, labels, source):
        labels, source = Path(labels), Path(source)
        _, _, _, records, manifest = read_annotations(labels)
        with Image.open(source) as original:
            if original.size != (manifest["width"], manifest["height"]):
                raise ValueError("Source dimensions differ from manifest")
            angle, visible, adapter = decode_display_palette(original)
            rgb = original.convert("RGB")
        channels = np.stack([np.sin(angle), np.cos(angle)]).astype(np.float32)
        channels[:, ~visible] = 0
        self.tiles, objects = [], []
        for row in records:
            doc = json.loads((labels / row["file"]).read_text(encoding="utf-8"))
            target = decode_annotation_mask(doc)
            x, y, right, bottom = row["bbox"]
            with Image.open(labels / row["image"]) as tile_image:
                if not np.array_equal(
                    np.asarray(tile_image.convert("RGB")), np.asarray(rgb.crop(row["bbox"]))
                ):
                    raise ValueError(f"Source pixel alignment differs: {row['file']}")
            ids = []
            for i, shape in enumerate(doc["shapes"]):
                object_id = f"{row['file']}#shape-{i}"
                ids.append(object_id)
                objects.append(
                    dict(
                        id=object_id,
                        annotation_file=row["file"],
                        shape_index=i,
                        label=shape["label"],
                        group_id=shape.get("group_id"),
                        points=shape["points"],
                        shape_type=shape["shape_type"],
                    )
                )
            self.tiles.append(
                dict(
                    record=row,
                    channels=channels[:, y:bottom, x:right].copy(),
                    target=target,
                    object_ids=ids,
                )
            )
        self.source, self.labels = source, labels
        self.provenance = dict(
            source_name=source.name,
            source_sha256=sha256(source),
            manifest_sha256=sha256(labels / "manifest.json"),
            annotations=records,
            annotation_files=len(records),
            annotation_objects=len(objects),
            objects=objects,
            adapter=adapter,
            label_status="user-confirmed final annotations",
            target_rule="exact mask union within each original LabelMe file",
            background_rule="unmarked pixels within saved final annotation tiles are background; unsaved tiles excluded",
            count_rule="preserve source shape records and group_id; cross-tile physical instance count not inferred",
            overlap_rule="retain each tile final annotation; per-tile metrics count overlapping views separately",
            display_zero_policy="zero input vector for absent display code; retain targets and do not suppress predictions",
        )


class RealCrops(Dataset):
    def __init__(self, scene, size=256, draws=4, seed=42, training=True):
        if size < 8 or draws < 1:
            raise ValueError("Require size >=8 and positive draws")
        self.scene, self.size, self.draws = scene, size, draws
        self.seed, self.training, self.epoch = seed, training, 0
        self.positions = []
        if not training:
            for i, tile in enumerate(scene.tiles):
                h, w = tile["target"].shape
                for y in range(0, h, size):
                    for x in range(0, w, size):
                        self.positions.append((i, y, x))

    def __len__(self):
        return len(self.scene.tiles) * self.draws if self.training else len(self.positions)

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        if self.training:
            tile_index, draw = divmod(index, self.draws)
            tile = self.scene.tiles[tile_index]
            h, w = tile["target"].shape
            if draw % 2 == 0 and tile["target"].any():
                pixels = np.argwhere(tile["target"])
                cy, cx = pixels[rng.integers(len(pixels))]
                y = int(np.clip(cy - rng.integers(self.size), 0, max(0, h - self.size)))
                x = int(np.clip(cx - rng.integers(self.size), 0, max(0, w - self.size)))
            else:
                y, x = (int(rng.integers(max(0, n - self.size) + 1)) for n in (h, w))
        else:
            tile_index, y, x = self.positions[index]
            tile = self.scene.tiles[tile_index]
        target = tile["target"][y : y + self.size, x : x + self.size].astype(np.float32)[None]
        inputs = tile["channels"][:, y : y + self.size, x : x + self.size]
        valid = np.ones_like(target)
        h, w = target.shape[-2:]
        padding = ((0, 0), (0, self.size - h), (0, self.size - w))
        inputs = np.pad(inputs, padding, mode="edge")
        target, valid = (np.pad(v, padding) for v in (target, valid))
        if self.training:
            for axis in (1, 2):
                if rng.random() < 0.5:
                    inputs, target, valid = (np.flip(v, axis=axis) for v in (inputs, target, valid))
        return tuple(
            torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
            for v in (inputs, target, valid)
        )


def disjoint_dates(train, val):
    if train["source_sha256"] == val["source_sha256"]:
        raise ValueError("Training and validation source are identical")
    dates = [set(re.findall(r"(?<!\d)\d{8}(?!\d)", info["source_name"])) for info in (train, val)]
    if any(len(d) != 2 for d in dates):
        raise ValueError("Expected two acquisition dates in each source filename")
    if dates[0] & dates[1]:
        raise ValueError("Training and validation share an acquisition date")
