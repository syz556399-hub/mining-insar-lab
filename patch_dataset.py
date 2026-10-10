"""Aligned scene patches; every patch inherits its source scene's split."""

import csv
import json
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image, ImageDraw

from data_io import phase_rgb
from segmentation import downsample_channels, tile_starts


class ScenePatchDataset:
    """Crop observed input, target and validity together at several context sizes.

    Coordinates use source pixels. Larger context is averaged for the input and
    nearest-neighbour sampled for the binary target/validity. Contexts must fit.
    Patch selection never depends on predictions or target presence.
    """

    def __init__(self, scenes, scene_size, patch_size, stride, context_scales=(1,), augment=False):
        if type(patch_size) is not int or patch_size < 8:
            raise ValueError("patch_size must be an integer >= 8")
        if not 1 <= stride <= patch_size:
            raise ValueError("Require 1 <= patch stride <= patch_size")
        scales = tuple(context_scales)
        if not scales or len(set(scales)) != len(scales):
            raise ValueError("Context scales must be nonempty and unique")
        if any(type(scale) is not int or scale < 1 for scale in scales):
            raise ValueError("Context scales must be positive integers")
        if any(patch_size * scale > scene_size for scale in scales):
            raise ValueError("Each context must fit the generated scene; increase scene size")
        if not scenes.return_valid or scenes.augment_phase:
            raise ValueError("Use unaugmented scene tensors with return_valid=True")
        self.scenes = scenes
        self.root = scenes.root
        self.input_mode = scenes.input_mode
        self.patch_size = patch_size
        self.augment = augment
        self.rows = []
        self.cache = OrderedDict()
        for scene_index, row in enumerate(scenes.rows):
            for scale in scales:
                span = patch_size * scale
                for top in tile_starts(scene_size, span, stride * scale):
                    for left in tile_starts(scene_size, span, stride * scale):
                        self.rows.append(
                            dict(
                                row,
                                scene_index=scene_index,
                                context_scale=scale,
                                left=left,
                                top=top,
                                right=left + span,
                                bottom=top + span,
                            )
                        )

    def __len__(self):
        return len(self.rows)

    def _scene(self, index):
        if index not in self.cache:
            self.cache[index] = self.scenes[index]
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def __getitem__(self, index):
        row = self.rows[index]
        channels, target, valid = self._scene(row["scene_index"])
        top, bottom, left, right = (row[k] for k in ("top", "bottom", "left", "right"))
        channels = channels.numpy()[:, top:bottom, left:right].copy()
        target = target.numpy()[0, top:bottom, left:right]
        valid = valid.numpy()[0, top:bottom, left:right]
        if channels.shape[-2:] != (bottom - top, right - left):
            raise ValueError("Patch coordinates extend beyond the source scene")
        scale = row["context_scale"]
        channels = downsample_channels(channels, scale, self.input_mode == "phase")
        if scale != 1:
            shape = (self.patch_size, self.patch_size)
            target = np.asarray(Image.fromarray(target).resize(shape, Image.Resampling.NEAREST))
            valid = np.asarray(Image.fromarray(valid).resize(shape, Image.Resampling.NEAREST))
        if self.augment:
            # Flip all three tensors together; phase origin/sign change only input.
            for axis in (1, 2):
                if np.random.random() < 0.5:
                    channels = np.flip(channels, axis)
                    target = np.flip(target, axis - 1)
                    valid = np.flip(valid, axis - 1)
            if self.input_mode == "phase":
                sine, cosine = channels
                sign = np.random.choice([-1, 1])
                offset = np.random.uniform(-np.pi, np.pi)
                channels = np.stack(
                    (
                        sign * sine * np.cos(offset) + cosine * np.sin(offset),
                        cosine * np.cos(offset) - sign * sine * np.sin(offset),
                    )
                )
        return tuple(
            torch.from_numpy(np.array(value, dtype=np.float32, copy=True))
            for value in (channels, target[None], valid[None])
        )

    def save_manifest(self, path):
        fields = [
            "scene_id",
            "group_id",
            "split",
            "category",
            "arrays",
            "context_scale",
            "left",
            "top",
            "right",
            "bottom",
        ]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

    def save_examples(self, output):
        """Six fixed windows from the first nonnegative training scene, without augmentation."""
        scene_index = next((r["scene_index"] for r in self.rows if r["category"] == "single"), 0)
        candidates = [i for i, r in enumerate(self.rows) if r["scene_index"] == scene_index]
        indices = sorted(
            set(
                int(candidates[i])
                for i in np.linspace(0, len(candidates) - 1, min(6, len(candidates)), dtype=int)
            )
        )
        folder = output / "patch_examples"
        folder.mkdir(exist_ok=False)
        board = Image.new("RGB", (576, 226 * len(indices)), "#101722")
        draw = ImageDraw.Draw(board)
        augmented = self.augment
        self.augment = False
        records = []
        try:
            for order, index in enumerate(indices):
                channels, target, valid = self[index]
                observed = (
                    phase_rgb(np.arctan2(channels[0], channels[1]))
                    if self.input_mode == "phase"
                    else np.round(channels.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                panels = [
                    Image.fromarray(observed),
                    Image.fromarray((target[0].numpy() * 255).astype(np.uint8)),
                    Image.fromarray((valid[0].numpy() * 255).astype(np.uint8)),
                ]
                row = self.rows[index]
                caption = f"{row['scene_id']}  crop=({row['left']},{row['top']},{row['right']},{row['bottom']})  context={row['context_scale']}"
                draw.text((6, order * 226 + 6), caption, fill="white")
                for column, (label, panel) in enumerate(zip(("input", "mask", "valid"), panels)):
                    panel.save(folder / f"patch_{order:02d}_{label}.png")
                    board.paste(
                        panel.convert("RGB").resize((192, 192), Image.Resampling.NEAREST),
                        (column * 192, order * 226 + 28),
                    )
                records.append(dict(row, example_index=order))
        finally:
            self.augment = augmented
        (folder / "examples.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        board.save(output / "patch_examples.png")


class ScenePatchSampler(torch.utils.data.Sampler):
    """A fixed draw budget per original training scene, with epoch-seeded choices.

    Uniform draws and positive/background-balanced draws have identical budgets.
    Balance uses training masks only. Validation and test sampling are unchanged.
    """

    def __init__(self, dataset, draws_per_scene=3, balanced=False, seed=42):
        if type(draws_per_scene) is not int or draws_per_scene < 1:
            raise ValueError("draws_per_scene must be a positive integer")
        if not dataset.rows or any(row["split"] != "train" for row in dataset.rows):
            raise ValueError("Scene sampling is restricted to the training split")
        self.seed, self.epoch = seed, 0
        self.draws_per_scene = draws_per_scene
        self.balanced = balanced
        self.groups = {}
        previous = dataset.augment
        dataset.augment = False
        try:
            for index, row in enumerate(dataset.rows):
                _, target, valid = dataset[index]
                usable = valid.numpy()[0] > 0.5
                if not usable.any():
                    continue
                positive = bool(((target.numpy()[0] > 0.5) & usable).any())
                group = self.groups.setdefault(row["scene_id"], {"positive": [], "background": []})
                group["positive" if positive else "background"].append(index)
        finally:
            dataset.augment = previous
        if not self.groups:
            raise ValueError("No usable training patches")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.groups) * self.draws_per_scene

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 771]))
        indices = []
        for group in self.groups.values():
            positives, background = group["positive"], group["background"]
            for _ in range(self.draws_per_scene):
                pool = positives + background
                if self.balanced and positives and background:
                    pool = positives if rng.random() < 0.5 else background
                indices.append(int(rng.choice(pool)))
        rng.shuffle(indices)
        return iter(indices)

    def summary(self):
        return dict(
            mode="balanced" if self.balanced else "uniform",
            draws_per_scene=self.draws_per_scene,
            draws_per_epoch=len(self),
            usable_scenes=len(self.groups),
            positive_patches=sum(len(group["positive"]) for group in self.groups.values()),
            background_patches=sum(len(group["background"]) for group in self.groups.values()),
            rule="equal draws per training scene; balanced chooses positive/background with 0.5 probability when both exist",
            scope="training masks only; no validation/test/real annotations used",
        )
