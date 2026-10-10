"""Batch boundaries: explicit review authority, fresh sources and fixed grouped splits."""

import copy
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from simulator_batch import (
    accepted_templates,
    recover_completed,
    sample_paths,
    sample_record,
    sampling_plan,
)
from simulator_review import sha, write_json
from simulator_v3 import batch_configs, validate


class BatchPlanTests(unittest.TestCase):
    def setUp(self):
        self.templates = [
            {"id": f"S{i + 1:02d}", "revision": 1, "config": c, "group_id": f"baseline-{i}"}
            for i, c in enumerate(batch_configs())
        ]

    def test_fresh_reproducible_sources_and_exact_split_sizes(self):
        original = copy.deepcopy(self.templates)
        plan = sampling_plan(self.templates, 500, 20261010)
        self.assertEqual(plan, sampling_plan(self.templates, 500, 20261010))
        self.assertEqual(original, self.templates)
        self.assertEqual(Counter(r["split"] for r in plan), {"train": 400, "val": 50, "test": 50})
        seeds = [r["config"]["seed"] for r in plan]
        self.assertEqual(len(set(seeds)), 500)
        self.assertFalse(set(seeds) & {r["config"]["seed"] for r in self.templates})
        cases = defaultdict(set)
        for row in plan:
            c = row["config"]
            self.assertEqual(c, validate(c))
            self.assertEqual(c["boundary_rad"], 0.8)
            self.assertEqual(c["footprint"], "rectangular")
            self.assertEqual(c["style"], "bmp")
            cases[c["case"]].add(row["split"])
        self.assertTrue(all(s == {"train", "val", "test"} for s in cases.values()))
        self.assertEqual(sum(r["config"]["case"] == "negative" for r in plan), 50)

    def test_arbitrary_batch_counts_keep_requested_holdouts(self):
        for count in (20, 37, 50, 199):
            plan = sampling_plan(self.templates, count, 999)
            counts = Counter(r["split"] for r in plan)
            self.assertEqual(counts["test"], count // 10)
            self.assertEqual(counts["val"], count // 10)
            self.assertEqual(counts["train"], count - 2 * (count // 10))
        self.assertNotEqual(
            sampling_plan(self.templates, 20, 998)[0]["config"],
            sampling_plan(self.templates, 20, 999)[0]["config"],
        )


class BaselineAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rows = []
        for i, c in enumerate(batch_configs()):
            sid = f"S{i + 1:02d}"
            scene = self.root / sid
            scene.mkdir()
            Image.new("RGB", (8, 8), (i, 80, 100)).save(scene / "bmp.png")
            Image.new("L", (8, 8), 0).save(scene / "mask.png")
            (scene / "arrays.npz").write_bytes(b"hash-only test fixture")
            write_json(scene / "metadata.json", {"config": c, "group_id": f"source-{i}"})
            write_json(
                scene / "review.json",
                {
                    "status": "accepted",
                    "accepted_receipt": {
                        "image_sha256": sha(scene / "bmp.png"),
                        "mask_sha256": sha(scene / "mask.png"),
                    },
                },
            )
            rows.append({"id": sid, "folder": sid, "revision": 1})
        write_json(self.root / "manifest.json", {"samples": rows})
        write_json(self.root / "display_profile.json", None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pending_baseline_cannot_authorize_bulk_generation(self):
        self.assertEqual(len(accepted_templates(self.root)[0]), 20)
        write_json(self.root / "S13/review.json", {"status": "pending"})
        with self.assertRaisesRegex(ValueError, "not explicitly accepted"):
            accepted_templates(self.root)

    def test_changed_label_invalidates_the_acceptance_receipt(self):
        Image.new("L", (8, 8), 255).save(self.root / "S16/mask.png")
        with self.assertRaisesRegex(ValueError, "changed after acceptance"):
            accepted_templates(self.root)

    def test_resume_preserves_complete_files_and_rejects_tampering_or_partial_files(self):
        templates, _ = accepted_templates(self.root)
        plan = sampling_plan(templates, 20, 777)
        batch = self.root / "batch"
        for folder in ("images", "masks", "arrays", "metadata", "labelme"):
            (batch / folder).mkdir(parents=True)
        row = plan[0]
        paths = sample_paths(row["id"])
        for path in paths.values():
            (batch / path).write_bytes(b"complete-file protocol fixture")
        meta = {
            "config": row["config"],
            "sample_id": row["id"],
            "split": row["split"],
            "template_id": row["template_id"],
            "group_id": "new-source",
            "label_percent": 3.0,
            "water_percent": 0.0,
            "warnings": [],
        }
        write_json(batch / paths["metadata"], meta)
        record = sample_record(batch, row, meta)
        before = {p: sha(batch / p) for p in paths.values()}
        self.assertEqual(recover_completed(batch, plan, []), [record])
        self.assertEqual(recover_completed(batch, plan, [record]), [record])
        self.assertEqual(before, {p: sha(batch / p) for p in paths.values()})
        (batch / paths["mask"]).write_bytes(b"altered mask")
        with self.assertRaisesRegex(ValueError, "changed before resuming"):
            recover_completed(batch, plan, [record])
        (batch / paths["mask"]).write_bytes(b"complete-file protocol fixture")
        (batch / sample_paths(plan[1]["id"])["image"]).write_bytes(b"partial image")
        with self.assertRaisesRegex(ValueError, "Partial or non-contiguous"):
            recover_completed(batch, plan, [record])


if __name__ == "__main__":
    unittest.main()
