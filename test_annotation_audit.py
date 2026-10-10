"""Tests for spatial label integrity, small targets and provisional metrics."""

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from annotation_audit import connected_regions, read_annotations, select_review_tiles
from scale_audit import scores


class AnnotationAuditTests(unittest.TestCase):
    def test_diagonals_join_and_single_pixel_survives(self):
        mask = np.zeros((5, 6), dtype=bool)
        mask[0, 0] = mask[1, 1] = mask[2, 2] = mask[4, 5] = True
        regions = connected_regions(mask)
        self.assertEqual([r["area_px"] for r in regions], [3, 1])
        self.assertEqual(regions[0]["bbox"], [0, 0, 3, 3])
        self.assertEqual(regions[1]["bbox"], [5, 4, 6, 5])
        self.assertEqual(connected_regions(np.zeros((2, 2))), [])

    def test_components_match_independent_flood_fill(self):
        for seed in range(12):
            mask = np.random.default_rng(seed).random((13, 17)) > 0.7
            unseen = set(map(tuple, np.argwhere(mask)))
            areas = []
            while unseen:
                todo, count = [unseen.pop()], 0
                while todo:
                    y, x = todo.pop()
                    count += 1
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            p = y + dy, x + dx
                            if p in unseen:
                                unseen.remove(p)
                                todo.append(p)
                areas.append(count)
            self.assertEqual(sorted(areas), sorted(r["area_px"] for r in connected_regions(mask)))

    def test_overlapping_annotations_union_and_disagreement_are_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = dict(
                width=4,
                height=2,
                tiles=[
                    dict(file="a.png", bbox=[0, 0, 3, 2]),
                    dict(file="b.png", bbox=[1, 0, 4, 2]),
                ],
            )
            (root / "manifest.json").write_text(json.dumps(manifest))
            stream = io.BytesIO()
            Image.fromarray(np.ones((1, 1), dtype=np.uint8) * 255).save(stream, format="PNG")
            shape = dict(
                label="1",
                shape_type="mask",
                points=[[1, 0], [1, 0]],
                mask=base64.b64encode(stream.getvalue()).decode(),
            )
            for name, shapes in [("a", [shape]), ("b", [])]:
                (root / f"{name}.json").write_text(
                    json.dumps(
                        dict(imagePath=f"{name}.png", imageWidth=3, imageHeight=2, shapes=shapes)
                    )
                )
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            truth, coverage, conflict, records, _ = read_annotations(root)
            self.assertEqual(truth.sum(), 1)
            self.assertTrue(truth[0, 1])
            self.assertEqual(conflict.sum(), 1)
            self.assertTrue(coverage.all())
            self.assertEqual(len(records), 2)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            doc = json.loads((root / "a.json").read_text())
            doc["shapes"][0]["points"][0] = [3, 0]
            (root / "a.json").write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "beyond"):
                read_annotations(root)

    def test_review_selection_and_scope(self):
        records = [dict(file=str(i), positive_pixels=0) for i in range(25)]
        chosen = select_review_tiles(records)
        self.assertEqual(len(chosen), 20)
        self.assertEqual(chosen, select_review_tiles(records))
        self.assertEqual(len({r["file"] for r in chosen}), 20)
        metrics = scores(
            np.array([[1, 1, 0]], bool), np.array([[1, 0, 1]], bool), np.array([[1, 0, 1]], bool)
        )
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 0, 1))
        self.assertEqual(metrics["iou"], 0.5)


if __name__ == "__main__":
    unittest.main()
