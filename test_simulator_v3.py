"""Scientific invariants and exact label/export review semantics for the reboot."""

import base64
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from annotation_audit import decode_annotation_mask
from mine_model import Face, panel_epochs
from simulator_review import (
    boundary_preview,
    decide,
    export_archive,
    read_json,
    regenerate,
    save_boundary,
    write_json,
)
from simulator_v3 import (
    batch_configs,
    block_mean,
    bmp_rgb,
    candidate_mask,
    correlated_pair,
    defaults,
    footprint_epochs,
    mined_strips,
    save_scene,
    simulate,
    validate,
)


class ObservationTests(unittest.TestCase):
    def test_color_table_wraps_float32_endpoint_without_overflow(self):
        phase = np.array([[-1e-8, 0, 2 * np.pi, -2 * np.pi]], dtype=np.float32)
        arrays = {
            "wrapped_phase_rad": phase,
            "intensity_before": np.ones_like(phase),
            "intensity_after": np.ones_like(phase),
            "water_fraction": np.zeros_like(phase),
        }
        colors = [[i, 0, 255 - i] for i in range(16)]
        profile = {"color_table": colors, "brightness_weights": [0] * 15 + [1]}
        image = bmp_rgb(arrays, profile)
        np.testing.assert_array_equal(image, [[colors[15], colors[0], colors[0], colors[15]]])
        np.testing.assert_array_equal(arrays["wrapped_phase_rad"], phase)

    def test_phase_sign_and_complex_spatial_average(self):
        phase = np.array([[3.0, -3.0], [3.0, -3.0]])
        cross, _, _, coh = correlated_pair(
            phase, np.ones((2, 2)), np.ones((2, 2)), np.random.default_rng(2), 500, 2
        )
        # Averaging wrapped radians would give zero. Complex averaging stays near pi.
        self.assertGreater(abs(float(np.angle(cross)[0, 0])), 3.0)
        self.assertGreater(coh[0, 0], 0.98)
        phase = np.full((16, 16), -1.2)
        cross, _, _, _ = correlated_pair(
            phase, np.ones((16, 16)), np.ones((16, 16)), np.random.default_rng(7), 1, 1
        )
        np.testing.assert_allclose(np.angle(cross), phase, atol=1e-12)

    def test_expected_complex_correlation(self):
        phase = np.full((128, 128), 0.6)
        for gamma in (0.0, 0.65):
            cross, p1, p2, _ = correlated_pair(
                phase,
                np.full_like(phase, gamma),
                np.ones_like(phase),
                np.random.default_rng(66),
                8,
                1,
            )
            correlation = np.mean(cross) * np.exp(-0.6j) / np.sqrt(p1.mean() * p2.mean())
            self.assertAlmostEqual(correlation.real, gamma, delta=0.02)
            self.assertAlmostEqual(correlation.imag, 0, delta=0.02)

    def test_block_average_requires_aligned_dimensions(self):
        with self.assertRaises(ValueError):
            block_mean(np.ones((3, 4)), 2)


class SourceTests(unittest.TestCase):
    def test_gentle_domain_limits_bending_and_width_variation(self):
        face = Face()
        for seed in range(8):
            _, profile = mined_strips(face, seed, 0, mode="gentle")
            center = np.asarray(profile["center_knots_m"])
            width = np.asarray(profile["width_knots_m"])
            self.assertLessEqual(np.max(np.abs(center)), 0.25 * face.width_m)
            self.assertGreater(width.min(), 0.75 * face.width_m)
            self.assertLess(width.max(), 1.35 * face.width_m)
            slopes = np.diff(center)
            signs = np.sign(slopes[np.abs(slopes) > 1e-12])
            self.assertLessEqual(np.count_nonzero(signs[1:] != signs[:-1]), 1)

    def test_constant_strip_footprint_equals_rectangle(self):
        face = Face()
        edges = np.linspace(-face.length_m / 2, face.length_m / 2, 33)
        strips = [
            {
                "along_lower_m": low,
                "along_upper_m": high,
                "across_lower_m": -face.width_m / 2,
                "across_upper_m": face.width_m / 2,
            }
            for low, high in zip(edges[:-1], edges[1:])
        ]
        original = panel_epochs(face, 128, 2200, 32, 70, 82)
        actual = footprint_epochs(face, 128, 2200, strips, 70, 82)
        for a, b in zip(actual, original):
            np.testing.assert_array_equal(a, b)
        strips[1]["along_lower_m"] = strips[0]["along_lower_m"]
        with self.assertRaises(ValueError):
            footprint_epochs(face, 128, 2200, strips, 70, 82)

    def test_irregular_domains_recorded_and_legacy_requests_preserved(self):
        strips, profile = mined_strips(Face(), 99, 0)
        self.assertEqual(strips, profile["strips"])
        self.assertGreater(np.ptp(profile["width_knots_m"]), 0)
        self.assertGreater(np.ptp(profile["center_knots_m"]), 0)
        self.assertEqual(strips, mined_strips(Face(), 99, 0)[0])
        old = {key: value for key, value in self.config.items() if key != "footprint"}
        self.assertEqual(validate({})["footprint"], "rectangular")
        self.assertEqual(validate(old)["footprint"], "rectangular")
        a, _ = simulate(old)
        b, _ = simulate({**old, "footprint": "rectangular"})
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    @classmethod
    def setUpClass(cls):
        cls.config = {**defaults(), "size": 128, "spatial_factor": 1, "shape_warp": 0.0}
        cls.arrays, cls.meta = simulate(cls.config)

    def test_sign_units_and_source_consistency(self):
        a = self.arrays
        expected = a["delta_down_m"] * np.cos(np.deg2rad(33)) - a["delta_east_m"] * np.sin(
            np.deg2rad(33)
        )
        np.testing.assert_allclose(a["delta_los_m"], expected, atol=1e-7)
        np.testing.assert_allclose(
            a["deformation_phase_rad"], -4 * np.pi * a["delta_los_m"] / 0.055, atol=1e-5
        )
        np.testing.assert_allclose(
            a["reference_phase_rad"],
            a["deformation_phase_rad"] + a["background_phase_rad"],
            atol=1e-5,
        )
        self.assertGreater(a["delta_down_m"].max(), 0)
        self.assertEqual(a["candidate_mask"].dtype, np.uint8)
        self.assertTrue(np.isin(a["candidate_mask"], [0, 1]).all())
        self.assertTrue(all(np.isfinite(v).all() and v.shape == (128, 128) for v in a.values()))

    def test_reproducible_and_nuisance_does_not_relabel_source(self):
        a2, m2 = simulate(self.config)
        for key in self.arrays:
            np.testing.assert_array_equal(a2[key], self.arrays[key])
        self.assertEqual(m2["group_id"], self.meta["group_id"])
        a3, m3 = simulate(
            {
                **self.config,
                "style": "research",
                "coherence": 0.4,
                "atmosphere_mm": 8,
                "water": True,
            }
        )
        for key in ("deformation_phase_rad", "delta_down_m", "candidate_mask"):
            np.testing.assert_array_equal(a3[key], self.arrays[key])
        self.assertEqual(m3["group_id"], self.meta["group_id"])
        self.assertFalse(np.array_equal(a3["wrapped_phase_rad"], self.arrays["wrapped_phase_rad"]))

    def test_negative_and_boundary_are_explicit(self):
        arrays, _ = simulate({**self.config, "case": "negative"})
        self.assertFalse(arrays["candidate_mask"].any())
        self.assertFalse(arrays["deformation_phase_rad"].any())
        self.assertGreater(arrays["background_phase_rad"].std(), 0)
        smaller = candidate_mask(self.arrays["deformation_phase_rad"], 1.0)
        self.assertTrue(np.all(smaller <= self.arrays["candidate_mask"]))
        self.assertLess(smaller.sum(), self.arrays["candidate_mask"].sum())

    def test_review_batch_has_two_display_targets_and_negatives(self):
        configs = batch_configs()
        self.assertEqual(len(configs), 20)
        self.assertEqual(len({c["seed"] for c in configs}), 20)
        self.assertEqual(sum(c["style"] == "bmp" for c in configs), 20)
        self.assertEqual(sum(c["case"] == "negative" for c in configs), 2)
        self.assertEqual(sum(c["water"] for c in configs), 4)


class ReviewExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        self.config = {**defaults(), "size": 128, "spatial_factor": 1}
        arrays, meta = simulate(self.config)
        save_scene(self.run / "S01", arrays, meta, None)
        write_json(self.run / "display_profile.json", None)
        write_json(
            self.run / "manifest.json", {"samples": [{"id": "S01", "folder": "S01", "revision": 1}]}
        )
        write_json(
            self.run / "S01/review.json",
            {
                "status": "pending",
                "revision": 1,
                "threshold_rad": 0.4,
                "manual_mask": False,
                "note": "",
                "history": [],
            },
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_implicit_approval_and_draft_marks(self):
        with self.assertRaises(ValueError):
            export_archive(self.run)
        data, doc = export_archive(self.run, draft=True)
        self.assertEqual(doc["status"], "draft")
        self.assertEqual(doc["samples"][0]["split"], "unassigned")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            self.assertIn("load_dataset.py", archive.namelist())
            self.assertFalse(any("reference" in p for p in archive.namelist()))

    def test_manual_holes_tiny_objects_labelme_and_accepted_export(self):
        exact = np.zeros((128, 128), dtype=np.uint8)
        exact[30:70, 40:90] = 1
        exact[40:50, 50:60] = 0
        exact[120, 120] = 1
        buf = io.BytesIO()
        Image.fromarray(exact * 255).save(buf, format="PNG")
        save_boundary(
            self.run,
            "S01",
            {"mask_png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()},
        )
        decide(self.run, "S01", {"status": "accepted", "note": "test fixture, not user approval"})
        data, manifest = export_archive(self.run)
        self.assertEqual(len(manifest["samples"]), 1)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            with Image.open(io.BytesIO(archive.read("masks/S01.png"))) as image:
                np.testing.assert_array_equal(np.asarray(image) > 0, exact > 0)
            import json

            doc = json.loads(archive.read("labelme/S01.json"))
            np.testing.assert_array_equal(decode_annotation_mask(doc), exact > 0)
            self.assertEqual(
                json.loads(archive.read("metadata/S01.json"))["label_rule"],
                "manual exact raster boundary",
            )
        save_boundary(self.run, "S01", {"threshold_rad": 0.8})
        self.assertEqual(read_json(self.run / "S01/review.json")["status"], "pending")
        with self.assertRaises(ValueError):
            export_archive(self.run)

    def test_receipt_detects_post_review_changes(self):
        decide(self.run, "S01", {"status": "accepted"})
        Image.fromarray(np.zeros((128, 128), dtype=np.uint8)).save(self.run / "S01/mask.png")
        with self.assertRaises(ValueError):
            export_archive(self.run)

    def test_regeneration_preserves_previous_version(self):
        original = (self.run / "S01/arrays.npz").read_bytes()
        new = regenerate(self.run, "S01", {**self.config, "seed": 8})
        self.assertEqual(new["revision"], 2)
        self.assertEqual((self.run / "S01/arrays.npz").read_bytes(), original)
        self.assertEqual(read_json(self.run / "S01_r2/review.json")["status"], "pending")

    def test_boundary_preview_does_not_mutate_labels(self):
        original = (self.run / "S01/mask.png").read_bytes()
        for kind in ("input", "wide", "current", "tight"):
            raw = boundary_preview(self.run / "S01", "bmp", kind)
            with Image.open(io.BytesIO(raw)) as image:
                self.assertEqual(image.size, (320, 320))
        self.assertEqual((self.run / "S01/mask.png").read_bytes(), original)
        save_boundary(self.run, "S01", {"threshold_rad": 0.8})
        old = read_json(self.run / "S01/review.json")["history"][-1]["previous_mask_file"]
        self.assertEqual((self.run / "S01" / old).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
