import colorsys
import unittest

import numpy as np
from PIL import Image

from data_io import draw_sample
from mine_model import default_request
from phase_encoding import decode_display_palette, quantized_phase


def indexed_image():
    # Independent artificial table: no measured image or private palette is embedded.
    colors = np.array([colorsys.hsv_to_rgb(k / 16, 0.7, 1) for k in range(16)])
    table = np.round(np.arange(16)[:, None, None] / 15 * colors * 255).astype(np.uint8)
    image = Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16))
    image.putpalette(table.reshape(-1).tolist())
    return image


class PhaseEncodingTests(unittest.TestCase):
    def test_display_code_invariant_to_brightness_and_zero_is_invisible(self):
        phase, visible, info = decode_display_palette(indexed_image())
        np.testing.assert_array_equal(phase[1], phase[15])
        self.assertFalse(visible[0].any())
        self.assertTrue(visible[1:].all())
        self.assertIn("Unknown physical", info["phase_calibration"])

    def test_reject_unverified_palette_or_rgb_copy(self):
        image = indexed_image()
        with self.assertRaises(ValueError):
            decode_display_palette(image.convert("RGB"))
        palette = image.getpalette()
        palette[16 * 3] = 255
        image.putpalette(palette)
        with self.assertRaises(ValueError):
            decode_display_palette(image)

    def test_quantization_handles_wrapped_angles(self):
        step = 2 * np.pi / 16
        values = np.array([-0.1, 0.1, 2 * np.pi + 0.1])
        np.testing.assert_allclose(quantized_phase(values, 16), [15 * step, 0, 0], atol=1e-6)
        for bins in [-1, 1, 257, 2.5, True]:
            with self.assertRaises(ValueError):
                quantized_phase(values, bins)

    def test_profile_is_reproducible_and_does_not_modify_base(self):
        base = default_request()
        before = default_request()
        sample = draw_sample(base, 5, "single", "sparse_mine")
        self.assertEqual(base, before)
        self.assertEqual(sample, draw_sample(base, 5, "single", "sparse_mine"))
        self.assertEqual(draw_sample(base, 5, "single"), draw_sample(base, 5, "single", "standard"))
        self.assertNotEqual(sample, draw_sample(base, 5, "single", "standard"))
        self.assertEqual(sample["faces"][1]["enabled"], False)


if __name__ == "__main__":
    unittest.main()
