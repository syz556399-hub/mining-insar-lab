"""Display-code adapters, distinct from calibrated interferometric phase recovery."""

import colorsys

import numpy as np


def quantized_phase(phase, bins=0):
    if type(bins) is not int or (bins != 0 and not 2 <= bins <= 256):
        raise ValueError("phase bins must be 0 or an integer in 2..256")
    if not bins:
        return np.asarray(phase, dtype=np.float32)
    step = 2 * np.pi / bins
    return (np.floor(np.mod(phase, 2 * np.pi) / step) * step).astype(np.float32)


def decode_display_palette(image):
    """Accept a verified 16-brightness × 16-cyclic-color table only.

    The low nibble is a DISPLAY color code, not recovered physical phase.
    Phase origin/sign, original scale and interpolation remain uncalibrated.
    """
    if image.mode != "P":
        raise ValueError("Palette-code input requires the original P-mode BMP, not an RGB copy")
    image.load()
    palette = image.getpalette()
    if palette is None or len(palette) != 768:
        raise ValueError("Expected a 256-entry RGB palette")
    table = np.asarray(palette, dtype=np.float64).reshape(16, 16, 3)
    expected = np.arange(16)[:, None, None] / 15 * table[-1]
    if np.max(np.abs(table - expected)) > 2:
        raise ValueError("Palette does not separate 16 brightness levels and 16 color codes")
    hsv = np.array([colorsys.rgb_to_hsv(*(color / 255)) for color in table[-1]])
    differences = (np.roll(hsv[:, 0], -1) - hsv[:, 0] + 0.5) % 1 - 0.5
    if (
        hsv[:, 1].min() < 0.1
        or not (np.all(differences > 0) or np.all(differences < 0))
        or not np.isclose(abs(differences.sum()), 1, atol=0.05)
    ):
        raise ValueError("Palette color order is not a consistently directed color cycle")
    indices = np.asarray(image, dtype=np.uint8)
    display_angle = ((indices % 16).astype(np.float32) * (2 * np.pi / 16)).astype(np.float32)
    visible = indices // 16 > 0
    return (
        display_angle,
        visible,
        dict(
            encoding="16 ordered cyclic display colors × 16 brightness levels",
            phase_calibration="Unknown physical phase origin, sign and scale; display-code proxy only",
            validity="Brightness level zero has no visible color; this is not water/coherence validity",
        ),
    )
