"""Experimental task support, kept separate from physical subsidence truth."""

import numpy as np


def fringe_support(phase, threshold=np.pi / 2):
    """Clean deformation phase magnitude support, NOT observed fringe visibility.

    pi/2 is an explicit quarter-cycle design convention, not a calibrated
    boundary of a real mining area. Noise, terrain and observed colors never
    enter this label. Internal low-quality regions are not automatically erased.
    """
    phase = np.asarray(phase)
    if phase.ndim != 2 or not np.isfinite(phase).all():
        raise ValueError("Expected finite 2D clean deformation phase")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Require a positive finite support threshold")
    return (np.abs(phase) >= threshold).astype(np.uint8)
