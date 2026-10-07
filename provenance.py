"""Versioned, portable identifiers and numerical data descriptions."""

import hashlib
import json
import platform

import numpy as np
from PIL import __version__ as pillow_version

VERSION = "2.1.0"
ENGINE = "working-face-integral-2.1"
SCHEMA_VERSION = "1.0"


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def group_identity(request):
    """Identify underlying geometry/time regardless of observation noise or label threshold."""
    keys = ("size", "extent_m", "day_before", "day_after", "segments")
    return fingerprint(
        {"faces": request["faces"], "settings": {k: request["settings"][k] for k in keys}}
    )


def runtime_versions():
    return {
        "software": VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": pillow_version,
    }


def array_schema(arrays):
    output = {}
    for key, value in arrays.items():
        if key.endswith("_mask") or key == "mask":
            unit = "1"
            meaning = "binary 0/1 mask"
        elif key.endswith("_rad"):
            unit = "rad"
            meaning = "phase under the documented interferometric sign convention"
        elif key.endswith("_m"):
            unit = "m"
            meaning = "length; see MODEL.md for coordinate and component definitions"
        else:
            unit = "1"
            meaning = "dimensionless normalized quantity"
        output[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "unit": unit,
            "description": meaning,
        }
    return output
