"""Independent simulator reboot: physical source, statistical observation, review labels.

Uses only this project's own PIM primitives. No third-party simulator or weights.
Maps are statistical experiments, not focused or geocoded SAR acquisitions.
"""

import colorsys
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from mine_model import Face, Settings, grid, panel_epochs, strip_integral
from provenance import array_schema, fingerprint, runtime_versions
from scene_environment import rasterize_water, water_layout

ENGINE = "mine-simulator-reboot-3.2"
LABEL_RULE = "abs(clean, spatially averaged deformation phase) >= reviewed threshold_rad"


def block_mean(values, factor):
    h, w = values.shape
    if h % factor or w % factor:
        raise ValueError("The spatial averaging factor must divide both dimensions")
    return values.reshape(h // factor, factor, w // factor, factor).mean((1, 3))


def spectral_field(size, rng, scale_px, roughness=2.5, angle=0, aspect=1):
    """Own finite-band spectral construction; angle in radians, scale in pixels."""
    fy, fx = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij")
    along = fx * np.cos(angle) + fy * np.sin(angle)
    across = -fx * np.sin(angle) + fy * np.cos(angle)
    radius = np.sqrt((along * aspect) ** 2 + across**2)
    filt = (radius**2 + (1 / scale_px) ** 2) ** (-roughness / 4)
    spectrum = np.fft.fft2(rng.normal(size=(size, size))) * filt
    spectrum[0, 0] = 0
    field = np.fft.ifft2(spectrum).real
    return (field - field.mean()) / max(float(field.std()), 1e-12)


def warp(values, dy, dx):
    """Bilinear coordinate perturbation, explicitly an optional shape augmentation."""
    y, x = np.indices(values.shape, dtype=float)
    yy = np.clip(y + dy, 0, values.shape[0] - 1)
    xx = np.clip(x + dx, 0, values.shape[1] - 1)
    y0, x0 = yy.astype(int), xx.astype(int)
    y1, x1 = np.minimum(y0 + 1, values.shape[0] - 1), np.minimum(x0 + 1, values.shape[1] - 1)
    ay, ax = yy - y0, xx - x0
    return (
        values[y0, x0] * (1 - ay) * (1 - ax)
        + values[y1, x0] * ay * (1 - ax)
        + values[y0, x1] * (1 - ay) * ax
        + values[y1, x1] * ay * ax
    )


def candidate_mask(phase, threshold):
    if not np.isfinite(threshold) or not 0.02 <= threshold <= 10:
        raise ValueError("Review threshold must be 0.02–10 rad")
    phase = np.asarray(phase)
    if phase.ndim != 2 or not np.isfinite(phase).all():
        raise ValueError("Expected a finite 2D deformation phase")
    return (np.abs(phase) >= threshold).astype(np.uint8)


def correlated_pair(reference, gamma, power, rng, looks, spatial_factor):
    """Unit-variance circular complex Gaussian pairs with E[z1 conj(z2)] = gamma e^iφ.

    Spatial averaging operates on cross products and powers, NEVER wrapped phases.
    Independent ensembles and map-grid cells do not simulate SAR focusing geometry.
    """
    cross = np.zeros(reference.shape, dtype=np.complex128)
    p1, p2 = np.zeros_like(reference), np.zeros_like(reference)
    for _ in range(looks):
        z1 = (rng.normal(size=reference.shape) + 1j * rng.normal(size=reference.shape)) / np.sqrt(2)
        z0 = (rng.normal(size=reference.shape) + 1j * rng.normal(size=reference.shape)) / np.sqrt(2)
        first = np.sqrt(power) * z1
        second = (
            np.sqrt(power)
            * (gamma * z1 + np.sqrt(np.clip(1 - gamma**2, 0, 1)) * z0)
            * np.exp(-1j * reference)
        )
        cross += first * np.conj(second)
        p1 += np.abs(first) ** 2
        p2 += np.abs(second) ** 2
    cross = block_mean(cross / looks, spatial_factor)
    p1, p2 = block_mean(p1 / looks, spatial_factor), block_mean(p2 / looks, spatial_factor)
    estimated = np.clip(np.abs(cross) / np.sqrt(np.maximum(p1 * p2, 1e-20)), 0, 1)
    return cross, p1, p2, estimated


def fit_display_profile(source):
    """Read palette and a brightness histogram only; no patch reuse or phase recovery."""
    source = Path(source)
    with Image.open(source) as im:
        if im.mode != "P":
            raise ValueError("This display adapter requires an indexed BMP")
        idx = np.asarray(im)
        palette = np.array(im.getpalette()[:768]).reshape(16, 16, 3)
        full = palette[-1].astype(float)
        predicted = np.arange(16)[:, None, None] / 15 * full[None, :, :]
        if np.max(np.abs(predicted - palette)) > 3:
            raise ValueError("The palette is not a 16 brightness × 16 color lookup")
        histogram = np.bincount((idx // 16).ravel(), minlength=16)
        histogram[0] = 0  # Black display pixels are not interpreted as water or SAR invalidity.
        weights = histogram / histogram.sum()
        result = {
            "name": "user-gamma-bmp-display-v1",
            "source_name": source.name,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "size": list(im.size),
            "color_table": full.astype(int).tolist(),
            "brightness_weights": weights.tolist(),
            "phase_mapping": "cyclic display proxy only; physical phase zero/sign/scale unknown",
            "scope": "January development reference only; statistics, no image patches",
        }
    return result


def defaults():
    return {
        "size": 256,
        "extent_m": 2200,
        "seed": 20261009,
        "case": "overlap",
        "footprint": "rectangular",
        "style": "bmp",
        "cycles": 4.0,
        "background_cycles": 3.0,
        "background_angle_deg": 0.0,
        "atmosphere_mm": 3.0,
        "coherence": 0.85,
        "looks": 2,
        "spatial_factor": 2,
        "terrain_relief_m": 120.0,
        "dem_error_m": 5.0,
        "baseline_m": 100.0,
        "water": False,
        "shape_warp": 0.012,
        "target_radius_fraction": 0.08,
        "boundary_rad": 0.8,
    }


CASES = ("single", "overlap", "separated", "edge", "small", "negative")
CASE_NAMES = {
    "single": "单盆地",
    "overlap": "双盆地重叠",
    "separated": "双盆地分离",
    "edge": "边缘截断",
    "small": "小范围沉降",
    "negative": "只有背景",
}


def validate(config):
    if not isinstance(config, dict) or set(config) - set(defaults()):
        raise ValueError("Unknown simulator parameter")
    c = {**defaults(), **config}
    # Saved 3.0 requests predate the footprint selector. Preserve their old source.
    if "footprint" not in config and set(defaults()) - {"footprint"} <= set(config):
        c["footprint"] = "rectangular"
    if c["case"] not in CASES or c["style"] not in ("bmp", "research"):
        raise ValueError("Unknown scene or display style")
    if c["footprint"] not in ("gentle", "irregular", "rectangular"):
        raise ValueError("Unknown mined footprint")
    for key, options in {
        "size": (128, 256),
        "spatial_factor": (1, 2),
        "looks": (1, 2, 4, 8),
    }.items():
        if type(c[key]) is not int or c[key] not in options:
            raise ValueError(f"Invalid {key}")
    if type(c["seed"]) is not int or not 0 <= c["seed"] < 2**32 or type(c["water"]) is not bool:
        raise ValueError("Invalid seed or water switch")
    ranges = {
        "extent_m": (800, 6000),
        "cycles": (0.5, 10),
        "background_cycles": (0, 12),
        "background_angle_deg": (-180, 180),
        "atmosphere_mm": (0, 15),
        "coherence": (0.2, 1),
        "terrain_relief_m": (0, 600),
        "dem_error_m": (0, 40),
        "baseline_m": (-500, 500),
        "shape_warp": (0, 0.04),
        "target_radius_fraction": (0.03, 0.2),
        "boundary_rad": (0.02, 10),
    }
    for key, (low, high) in ranges.items():
        value = c[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Invalid {key}")
        if not low <= value <= high:
            raise ValueError(f"{key} must be {low}–{high}")
    return c


def geometry(c):
    rng = np.random.default_rng(np.random.SeedSequence([c["seed"], 311]))
    extent = c["extent_m"]
    radius = c["target_radius_fraction"] * extent
    if c["case"] == "small":
        radius *= 0.6
    center = rng.uniform(-0.16, 0.16, 2) * extent
    if c["case"] == "edge":
        center[0] = 0.44 * extent
    theta = rng.uniform(-np.pi, np.pi)
    gap = {"overlap": 1.2, "separated": 4.5}.get(c["case"], 0) * radius
    irregular = c["footprint"] != "rectangular"
    gentle = c["footprint"] == "gentle"
    offset_theta = theta + np.pi / 2 if irregular else theta
    faces = []
    for i in range(2 if c["case"] in ("overlap", "separated") else 1):
        offset = (i - 0.5) * gap
        faces.append(
            Face(
                enabled=c["case"] != "negative",
                east_m=float(center[0] + offset * np.cos(offset_theta)),
                north_m=float(center[1] + offset * np.sin(offset_theta)),
                length_m=float(
                    radius * rng.uniform(3.2, 4.4)
                    if gentle
                    else radius * rng.uniform(4.8, 6.2)
                    if irregular
                    else radius * rng.uniform(2.0, 3.0)
                ),
                width_m=float(
                    radius * rng.uniform(1.2, 1.8) if gentle else radius * rng.uniform(0.8, 1.3)
                ),
                bearing_deg=float(
                    np.rad2deg(theta) + rng.uniform(-10, 10)
                    if gentle
                    else np.rad2deg(theta) + rng.uniform(-18, 18)
                    if irregular
                    else np.rad2deg(theta) + rng.uniform(-35, 35)
                ),
                depth_m=float(
                    max(
                        50,
                        radius * rng.uniform(1.2, 1.6)
                        if gentle
                        else radius * rng.uniform(0.9, 1.3)
                        if irregular
                        else radius * rng.uniform(1.8, 2.3),
                    )
                ),
                thickness_m=float(rng.uniform(1.5, 3.5)),
                subsidence_factor=float(rng.uniform(0.5, 0.8)),
                influence_tangent=float(rng.uniform(1.8, 2.5)),
                advance_m_day=float(
                    max(2, radius * (0.11 if gentle else 0.14 if irregular else 0.08))
                ),
                start_day=float(rng.uniform(0, 12)),
                response_days=float(rng.uniform(50, 100)),
            )
        )
    return faces


def mined_strips(face, seed, source_index, segments=48, mode="irregular"):
    """Own irregular domain: non-overlapping along strips, varying cross bounds.

    Piecewise-linear width/center knots describe an aggregate mined area, not a
    surveyed mine plan. The exact integration domains are saved for reproduction.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, 315, source_index]))
    knots = np.linspace(-face.length_m / 2, face.length_m / 2, 9)
    profile = {}
    if mode == "gentle":
        # A single broad bend and shallow waist replace independent node jitter.
        # Bounds are design choices informed by development labels, not mine-plan inversion.
        t = np.linspace(-1, 1, len(knots))
        parameters = {
            "center_tilt": float(rng.uniform(-0.10, 0.10)),
            "center_bend": float(rng.uniform(-0.15, 0.15)),
            "width_taper": float(rng.uniform(-0.12, 0.12)),
            "width_bulge": float(rng.uniform(0.02, 0.18)),
            "waist_depth": float(rng.uniform(0.02, 0.12)),
            "waist_position": float(rng.uniform(-0.25, 0.25)),
        }
        center = face.width_m * (
            parameters["center_tilt"] * t + parameters["center_bend"] * (1 - t**2)
        )
        width = face.width_m * (
            1
            + parameters["width_taper"] * t
            + parameters["width_bulge"] * (1 - t**2)
            - parameters["waist_depth"]
            * np.exp(-(((t - parameters["waist_position"]) / 0.45) ** 2))
        )
        profile = {
            "profile_family": "single broad bend, slow taper, shallow waist",
            "parameters": parameters,
        }
    elif mode == "irregular":
        width = face.width_m * rng.uniform(0.45, 1.7, len(knots))
        waist = int(rng.integers(2, len(knots) - 2))
        width[waist] = face.width_m * rng.uniform(0.25, 0.4)
        center = face.width_m * rng.uniform(-0.7, 0.7, len(knots))
    else:
        raise ValueError("Unknown strip profile mode")
    edges = np.linspace(knots[0], knots[-1], segments + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    widths, centers = np.interp(mids, knots, width), np.interp(mids, knots, center)
    strips = [
        {
            "along_lower_m": float(low),
            "along_upper_m": float(high),
            "across_lower_m": float(mid - w / 2),
            "across_upper_m": float(mid + w / 2),
        }
        for low, high, mid, w in zip(edges[:-1], edges[1:], centers, widths)
    ]
    return strips, {
        **profile,
        "along_knots_m": knots.tolist(),
        "center_knots_m": center.tolist(),
        "width_knots_m": width.tolist(),
        "strips": strips,
    }


def footprint_epochs(face, size, extent_m, strips, before, after):
    """Integrate the existing Gaussian influence kernel over our strip domains.

    Integrals are analytic within each rectangle; domain and mining time are
    discrete approximations. Along intervals must not overlap (no double counting).
    """
    east, north = grid(Settings(size=size, extent_m=extent_m))
    theta = np.deg2rad(face.bearing_deg)
    along = (east - face.east_m) * np.cos(theta) + (north - face.north_m) * np.sin(theta)
    across = -(east - face.east_m) * np.sin(theta) + (north - face.north_m) * np.cos(theta)
    epoch0, epoch1 = np.zeros_like(east), np.zeros_like(east)
    radius = face.depth_m / face.influence_tangent
    previous_upper = -np.inf
    for item in strips:
        low, high = item["along_lower_m"], item["along_upper_m"]
        left, right = item["across_lower_m"], item["across_upper_m"]
        if (
            not np.isfinite([low, high, left, right]).all()
            or low >= high
            or left >= right
            or low < previous_upper
        ):
            raise ValueError(
                "Mined strips must be finite, positive and non-overlapping along advance"
            )
        previous_upper = high
        if not face.enabled:
            continue
        extraction_day = (
            face.start_day + ((low + high) / 2 + face.length_m / 2) / face.advance_m_day
        )
        age0, age1 = max(0, before - extraction_day), max(0, after - extraction_day)
        if age1 == 0:
            continue
        spatial = strip_integral(along, low, high, radius) * strip_integral(
            across, left, right, radius
        )
        epoch0 += spatial * (-np.expm1(-age0 / face.response_days))
        epoch1 += spatial * (-np.expm1(-age1 / face.response_days))
    scale = face.thickness_m * face.subsidence_factor
    return epoch0 * scale, epoch1 * scale


def simulate(config):
    c = validate(config)
    n = c["size"] * c["spatial_factor"]
    s = Settings(
        size=n,
        extent_m=c["extent_m"],
        seed=c["seed"],
        water_enabled=c["water"],
        water_mode="ponds",
        river_width_m=45,
    )
    east, north = grid(s)
    faces = geometry(c)
    source_rng = np.random.default_rng(np.random.SeedSequence([c["seed"], 312]))
    dx = c["shape_warp"] * n * spectral_field(n, source_rng, n / 5, roughness=5)
    dy = c["shape_warp"] * n * spectral_field(n, source_rng, n / 5, roughness=5)
    components = []
    footprints = []
    for i, f in enumerate(faces):
        if c["footprint"] != "rectangular":
            strips, footprint = mined_strips(f, c["seed"], i, mode=c["footprint"])
            before, after = footprint_epochs(f, n, c["extent_m"], strips, 70, 82)
            footprints.append(footprint)
        else:
            before, after = panel_epochs(f, n, c["extent_m"], 32, 70, 82)
        components.append(warp(after - before, dy, dx))
    down = sum(components, start=np.zeros((n, n)))
    dn, de = np.gradient(down, -c["extent_m"] / n, c["extent_m"] / n)
    # One explicit effective horizontal-motion coefficient for the aggregate field.
    radius = float(np.mean([f.depth_m / f.influence_tangent for f in faces]))
    east_motion, north_motion = 0.15 * radius * de, 0.15 * radius * dn
    inc = np.deg2rad(33)
    los = down * np.cos(inc) - east_motion * np.sin(inc)
    clean = -4 * np.pi * los / 0.055
    # Declared amplitude augmentation scales ALL displacement components consistently.
    base_cycles = float(np.max(np.abs(clean)) / (2 * np.pi))
    scale = c["cycles"] / base_cycles if base_cycles > 1e-10 else 1.0
    down, los, clean = down * scale, los * scale, clean * scale
    east_motion, north_motion = east_motion * scale, north_motion * scale
    components = [v * scale for v in components]
    bg_rng = np.random.default_rng(np.random.SeedSequence([c["seed"], 313]))
    terrain = spectral_field(n, bg_rng, n / 5, roughness=3.5, angle=0.4, aspect=2)
    dem = (
        300 + (terrain - terrain.min()) / max(float(np.ptp(terrain)), 1e-12) * c["terrain_relief_m"]
    )
    dem_error = c["dem_error_m"] * spectral_field(n, bg_rng, n / 8)
    atmosphere = c["atmosphere_mm"] / 1000 * spectral_field(n, bg_rng, n / 3, roughness=4)
    angle = np.deg2rad(c["background_angle_deg"])
    ramp = (
        2
        * np.pi
        * c["background_cycles"]
        * (east * np.cos(angle) + north * np.sin(angle))
        / c["extent_m"]
    )
    residual = -4 * np.pi * c["baseline_m"] * dem_error / (0.055 * 850000 * np.sin(inc))
    background = ramp - 4 * np.pi * atmosphere / 0.055 + residual
    reference = clean + background
    water_scene = water_layout(s)
    water = rasterize_water(water_scene, east, north, c["extent_m"])
    obs_rng = np.random.default_rng(np.random.SeedSequence([c["seed"], 314]))
    texture = spectral_field(n, obs_rng, n / 12, roughness=2.2, angle=1.0, aspect=2.5)
    fine = spectral_field(n, obs_rng, 4, roughness=1.0)
    gy, gx = np.gradient(dem)
    ridges = np.hypot(gx, gy)
    ridge = ridges / max(float(ridges.std()), 1e-12)
    log_power = 0.55 * texture + 0.6 * fine + 0.28 * np.clip(ridge - ridge.mean(), -3, 3)
    power = np.exp(np.clip(log_power, -3.5, 3.5))
    power[water] *= 0.05
    # Spatially varying coherence and extra loss where the local phase is undersampled.
    variation = spectral_field(n, obs_rng, n / 10, roughness=3)
    gamma = np.clip(c["coherence"] - 0.07 * variation, 0.1, 0.995)
    py, px = np.gradient(reference)
    gamma *= np.exp(-0.045 * np.hypot(px, py))
    gamma[water] = 0.025
    cross, p1, p2, estimated = correlated_pair(
        reference, gamma, power, obs_rng, c["looks"], c["spatial_factor"]
    )
    factor = c["spatial_factor"]
    arrays = {
        name: block_mean(value, factor).astype(np.float32)
        for name, value in {
            "delta_down_m": down,
            "delta_los_m": los,
            "delta_east_m": east_motion,
            "delta_north_m": north_motion,
            "deformation_phase_rad": clean,
            "background_phase_rad": background,
            "reference_phase_rad": reference,
            "dem_m": dem,
            "dem_error_m": dem_error,
            "atmosphere_los_m": atmosphere,
            "residual_topographic_phase_rad": residual,
            "model_coherence": gamma,
            "backscatter_power": power,
        }.items()
    }
    arrays.update(
        wrapped_phase_rad=np.angle(cross).astype(np.float32),
        observation_real=cross.real.astype(np.float32),
        observation_imag=cross.imag.astype(np.float32),
        intensity_before=p1.astype(np.float32),
        intensity_after=p2.astype(np.float32),
        estimated_coherence=estimated.astype(np.float32),
        water_fraction=block_mean(water.astype(float), factor).astype(np.float32),
    )
    arrays["water_mask"] = (arrays["water_fraction"] >= 0.5).astype(np.uint8)
    arrays["candidate_mask"] = candidate_mask(arrays["deformation_phase_rad"], c["boundary_rad"])
    for i, component in enumerate(components):
        arrays[f"source_{i + 1}_down_m"] = block_mean(component, factor).astype(np.float32)
    max_step = max(float(np.max(np.abs(np.diff(reference, axis=a)))) for a in (0, 1))
    metadata = {
        "engine": ENGINE,
        "runtime_versions": runtime_versions(),
        "random_generator": "NumPy default_rng; "
        + np.random.default_rng(0).bit_generator.__class__.__name__,
        "config": c,
        "faces": [asdict(f) for f in faces],
        "mined_footprints": footprints,
        "epochs_days": [70, 82],
        "wavelength_m": 0.055,
        "incidence_deg": 33,
        "radar_azimuth_deg": 90,
        "coordinate_sign": "East, North, Down; LOS range increase positive; phase = -4pi LOS/lambda",
        "amplitude_scale": scale,
        "source_model": "own Gaussian influence integrals over non-overlapping advancing mined strips + exponential time response",
        "footprint_model": c["footprint"]
        + "; aggregate mined domain, not a calibrated or surveyed mine plan",
        "shape_model": "optional bilinear coordinate warp; augmentation, not a calibrated geological model",
        "observation_model": "own correlated complex Gaussian pairs; complex/power block averages on synthetic map grid",
        "display_phase_mapping": "display proxy only; real GAMMA phase zero/sign/scale unknown",
        "label_rule": LABEL_RULE,
        "label_status": "candidate; human review pending",
        "label_limitation": "Gaussian tails have no unique visible outer edge; this contour needs human confirmation/correction. No automatic hole filling, water erasure or size filtering.",
        "group_id": fingerprint(
            {
                "faces": [asdict(f) for f in faces],
                "size": c["size"],
                "extent": c["extent_m"],
                "cycles": c["cycles"],
                "shape_warp": c["shape_warp"],
                "source_seed": c["seed"],
                **(
                    {"footprint": c["footprint"], "mined_footprints": footprints}
                    if footprints
                    else {}
                ),
            }
        ),
        "request_sha256": fingerprint(c),
        "pixel_spacing_m": c["extent_m"] / c["size"],
        "source_cycles": float(np.max(np.abs(arrays["deformation_phase_rad"])) / (2 * np.pi)),
        "peak_down_mm": float(arrays["delta_down_m"].max() * 1000),
        "label_percent": float(arrays["candidate_mask"].mean() * 100),
        "water_percent": float(arrays["water_mask"].mean() * 100),
        "mean_model_coherence": float(arrays["model_coherence"].mean()),
        "max_fine_grid_phase_step_rad": max_step,
        "water_scene": water_scene,
        "array_schema": array_schema(arrays),
        "warnings": ["Fine-grid phase gradient exceeds pi; spatial aliasing is possible"]
        if max_step > np.pi
        else [],
        "scope": "synthetic data supplement; neither real-mine calibration nor arbitrary-image recognition validation",
    }
    return arrays, metadata


def research_rgb(phase):
    # Our own explicit jet-like piecewise RGB ramp, for comparison with paper displays.
    t = np.mod(phase, 2 * np.pi) / (2 * np.pi)
    colors = np.stack([np.clip(1.5 - np.abs(4 * t - p), 0, 1) for p in (3, 2, 1)], -1)
    return np.round(colors * 255).astype(np.uint8)


def display_brightness(arrays, profile=None):
    if profile is None:
        weights = np.exp(-(((np.arange(16) - 8) / 3) ** 2))
        weights[0] = 0
        weights /= weights.sum()
    else:
        weights = np.array(profile["brightness_weights"])
    intensity = np.sqrt(arrays["intensity_before"] * arrays["intensity_after"])
    # Empirical display transport: intensity ranks -> development-reference brightness CDF.
    # This is a display-domain adapter, NOT a physically calibrated radiometric response.
    order = np.argsort(intensity.ravel(), kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = (np.arange(order.size) + 0.5) / order.size
    brightness = np.searchsorted(np.cumsum(weights), ranks.reshape(intensity.shape), side="left")
    brightness = np.minimum(brightness, 15)
    gain = brightness.astype(float) / 15
    # Low reflectivity water remains dim, but non-water dim pixels stay in the image.
    gain *= np.where(arrays["water_fraction"] > 0.5, 0.2, 1)
    return gain


def bmp_rgb(arrays, profile=None):
    # float32 remainder can round a tiny negative phase up to exactly 2*pi.
    # Quantize in float64 and retain the cyclic endpoint without changing phase data.
    phase = np.asarray(arrays["wrapped_phase_rad"], dtype=np.float64)
    phase_bin = np.floor(np.mod(phase, 2 * np.pi) / (2 * np.pi) * 16).astype(int) % 16
    colors = (
        np.array([colorsys.hsv_to_rgb(i / 16, 0.65, 1) for i in range(16)]) * 255
        if profile is None
        else np.array(profile["color_table"])
    )
    rgb = colors[phase_bin] * display_brightness(arrays, profile)[..., None]
    return np.round(rgb).astype(np.uint8)


def views(arrays, profile=None):
    return {
        "bmp": bmp_rgb(arrays, profile),
        "research": np.round(
            research_rgb(arrays["wrapped_phase_rad"])
            * display_brightness(arrays, profile)[..., None]
        ).astype(np.uint8),
    }


def overlay(image, mask):
    mask = np.asarray(mask, dtype=bool)
    p = np.pad(mask, 1)
    edge = mask & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    halo = edge.copy()
    halo[1:] |= edge[:-1]
    halo[:-1] |= edge[1:]
    halo[:, 1:] |= edge[:, :-1]
    halo[:, :-1] |= edge[:, 1:]
    result = image.copy()
    result[halo] = [0, 0, 0]
    result[edge] = [255, 255, 255]
    return result


def batch_configs(seed=20261009):
    """20-scene review scaffold, not an estimated real-mine population distribution."""
    cases = [
        "single",
        "overlap",
        "separated",
        "small",
        "edge",
        "single",
        "overlap",
        "small",
        "single",
        "negative",
    ]
    output = []
    for i in range(20):
        c = defaults()
        c.update(
            seed=seed + i * 7919,
            case=cases[i % 10],
            style="bmp",
            cycles=[2.2, 4, 3, 2, 3, 5, 5.5, 3, 1.5, 2][i % 10],
            extent_m=[2800, 2200, 3200, 4200, 2000, 1800, 2400, 3500, 3000, 2600][i % 10],
            target_radius_fraction=[
                0.065,
                0.085,
                0.055,
                0.055,
                0.09,
                0.12,
                0.095,
                0.04,
                0.07,
                0.08,
            ][i % 10],
            background_cycles=[2.5, 3, 4, 2, 3, 1.5, 4.5, 3, 2, 4][i % 10],
            background_angle_deg=[10, -15, 50, 80, 30, -35, 0, 65, -60, 15][i % 10],
            atmosphere_mm=[3, 2, 5, 3, 4, 2, 3.5, 5, 4, 6][i % 10],
            coherence=[0.88, 0.90, 0.85, 0.86, 0.85, 0.92, 0.88, 0.83, 0.78, 0.65][i % 10],
            water=i in (4, 8, 14, 18),
            shape_warp=0.012 if i % 3 or i in (12, 15, 18) else 0,
            boundary_rad=0.8,
        )
        if i == 17:
            # Replacement for the explicitly rejected near-invisible S18 example.
            c.update(case="single", target_radius_fraction=0.065)
        output.append(validate(c))
    return output


def save_scene(folder, arrays, metadata, profile):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder / "arrays.npz", **arrays)
    images = views(arrays, profile)
    for style, image in images.items():
        Image.fromarray(image).save(folder / f"{style}.png")
        Image.fromarray(overlay(image, arrays["candidate_mask"])).save(
            folder / f"{style}_overlay.png"
        )
    Image.fromarray(arrays["candidate_mask"] * 255).save(folder / "mask.png")
    for name in ("estimated_coherence", "dem_m", "delta_down_m"):
        v = arrays[name]
        if name == "estimated_coherence":
            color = np.round(v * 255).astype(np.uint8)
        else:
            t = (v - v.min()) / max(float(np.ptp(v)), 1e-12)
            color = research_rgb(t * 1.4 * np.pi + 0.5 * np.pi)
        Image.fromarray(color).save(folder / f"{name}.png")
    metadata["display_profile_sha256"] = fingerprint(profile) if profile else None
    metadata["display_engine"] = "palette-or-research-colors; intensity-CDF brightness-v2"
    (folder / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
