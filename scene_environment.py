"""独立实现的合成地形和复数观测层；地图格网上的统计近似，不是 SAR 聚焦器。"""

import numpy as np


def smooth_field(size, rng, exponent=3):
    noise = rng.normal(size=(size, size))
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    spectrum = np.fft.fft2(noise) * (radius * radius + (2 / size) ** 2) ** (-exponent / 4)
    spectrum[0, 0] = 0
    result = np.fft.ifft2(spectrum).real
    return (result - result.mean()) / max(result.std(), 1e-12)


def environment(s, east, north):
    rng = np.random.default_rng(np.random.SeedSequence([s.seed, 2301]))
    water_rng = np.random.default_rng(np.random.SeedSequence([s.seed, 2302]))
    error_rng = np.random.default_rng(np.random.SeedSequence([s.seed, 2303]))
    relief = smooth_field(s.size, rng)
    dem = (
        300 + (relief - relief.min()) / max(np.ptp(relief), 1e-12) * s.relief_m
        if s.terrain_enabled
        else np.full(east.shape, 300.0)
    )
    water = np.zeros(east.shape, dtype=bool)
    if s.water_enabled:
        # 可控几何河流/湖泊，不声称水文流路推导。
        center = -0.28 * s.extent_m + 0.065 * s.extent_m * np.sin(
            2 * np.pi * north / s.extent_m + water_rng.uniform(-1, 1)
        )
        river = np.abs(east - center) < s.river_width_m / 2
        lake = ((east - 0.28 * s.extent_m) / (0.12 * s.extent_m)) ** 2 + (
            (north - 0.25 * s.extent_m) / (0.085 * s.extent_m)
        ) ** 2 < 1
        water = river | lake
    dem_error = (
        s.dem_error_m * smooth_field(s.size, error_rng, 2.5)
        if s.terrain_enabled
        else np.zeros_like(dem)
    )
    height = dem - dem.mean()
    coefficient = (
        0
        if abs(s.baseline_m) < 1e-12
        else -4
        * np.pi
        * s.baseline_m
        / (s.wavelength_m * s.slant_range_m * np.sin(np.deg2rad(s.incidence_deg)))
    )
    topographic = coefficient * height
    residual = coefficient * dem_error
    used = topographic if s.raw_topography else residual
    height_atmosphere = height / 100 * s.height_atmosphere_mm / 1000
    gamma = np.full(east.shape, s.land_coherence, dtype=float)
    # 时间失相干为独立指数近似；参考日期间隔以天计。
    gamma *= (
        np.exp(-(s.day_after - s.day_before) / s.decorrelation_days)
        if s.decorrelation_days > 0
        else 1
    )
    dn, de = np.gradient(dem, -s.extent_m / s.size, s.extent_m / s.size)
    gamma *= np.exp(-s.slope_coherence_loss * np.hypot(dn, de))
    gamma[water] = s.water_coherence
    amplitude = np.where(water, 0.15, 1.0)
    return {
        "dem_m": dem,
        "dem_error_m": dem_error,
        "water_mask": water.astype(np.uint8),
        "topographic_phase_rad": topographic,
        "residual_topographic_phase_rad": residual,
        "applied_topographic_phase_rad": used,
        "height_atmosphere_los_m": height_atmosphere,
        "model_coherence": np.clip(gamma, 0, 1),
        "backscatter_power": amplitude,
    }


def local_mean(array, width):
    """奇数正方形滑动窗口，同尺寸输出、反射边界。"""
    if width == 1:
        return array.copy()
    p = width // 2
    padded = np.pad(array, ((p, p), (p, p)), mode="reflect")
    table = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (
        table[width:, width:]
        - table[:-width, width:]
        - table[width:, :-width]
        + table[:-width, :-width]
    ) / (width * width)


def observe(reference, s, env, east, north):
    rng = np.random.default_rng(np.random.SeedSequence([s.seed, 2407]))
    gamma = env["model_coherence"].copy()
    if s.disturbed_patch:
        loss = 0.85 * np.exp(
            -((east - 0.2 * s.extent_m) ** 2 + (north + 0.1 * s.extent_m) ** 2)
            / (2 * (0.13 * s.extent_m) ** 2)
        )
        gamma *= 1 - loss
    cross = np.zeros(reference.shape, dtype=np.complex128)
    p1 = np.zeros(reference.shape)
    p2 = np.zeros(reference.shape)
    power = env["backscatter_power"]
    for _ in range(s.looks):
        z = (rng.normal(size=reference.shape) + 1j * rng.normal(size=reference.shape)) / np.sqrt(2)
        independent = (
            rng.normal(size=reference.shape) + 1j * rng.normal(size=reference.shape)
        ) / np.sqrt(2)
        first = np.sqrt(power) * z
        second = (
            np.sqrt(power)
            * (gamma * z + np.sqrt(np.maximum(0, 1 - gamma**2)) * independent)
            * np.exp(-1j * reference)
        )
        cross += first * np.conj(second)
        p1 += np.abs(first) ** 2
        p2 += np.abs(second) ** 2
    cross = local_mean(cross / s.looks, s.spatial_window)
    p1 = local_mean(p1 / s.looks, s.spatial_window)
    p2 = local_mean(p2 / s.looks, s.spatial_window)
    coherence = np.clip(np.abs(cross) / np.sqrt(np.maximum(p1 * p2, 1e-20)), 0, 1)
    return cross, gamma, coherence, p1, p2
