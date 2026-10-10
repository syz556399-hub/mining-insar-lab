"""工作面驱动的两期矿区模拟。全部单位为 m、day、rad，界面转换在调用层完成。

矩形开采单元的高斯影响函数积分属于已有概率积分思想。这里从该数学模型
自行实现几何、时间离散、位移投影与观测，不导入任何旧模拟器模块。
"""

import math
from dataclasses import asdict, dataclass, fields
from functools import lru_cache

import numpy as np

from provenance import (
    ENGINE,
    SCHEMA_VERSION,
    array_schema,
    fingerprint,
    group_identity,
    runtime_versions,
)
from scene_environment import WATER_MODES, environment, observe, water_layout
from task_labels import fringe_support


@dataclass(frozen=True)
class Face:
    enabled: bool = True
    east_m: float = -110
    north_m: float = 0
    length_m: float = 440
    width_m: float = 160
    bearing_deg: float = 20  # 推进方向：从正东向正北逆时针为正。
    depth_m: float = 240
    thickness_m: float = 2.2
    subsidence_factor: float = 0.65
    influence_tangent: float = 2
    advance_m_day: float = 7
    start_day: float = 0
    response_days: float = 80


@dataclass(frozen=True)
class Settings:
    size: int = 256
    extent_m: float = 1600
    seed: int = 20261007
    day_before: float = 40
    day_after: float = 52
    segments: int = 32
    wavelength_m: float = 0.055
    incidence_deg: float = 33
    radar_azimuth_deg: float = 90  # 地面点指向卫星的水平方位：正北=0、正东=90。
    horizontal_factor: float = 0.2
    target_threshold_mm: float = 5
    fringe_threshold_rad: float = float(np.pi / 2)
    atmosphere_mm: float = 1.5
    atmosphere_scale_m: float = 600
    orbit_cycles: float = 3
    phase_spread_rad: float = 0.45
    looks: int = 4
    disturbed_patch: bool = False
    complex_observation: bool = True
    terrain_enabled: bool = True
    water_enabled: bool = True
    water_mode: str = "random"
    raw_topography: bool = False
    relief_m: float = 180
    river_width_m: float = 65
    dem_error_m: float = 5
    baseline_m: float = 100
    slant_range_m: float = 850000
    height_atmosphere_mm: float = 1
    land_coherence: float = 0.92
    water_coherence: float = 0.03
    decorrelation_days: float = 180
    slope_coherence_loss: float = 0.3
    spatial_window: int = 3
    valid_coherence_threshold: float = 0.2


def default_request():
    second = Face(
        east_m=105,
        north_m=45,
        length_m=360,
        width_m=150,
        bearing_deg=-25,
        depth_m=210,
        start_day=8,
        advance_m_day=6,
    )
    return {"settings": asdict(Settings()), "faces": [asdict(Face()), asdict(second)]}


def _construct(cls, values):
    if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
        raise ValueError(f"{cls.__name__} 参数字段不正确")
    obj = cls(**values)
    for name, value in asdict(obj).items():
        if name in (
            "enabled",
            "disturbed_patch",
            "complex_observation",
            "terrain_enabled",
            "water_enabled",
            "raw_topography",
        ):
            if type(value) is not bool:
                raise ValueError(f"{name} 需要是开关")
        elif name == "water_mode":
            if not isinstance(value, str) or value not in WATER_MODES:
                raise ValueError("水体场景请选择 random、river、ponds 或 mixed")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} 必须是有限数字")
    return cls(
        **{
            f.name: float(getattr(obj, f.name)) if f.type is float else getattr(obj, f.name)
            for f in fields(cls)
        }
    )


def parse_request(request):
    if not isinstance(request, dict) or set(request) - {"settings", "faces"}:
        raise ValueError("需要 settings 和 faces 参数")
    settings = _construct(Settings, request.get("settings", {}))
    face_values = request.get("faces", default_request()["faces"])
    if not isinstance(face_values, list) or not 1 <= len(face_values) <= 2:
        raise ValueError("当前版本支持 1–2 个工作面")
    faces = [_construct(Face, item) for item in face_values]
    if type(settings.size) is not int or settings.size not in (128, 256, 512):
        raise ValueError("画幅请选择 128、256 或 512 像素")
    for name, low, high in [("seed", 0, 2**32 - 1), ("segments", 8, 128), ("looks", 1, 32)]:
        value = getattr(settings, name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} 需要是 {low}–{high} 的整数")
    bounds = {
        "extent_m": (400, 6000),
        "day_before": (-365, 2000),
        "day_after": (-365, 2000),
        "wavelength_m": (0.02, 0.25),
        "incidence_deg": (0, 80),
        "radar_azimuth_deg": (0, 360),
        "horizontal_factor": (0, 0.5),
        "target_threshold_mm": (0.1, 100),
        "fringe_threshold_rad": (0.01, 20),
        "atmosphere_mm": (0, 20),
        "atmosphere_scale_m": (50, 6000),
        "orbit_cycles": (-30, 30),
        "relief_m": (0, 1500),
        "river_width_m": (0, 500),
        "dem_error_m": (0, 100),
        "baseline_m": (-1000, 1000),
        "slant_range_m": (500000, 1500000),
        "height_atmosphere_mm": (-20, 20),
        "valid_coherence_threshold": (0, 1),
        "land_coherence": (0, 1),
        "water_coherence": (0, 1),
        "decorrelation_days": (0, 1000000),
        "slope_coherence_loss": (0, 5),
        "phase_spread_rad": (0, 3),
    }
    for name, (low, high) in bounds.items():
        if not low <= getattr(settings, name) <= high:
            raise ValueError(f"{name} 超出允许范围 {low}–{high}")
    if type(settings.spatial_window) is not int or settings.spatial_window not in (1, 3, 5, 7, 9):
        raise ValueError("空间平均窗口请选择 1、3、5、7、9")
    if settings.baseline_m != 0 and settings.incidence_deg < 1:
        raise ValueError("非零基线的地形计算需要入射角至少 1°；垂直观测请把基线设为 0")
    if settings.day_after < settings.day_before:
        raise ValueError("后期观测日期不能早于前期")
    face_bounds = {
        "east_m": (-6000, 6000),
        "north_m": (-6000, 6000),
        "length_m": (20, 2000),
        "width_m": (20, 1000),
        "bearing_deg": (-180, 180),
        "depth_m": (50, 1200),
        "thickness_m": (0.1, 8),
        "subsidence_factor": (0.05, 1),
        "influence_tangent": (0.5, 5),
        "advance_m_day": (0.2, 30),
        "start_day": (-365, 2000),
        "response_days": (1, 500),
    }
    for face in faces:
        for name, (low, high) in face_bounds.items():
            if not low <= getattr(face, name) <= high:
                raise ValueError(f"工作面 {name} 超出允许范围 {low}–{high}")
    return settings, faces


def request_dict(settings, faces):
    return {"settings": asdict(settings), "faces": [asdict(face) for face in faces]}


def wrap(phase):
    return np.angle(np.exp(1j * np.asarray(phase, dtype=np.float64))).astype(np.float32)


def grid(settings):
    pitch = settings.extent_m / settings.size
    centers = (np.arange(settings.size) + 0.5) * pitch - settings.extent_m / 2
    return np.meshgrid(centers, centers[::-1])  # 图像顶部为北。


def erf(values):
    # 使用标准库定义，不移植近似系数或第三方实现。
    return np.frompyfunc(math.erf, 1, 1)(values).astype(np.float64)


def strip_integral(coordinate, lower, upper, radius):
    """积分 exp[-pi*(x-s)^2/r²]/r ds，积分域为 [lower, upper]。"""
    scale = np.sqrt(np.pi) / radius
    return 0.5 * (erf((coordinate - lower) * scale) - erf((coordinate - upper) * scale))


@lru_cache(maxsize=6)
def panel_epochs(face, size, extent_m, segments, before, after):
    """沿推进方向离散为等长段，各段在开采到中心时开始一阶沉降响应。

    每段的空间积分是解析的，时间过程是分段近似；segments 控制收敛。
    缓存最多六个工作面结果，改变观测噪声不会反复计算几何。
    """
    east, north = grid(Settings(size=size, extent_m=extent_m))
    theta = np.deg2rad(face.bearing_deg)
    along = (east - face.east_m) * np.cos(theta) + (north - face.north_m) * np.sin(theta)
    across = -(east - face.east_m) * np.sin(theta) + (north - face.north_m) * np.cos(theta)
    radius = face.depth_m / face.influence_tangent
    side = strip_integral(across, -face.width_m / 2, face.width_m / 2, radius)
    edges = np.linspace(-face.length_m / 2, face.length_m / 2, segments + 1)
    epoch0, epoch1 = np.zeros_like(east), np.zeros_like(east)
    if face.enabled:
        for lower, upper in zip(edges[:-1], edges[1:]):
            extraction_day = (
                face.start_day + ((lower + upper) / 2 + face.length_m / 2) / face.advance_m_day
            )
            age0, age1 = max(0, before - extraction_day), max(0, after - extraction_day)
            if age1 == 0:
                continue
            spatial = strip_integral(along, lower, upper, radius) * side
            epoch0 += spatial * (-np.expm1(-age0 / face.response_days))
            epoch1 += spatial * (-np.expm1(-age1 / face.response_days))
        scale = face.thickness_m * face.subsidence_factor
        epoch0 *= scale
        epoch1 *= scale
    epoch0.setflags(write=False)
    epoch1.setflags(write=False)
    return epoch0, epoch1


def fourier_screen(settings, east, north):
    """独立随机平面波叠加；长度尺度可控，均值为零。"""
    rng = np.random.default_rng(np.random.SeedSequence([settings.seed, 701]))
    result = np.zeros_like(east)
    for _ in range(24):
        angle = rng.uniform(0, 2 * np.pi)
        wavelength = settings.atmosphere_scale_m * np.exp(rng.uniform(-0.7, 0.7))
        phase = rng.uniform(-np.pi, np.pi)
        result += np.cos(
            2 * np.pi * (east * np.cos(angle) + north * np.sin(angle)) / wavelength + phase
        )
    result -= result.mean()
    return result / max(result.std(), 1e-12) * settings.atmosphere_mm / 1000


def circular_observation(reference, settings, east, north):
    """经验圆周观测模型：von Mises 相位扰动，复数平均后取相位。

    phase_spread_rad 通过 kappa=1/spread² 映射到集中度，只在小噪声下近似标准差。
    looks 是独立重复观测数量；不声称实现了 SAR 空间多视或 SLC 成像。
    """
    rng = np.random.default_rng(np.random.SeedSequence([settings.seed, 907]))
    total = np.zeros(reference.shape, dtype=np.complex128)
    probability = np.zeros_like(reference)
    if settings.disturbed_patch:
        probability = 0.85 * np.exp(
            -((east - 0.2 * settings.extent_m) ** 2 + (north + 0.1 * settings.extent_m) ** 2)
            / (2 * (0.13 * settings.extent_m) ** 2)
        )
    for _ in range(settings.looks):
        if settings.phase_spread_rad == 0:
            jitter = np.zeros_like(reference)
        else:
            jitter = rng.vonmises(0, 1 / settings.phase_spread_rad**2, reference.shape)
        if settings.disturbed_patch:
            randomized = rng.random(reference.shape) < probability
            jitter = np.where(randomized, rng.uniform(-np.pi, np.pi, reference.shape), jitter)
        total += np.exp(1j * (reference + jitter))
    total /= settings.looks
    return total


def simulate(request):
    settings, faces = parse_request(request)
    east, north = grid(settings)
    before, after = np.zeros_like(east), np.zeros_like(east)
    east_motion, north_motion = np.zeros_like(east), np.zeros_like(east)
    components = []
    for face in faces:
        first, second = panel_epochs(
            face,
            settings.size,
            settings.extent_m,
            settings.segments,
            settings.day_before,
            settings.day_after,
        )
        before += first
        after += second
        change = second - first
        components.append(change)
        dn, de = np.gradient(
            change, -settings.extent_m / settings.size, settings.extent_m / settings.size
        )
        # 经验水平移动近似，方向指向沉降增量更大的区域；单位仍是米。
        east_motion += settings.horizontal_factor * face.depth_m / face.influence_tangent * de
        north_motion += settings.horizontal_factor * face.depth_m / face.influence_tangent * dn
    down = after - before
    inc, az = np.deg2rad(settings.incidence_deg), np.deg2rad(settings.radar_azimuth_deg)
    # 距离增加为正。水平运动朝向卫星会缩短距离。
    los = down * np.cos(inc) - (east_motion * np.sin(az) + north_motion * np.cos(az)) * np.sin(inc)
    phase = -4 * np.pi * los / settings.wavelength_m
    atmosphere = fourier_screen(settings, east, north)
    orbit = 2 * np.pi * settings.orbit_cycles * east / settings.extent_m
    env = environment(settings, east, north)
    atmosphere += env["height_atmosphere_los_m"]
    reference = (
        phase
        - 4 * np.pi * atmosphere / settings.wavelength_m
        + orbit
        + env["applied_topographic_phase_rad"]
    )
    if settings.complex_observation:
        observed, gamma, estimated, p1, p2 = observe(reference, settings, env, east, north)
    else:
        observed = circular_observation(reference, settings, east, north)
        gamma = env["model_coherence"]
        estimated = np.abs(observed)
        p1 = p2 = np.ones_like(phase)
    threshold = settings.target_threshold_mm / 1000
    mask = (down.astype(np.float32) >= threshold).astype(np.uint8)
    labels = [
        (component.astype(np.float32) >= threshold).astype(np.uint8) for component in components
    ]
    while len(components) < 2:
        components.append(np.zeros_like(down))
        labels.append(np.zeros_like(mask))
    arrays = {
        name: values.astype(np.float32)
        for name, values in {
            "subsidence_before_m": before,
            "subsidence_after_m": after,
            "delta_down_m": down,
            "delta_east_m": east_motion,
            "delta_north_m": north_motion,
            "delta_los_m": los,
            "deformation_phase_rad": phase,
            "reference_phase_rad": reference,
            "wrapped_phase_rad": np.angle(observed),
            "atmosphere_equivalent_los_m": atmosphere,
            "orbit_phase_rad": orbit,
            "observation_real": observed.real,
            "observation_imag": observed.imag,
            "face_1_delta_m": components[0],
            "face_2_delta_m": components[1],
        }.items()
    }
    arrays.update(
        {key: value.astype(np.float32) for key, value in env.items() if key != "water_mask"}
    )
    arrays.update(
        model_coherence=gamma.astype(np.float32),
        estimated_coherence=estimated.astype(np.float32),
        intensity_before=p1.astype(np.float32),
        intensity_after=p2.astype(np.float32),
    )
    valid = (
        (env["water_mask"] == 0) & (gamma.astype(np.float32) >= settings.valid_coherence_threshold)
    ).astype(np.uint8)
    arrays.update(
        mask=mask,
        fringe_mask=fringe_support(arrays["deformation_phase_rad"], settings.fringe_threshold_rad),
        fringe_valid_mask=np.ones_like(mask),
        face_1_mask=labels[0],
        face_2_mask=labels[1],
        water_mask=env["water_mask"],
        valid_mask=valid,
    )
    peak = int(np.argmax(down))
    warnings = []
    largest_step = max(float(np.abs(np.diff(reference, axis=axis)).max()) for axis in (0, 1))
    if largest_step > np.pi:
        warnings.append("相邻像素相位变化超过 π：请缩短两期时间间隔，或提高画幅分辨率。")
    if not mask.any():
        warnings.append("本场景没有达到阈值的沉降增量，可作为负样本；也可调整日期或标签阈值。")
    if settings.complex_observation and settings.looks == 1 and settings.spatial_window == 1:
        warnings.append("单次观测且不做空间平均时，样本相干性恒为 1，不能作为质量判断。")
    normalized = request_dict(settings, faces)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "software_versions": runtime_versions(),
        "request_sha256": fingerprint(normalized),
        "group_id": group_identity(normalized),
        "array_schema": array_schema(arrays),
        "validity_rule": "water_mask == 0 and model_coherence >= valid_coherence_threshold",
        "estimated_coherence_kind": (
            "sample complex correlation magnitude"
            if settings.complex_observation
            else "circular resultant length; not SAR coherence"
        ),
        "calibration_status": "synthetic; no real-mine calibration or external validation",
        "environment_source": "synthetic spectral terrain and randomized geometric water; not measured DEM or DEM-derived hydrology",
        "water_scene": water_layout(settings),
        "water_percent": float(env["water_mask"].mean() * 100),
        "valid_percent": float(valid.mean() * 100),
        "mean_model_coherence": float(gamma.mean()),
        "mean_estimated_coherence": float(estimated.mean()),
        "dem_min_m": float(env["dem_m"].min()),
        "dem_max_m": float(env["dem_m"].max()),
        "engine": ENGINE,
        "request": normalized,
        "pixel_spacing_m": settings.extent_m / settings.size,
        "peak_delta_down_mm": float(down.max() * 1000),
        "min_los_mm": float(los.min() * 1000),
        "max_los_mm": float(los.max() * 1000),
        "mask_pixels": int(mask.sum()),
        "mask_percent": float(mask.mean() * 100),
        "overlap_pixels": int(np.sum(labels[0] & labels[1])),
        "phase_cycles": float(np.ptp(phase) / (2 * np.pi)),
        "profile_row": peak // settings.size,
        "days": [settings.day_before, settings.day_after],
        "warnings": warnings,
        "fringe_label_rule": "abs(clean deformation phase) >= fringe_threshold_rad; experimental support proxy, not calibrated visible boundary",
        "fringe_validity_rule": "all pixels supervised; no automatic water or low-coherence erasure",
        "fringe_mask_percent": float(arrays["fringe_mask"].mean() * 100),
        "label_rule": "delta_down_m >= target_threshold_mm/1000 before observation disturbances",
        "observation_model": (
            "correlated complex Gaussian pair; independent ensembles and spatial box average on map grid"
            if settings.complex_observation
            else "legacy empirical circular phase noise; environment coherence not applied"
        ),
        "time_model": "equal-length extraction segments with center-passage activation and exponential response",
        "scope": "horizontal seam and rectangular panels; synthetic educational model, not calibrated mine prediction",
    }
    return arrays, metadata
