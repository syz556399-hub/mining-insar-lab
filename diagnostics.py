"""Report fixed-scenario numerical/statistical diagnostics, not field accuracy."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from mine_model import Face, Settings, grid, panel_epochs
from provenance import ENGINE, runtime_versions
from scene_environment import observe


def diagnostics():
    face = Face()
    changes = {}
    for count in (8, 16, 32, 64, 128):
        first, second = panel_epochs(face, 128, 1600, count, 40, 52)
        changes[count] = second - first
    reference = changes[128]
    convergence = [
        {
            "segments": count,
            "relative_l2_to_128": float(
                np.linalg.norm(value - reference) / np.linalg.norm(reference)
            ),
            "peak_delta_mm": float(value.max() * 1000),
        }
        for count, value in changes.items()
    ]
    settings = Settings(size=128, looks=32, spatial_window=1)
    east, north = grid(settings)
    correlations = []
    for gamma in (0, 0.3, 0.7, 1):
        env = {
            "model_coherence": np.full(east.shape, gamma),
            "backscatter_power": np.ones(east.shape),
        }
        cross, _, estimated, p1, p2 = observe(np.full(east.shape, 0.8), settings, env, east, north)
        rho = cross.mean() / np.sqrt(p1.mean() * p2.mean())
        correlations.append(
            {
                "model_gamma": gamma,
                "population_correlation_error": float(abs(rho - gamma * np.exp(0.8j))),
                "mean_sample_magnitude": float(estimated.mean()),
            }
        )
    one = replace(settings, looks=1)
    env = {"model_coherence": np.zeros(east.shape), "backscatter_power": np.ones(east.shape)}
    _, _, single, _, _ = observe(np.zeros(east.shape), one, env, east, north)
    return {
        "engine": ENGINE,
        "software_versions": runtime_versions(),
        "scenario": "Face defaults; 128px, 1600m, days 40/52",
        "convergence": convergence,
        "reference_note": "128 segments is a numerical reference, not an exact continuous-time solution",
        "complex_pair_diagnostics": correlations,
        "gamma_zero_single_sample_coherence_mean": float(single.mean()),
        "interpretation": "Numerical/statistical verification only. No real-mine error or detection performance measured.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; choose a new path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(diagnostics(), indent=2), encoding="utf-8")
    print(args.out)
