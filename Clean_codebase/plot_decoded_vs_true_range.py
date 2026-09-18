from pathlib import Path
from typing import Sequence, Tuple

import os

MPL_CACHE_DIR = Path(__file__).with_name(".mplcache")
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from grid_search_evaluation_range import FMSweepGridSearchFramework, GridTrialResult


EXPERIMENT_CFG = {
    "object_counts": [1],
    "trials_per_count": 100,
    "trials_per_batch": 32,
    "min_distance_m": 0.15,
    "max_distance_m": 3.4,
    "reflectivity_range": (0.1, 0.8),
    "roughness_range": (0.1, 0.35),
    "tolerance_steps": 2,
    "min_separation_steps": 0,
    "seed": 420,
}

NETWORK_GRID = {
    "weight": [0.1],
    "beta": [0.1],
    "threshold": [0.7],
    "rs": [2],
    "lis": [3],
    "lir": [8],
    "inhib_decay": [0.1],
}

READOUT_GRID = {
    "beta": [0.4],
    "threshold": [0.4],
    "is": [0],
    "ir": [4],
    "inhibition_falloff": ["gaussian"],
    "sigma": [2.0],
}

SCENE_GRID = {
    "snr_db": [None],
    "clutter_fraction": [0.0],
}

OUTPUT_PATH = Path(__file__).with_name("fig_decoded_vs_true_range.png")
BLUE = "#3B6CA8"
LIGHT_RED = "#D86B6B"


def collect_decoded_pairs(
    framework: FMSweepGridSearchFramework,
    results: Sequence[GridTrialResult],
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    true_distances = []
    decoded_distances = []
    missed_trials = 0
    extra_detections = 0

    for result in results:
        if len(result.true_distances_m) != 1 or len(result.true_delays) != 1:
            continue

        if not result.detection_rows:
            missed_trials += 1
            continue

        true_delay = result.true_delays[0]
        best_detection = min(
            result.detection_rows,
            key=lambda row: abs(row.delay_steps - true_delay),
        )

        true_distances.append(float(result.true_distances_m[0]))
        decoded_distances.append(framework._delay_steps_to_distance_m(best_detection.delay_steps))
        extra_detections += max(0, len(result.detection_rows) - 1)

    return (
        np.asarray(true_distances, dtype=float),
        np.asarray(decoded_distances, dtype=float),
        missed_trials,
        extra_detections,
    )


def run_experiment() -> Tuple[np.ndarray, np.ndarray, int, int]:
    torch.manual_seed(EXPERIMENT_CFG["seed"])
    np.random.seed(EXPERIMENT_CFG["seed"])

    framework = FMSweepGridSearchFramework(seed=EXPERIMENT_CFG["seed"])
    results = framework.evaluate_grid(
        network_grid=NETWORK_GRID,
        readout_grid=READOUT_GRID,
        scene_grid=SCENE_GRID,
        object_counts=EXPERIMENT_CFG["object_counts"],
        trials_per_count=EXPERIMENT_CFG["trials_per_count"],
        min_distance_m=EXPERIMENT_CFG["min_distance_m"],
        max_distance_m=EXPERIMENT_CFG["max_distance_m"],
        reflectivity_range=EXPERIMENT_CFG["reflectivity_range"],
        roughness_range=EXPERIMENT_CFG["roughness_range"],
        tolerance_steps=EXPERIMENT_CFG["tolerance_steps"],
        min_separation_steps=EXPERIMENT_CFG["min_separation_steps"],
        trials_per_batch=EXPERIMENT_CFG["trials_per_batch"],
    )

    return collect_decoded_pairs(framework=framework, results=results)


def plot_decoded_vs_true(
    true_distances: np.ndarray,
    decoded_distances: np.ndarray,
    missed_trials: int,
    extra_detections: int,
) -> None:
    if true_distances.size == 0:
        raise RuntimeError("No decoded single-object trials were available to plot.")

    signed_errors_m = decoded_distances - true_distances
    mae_m = float(np.mean(np.abs(signed_errors_m)))
    rmse_m = float(np.sqrt(np.mean(signed_errors_m**2)))

    min_distance = EXPERIMENT_CFG["min_distance_m"]
    max_distance = EXPERIMENT_CFG["max_distance_m"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    scatter_ax, error_ax = axes

    scatter_ax.scatter(
        true_distances,
        decoded_distances,
        s=44,
        color=BLUE,
        alpha=0.78,
        edgecolors="black",
        linewidths=0.35,
    )
    scatter_ax.plot(
        [min_distance, max_distance],
        [min_distance, max_distance],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Ideal",
    )
    scatter_ax.set_title("Decoded vs True Distance")
    scatter_ax.set_xlabel("True distance (m)")
    scatter_ax.set_ylabel("Decoded distance (m)")
    scatter_ax.set_xlim(min_distance - 0.05, max_distance + 0.05)
    scatter_ax.set_ylim(min_distance - 0.05, max_distance + 0.05)
    scatter_ax.grid(True, alpha=0.3)
    scatter_ax.legend(frameon=False)

    error_ax.scatter(
        true_distances,
        signed_errors_m * 100.0,
        s=44,
        color=LIGHT_RED,
        alpha=0.78,
        edgecolors="black",
        linewidths=0.35,
    )
    error_ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    error_ax.set_title("Error vs True Distance")
    error_ax.set_xlabel("True distance (m)")
    error_ax.set_ylabel("Decoded - true distance (cm)")
    error_ax.set_xlim(min_distance - 0.05, max_distance + 0.05)
    error_ax.grid(True, alpha=0.3)

    fig.suptitle("Single-Object Range Accuracy Across the Operating Range")
    fig.text(
        0.5,
        0.01,
        (
            f"n={true_distances.size}, MAE={mae_m:.4f} m ({mae_m * 100.0:.2f} cm), "
            f"RMSE={rmse_m:.4f} m, misses={missed_trials}, extra detections={extra_detections}"
        ),
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(OUTPUT_PATH, dpi=300)
    print(f"Saved figure to {OUTPUT_PATH}")
    if "agg" not in matplotlib.get_backend().lower():
        plt.show()


def main() -> None:
    true_distances, decoded_distances, missed_trials, extra_detections = run_experiment()
    plot_decoded_vs_true(
        true_distances=true_distances,
        decoded_distances=decoded_distances,
        missed_trials=missed_trials,
        extra_detections=extra_detections,
    )


if __name__ == "__main__":
    main()
