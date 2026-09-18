"""Focused SNR sweep for mean velocity error.

This script is a small wrapper around `velocity_evaluation_vel.py` that runs
only the SNR sweep and plots mean absolute velocity error against SNR.

Outputs, by default, are written to `Velocity_v1/evaluation_outputs/`:
  - snr_mean_velocity_error_summary.csv
  - snr_mean_velocity_error_trials.csv
  - fig_snr_mean_velocity_error.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "snn_velocity_mplcache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from velocity_evaluation_vel import SweepSummaryRow, VelocityEvaluationFramework


PLOT_BLUE = "#3B6CA8"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "evaluation_outputs"
DEFAULT_SNR_VALUES = [-20, -15, -10, -5, 0, 5, 10, 15, 20]


def _as_float_list(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


def plot_snr_mean_velocity_error(
    rows: Sequence[SweepSummaryRow],
    output_path: Path,
    show: bool = False,
) -> None:
    if not rows:
        raise RuntimeError("No SNR sweep rows were generated.")

    rows = sorted(rows, key=lambda row: row.x_value)
    snr_db = np.asarray([row.x_value for row in rows], dtype=float)
    mean_abs_error = np.asarray(
        [row.mean_abs_velocity_error_m_s for row in rows],
        dtype=float,
    )
    std_abs_error = np.asarray(
        [row.std_abs_velocity_error_m_s for row in rows],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(
        snr_db,
        mean_abs_error,
        yerr=std_abs_error,
        marker="o",
        linewidth=2,
        capsize=4,
        color=PLOT_BLUE,
        ecolor="0.35",
        label="Mean absolute velocity error +/- SD",
    )
    ax.scatter(snr_db, mean_abs_error, s=52, color=PLOT_BLUE, zorder=3)
    for x_value, y_value in zip(snr_db, mean_abs_error):
        ax.annotate(
            f"{y_value:.2f}",
            xy=(x_value, y_value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.set_title("SNR Against Mean Velocity Error")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Mean absolute velocity error (m/s)")
    ax.set_xticks(snr_db)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def print_summary(rows: Sequence[SweepSummaryRow]) -> None:
    print("\n" + "=" * 110)
    print("SNR AGAINST MEAN VELOCITY ERROR")
    print("=" * 110)
    print(
        "snr_db | trials | valid | mean_error_m_s | mean_abs_error_m_s | "
        "rmse_m_s | velocity_miss"
    )
    for row in sorted(rows, key=lambda item: item.x_value):
        print(
            f"{row.x_value:6.1f} | {row.trials:6d} | "
            f"{row.valid_velocity_trials:5d} | "
            f"{row.mean_velocity_error_m_s:14.3f} | "
            f"{row.mean_abs_velocity_error_m_s:18.3f} | "
            f"{row.rmse_velocity_m_s:8.3f} | "
            f"{row.velocity_miss_rate:13.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the velocity SNR sweep and plot mean velocity error."
    )
    parser.add_argument("--snrs", type=float, nargs="+", default=DEFAULT_SNR_VALUES)
    parser.add_argument("--trials-per-snr", type=int, default=100)
    parser.add_argument("--distance-m", type=float, default=1.15)
    parser.add_argument("--velocity-m-s", type=float, default=3.0)
    parser.add_argument("--reflectivity-min", type=float, default=0.85)
    parser.add_argument("--reflectivity-max", type=float, default=0.95)
    parser.add_argument("--roughness-min", type=float, default=0.1)
    parser.add_argument("--roughness-max", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    framework = VelocityEvaluationFramework(seed=args.seed)
    results, summary_rows = framework.evaluate_snr_sweep(
        snr_values=_as_float_list(args.snrs),
        distance_m=args.distance_m,
        velocity_m_s=args.velocity_m_s,
        trials_per_snr=args.trials_per_snr,
        reflectivity_range=(args.reflectivity_min, args.reflectivity_max),
        roughness_range=(args.roughness_min, args.roughness_max),
        progress=True,
        plot=False,
        show_plot=False,
    )

    summary_path = output_dir / "snr_mean_velocity_error_summary.csv"
    trial_path = output_dir / "snr_mean_velocity_error_trials.csv"
    figure_path = output_dir / "fig_snr_mean_velocity_error.png"

    framework.write_summary_csv(summary_rows, summary_path)
    framework.write_trial_results_csv(results, trial_path)
    plot_snr_mean_velocity_error(
        rows=summary_rows,
        output_path=figure_path,
        show=args.show_plot,
    )
    print_summary(summary_rows)
    print(f"\nSaved summary CSV to {summary_path}")
    print(f"Saved trial CSV to {trial_path}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
