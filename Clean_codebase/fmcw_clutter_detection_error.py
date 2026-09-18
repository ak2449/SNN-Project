"""Classical FMCW clutter sweep: detection error vs clutter fraction.

This is the clutter counterpart to `fmcw_snr_detection_error.py`. It reuses
the same FMCW dechirp + FFT detector and the same detection-error scoring,
but fixes SNR to no added white noise by default and sweeps multipath clutter
fractions.

Outputs, by default, are written next to this script:
  - fmcw_clutter_detection_error_summary.csv
  - fmcw_clutter_detection_error_trials.csv
  - fig_fmcw_clutter_detection_error.png
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from fmcw_snr_detection_error import (
    DEFAULT_OUTPUT_DIR,
    FmcwConfig,
    FmcwTrialResult,
    _build_chirp,
    _format_float_list,
    _print_progress,
    run_fmcw_trial,
)


DEFAULT_CLUTTER_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


@dataclass
class ClutterTrialResult:
    clutter_fraction: float
    snr_db: Optional[float]
    trial_index: int
    num_objects: int
    true_distances_m: List[float]
    detected_distances_m: List[float]
    misses: int
    false_positives: int
    abs_distance_errors_m: List[float]
    has_detection_error: int


@dataclass
class ClutterSweepResult:
    clutter_fraction: float
    snr_db: Optional[float]
    num_objects: int
    trials: int
    detection_error_trials: int
    detection_error_probability: float
    miss_rate: float
    false_positive_rate: float
    mean_num_detections: float
    mean_abs_distance_error_m: float
    range_tolerance_m: float


def _format_optional_float(value: Optional[float]) -> str:
    return "" if value is None else f"{float(value):g}"


def _wrap_trial_result(
    clutter_fraction: float,
    result: FmcwTrialResult,
) -> ClutterTrialResult:
    return ClutterTrialResult(
        clutter_fraction=float(clutter_fraction),
        snr_db=result.snr_db,
        trial_index=result.trial_index,
        num_objects=result.num_objects,
        true_distances_m=result.true_distances_m,
        detected_distances_m=result.detected_distances_m,
        misses=result.misses,
        false_positives=result.false_positives,
        abs_distance_errors_m=result.abs_distance_errors_m,
        has_detection_error=result.has_detection_error,
    )


def _summarise_group(
    clutter_fraction: float,
    snr_db: Optional[float],
    num_objects: int,
    group: Sequence[ClutterTrialResult],
    tolerance_m: float,
) -> ClutterSweepResult:
    if not group:
        raise RuntimeError("Cannot summarise an empty FMCW clutter result group.")

    trials = len(group)
    total_true = sum(len(result.true_distances_m) for result in group)
    total_detected = sum(len(result.detected_distances_m) for result in group)
    misses = sum(result.misses for result in group)
    false_positives = sum(result.false_positives for result in group)
    detection_error_trials = sum(result.has_detection_error for result in group)
    abs_distance_errors_m = [
        error
        for result in group
        for error in result.abs_distance_errors_m
    ]

    return ClutterSweepResult(
        clutter_fraction=float(clutter_fraction),
        snr_db=snr_db,
        num_objects=int(num_objects),
        trials=trials,
        detection_error_trials=detection_error_trials,
        detection_error_probability=detection_error_trials / float(trials),
        miss_rate=misses / float(total_true) if total_true else 0.0,
        false_positive_rate=(
            false_positives / float(total_detected) if total_detected else 0.0
        ),
        mean_num_detections=total_detected / float(trials),
        mean_abs_distance_error_m=(
            float(np.mean(abs_distance_errors_m))
            if abs_distance_errors_m
            else float("nan")
        ),
        range_tolerance_m=float(tolerance_m),
    )


def run_clutter_sweep(
    clutter_fractions: Sequence[float],
    object_counts: Sequence[int],
    trials_per_count: int,
    min_distance_m: float,
    max_distance_m: float,
    reflectivity_range: Tuple[float, float],
    min_separation_m: float,
    tolerance_m: float,
    snr_db: Optional[float],
    seed: int,
    cfg: FmcwConfig,
) -> Tuple[List[ClutterTrialResult], List[ClutterSweepResult]]:
    rng = np.random.default_rng(seed)
    chirp = _build_chirp(cfg)
    trial_rows: List[ClutterTrialResult] = []
    summary_rows: List[ClutterSweepResult] = []
    total_trials = (
        len(clutter_fractions)
        * len(object_counts)
        * int(trials_per_count)
    )
    completed_trials = 0
    progress_interval = max(1, total_trials // 64)

    _print_progress(0, total_trials)
    for clutter_fraction in clutter_fractions:
        for num_objects in object_counts:
            group: List[ClutterTrialResult] = []
            for trial_index in range(1, int(trials_per_count) + 1):
                result = run_fmcw_trial(
                    snr_db=snr_db,
                    trial_index=trial_index,
                    num_objects=int(num_objects),
                    min_distance_m=min_distance_m,
                    max_distance_m=max_distance_m,
                    reflectivity_range=reflectivity_range,
                    min_separation_m=min_separation_m,
                    tolerance_m=tolerance_m,
                    rng=rng,
                    cfg=cfg,
                    chirp=chirp,
                    clutter_fraction=float(clutter_fraction),
                )
                wrapped = _wrap_trial_result(
                    clutter_fraction=float(clutter_fraction),
                    result=result,
                )
                trial_rows.append(wrapped)
                group.append(wrapped)
                completed_trials += 1
                if (
                    completed_trials == total_trials
                    or completed_trials % progress_interval == 0
                ):
                    _print_progress(completed_trials, total_trials)

            summary_rows.append(
                _summarise_group(
                    clutter_fraction=float(clutter_fraction),
                    snr_db=snr_db,
                    num_objects=int(num_objects),
                    group=group,
                    tolerance_m=tolerance_m,
                )
            )

    if total_trials > 0:
        print()

    return trial_rows, summary_rows


def save_trial_csv(rows: Sequence[ClutterTrialResult], output_path: Path) -> None:
    fieldnames = [
        "clutter_fraction",
        "snr_db",
        "trial_index",
        "num_objects",
        "true_distances_m",
        "detected_distances_m",
        "misses",
        "false_positives",
        "has_detection_error",
        "abs_distance_errors_m",
    ]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "clutter_fraction": row.clutter_fraction,
                    "snr_db": _format_optional_float(row.snr_db),
                    "trial_index": row.trial_index,
                    "num_objects": row.num_objects,
                    "true_distances_m": _format_float_list(row.true_distances_m),
                    "detected_distances_m": _format_float_list(row.detected_distances_m),
                    "misses": row.misses,
                    "false_positives": row.false_positives,
                    "has_detection_error": row.has_detection_error,
                    "abs_distance_errors_m": _format_float_list(row.abs_distance_errors_m),
                }
            )


def save_summary_csv(rows: Sequence[ClutterSweepResult], output_path: Path) -> None:
    fieldnames = [
        "clutter_fraction",
        "snr_db",
        "num_objects",
        "trials",
        "detection_error_trials",
        "detection_error_probability",
        "miss_rate",
        "false_positive_rate",
        "mean_num_detections",
        "mean_abs_distance_error_m",
        "range_tolerance_m",
    ]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "clutter_fraction": row.clutter_fraction,
                    "snr_db": _format_optional_float(row.snr_db),
                    "num_objects": row.num_objects,
                    "trials": row.trials,
                    "detection_error_trials": row.detection_error_trials,
                    "detection_error_probability": row.detection_error_probability,
                    "miss_rate": row.miss_rate,
                    "false_positive_rate": row.false_positive_rate,
                    "mean_num_detections": row.mean_num_detections,
                    "mean_abs_distance_error_m": row.mean_abs_distance_error_m,
                    "range_tolerance_m": row.range_tolerance_m,
                }
            )


def plot_detection_error_vs_clutter(
    rows: Sequence[ClutterSweepResult],
    output_path: Path,
) -> None:
    if not rows:
        raise RuntimeError("No FMCW clutter sweep rows were generated.")

    mpl_cache_dir = output_path.parent / ".mplcache"
    mpl_cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    object_counts = sorted({row.num_objects for row in rows})

    for num_objects in object_counts:
        group = sorted(
            [row for row in rows if row.num_objects == num_objects],
            key=lambda row: row.clutter_fraction,
        )
        clutter_fraction = np.asarray(
            [row.clutter_fraction for row in group],
            dtype=float,
        )
        detection_error = np.asarray(
            [row.detection_error_probability for row in group],
            dtype=float,
        )
        label = f"{num_objects} object" + ("" if num_objects == 1 else "s")
        line, = ax.plot(
            clutter_fraction,
            detection_error,
            marker="o",
            markersize=4,
            linewidth=2,
            color="#D86B6B",
            label=label,
        )
        ax.scatter(
            clutter_fraction,
            detection_error,
            s=48,
            color=line.get_color(),
            zorder=3,
        )
        for x_value, y_value in zip(clutter_fraction, detection_error):
            y_offset = -14 if y_value >= 0.95 else 8
            ax.annotate(
                f"{y_value:.3f}",
                xy=(x_value, y_value),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax.set_title("FMCW Detection Error vs Clutter")
    ax.set_xlabel("Clutter fraction")
    ax.set_ylabel("Detection error probability")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(sorted({row.clutter_fraction for row in rows}))
    ax.grid(True, alpha=0.3)
    ax.text(
        0.5,
        -0.22,
        (
            f"Dechirp + FFT detector, snr_db={_format_optional_float(rows[0].snr_db) or 'None'}, "
            f"range tolerance={rows[0].range_tolerance_m:.3f} m, "
            f"n={rows[0].trials} trials/clutter value"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(rows: Sequence[ClutterSweepResult], cfg: FmcwConfig) -> None:
    print("\n" + "=" * 122)
    print("FMCW CLUTTER SWEEP")
    print("=" * 122)
    print(
        f"range_resolution={cfg.range_resolution_m:.4f} m | "
        f"duration={cfg.duration_s * 1e3:.1f} ms | "
        f"bandwidth={cfg.bandwidth_hz / 1e3:.1f} kHz | "
        f"clutter_paths={cfg.num_clutter_paths}"
    )
    print(
        "clutter | objects | trials | errors/trials | det_err | miss_rate | "
        "fp_rate | mean_detections | mean_abs_err_m"
    )

    for row in sorted(rows, key=lambda item: (item.num_objects, item.clutter_fraction)):
        error_count_text = f"{row.detection_error_trials}/{row.trials}"
        print(
            f"{row.clutter_fraction:7.2f} | {row.num_objects:7d} | "
            f"{row.trials:6d} | {error_count_text:13s} | "
            f"{row.detection_error_probability:7.3f} | {row.miss_rate:9.3f} | "
            f"{row.false_positive_rate:7.3f} | {row.mean_num_detections:15.2f} | "
            f"{row.mean_abs_distance_error_m:14.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classical FMCW detection-error probability vs clutter fraction."
    )
    parser.add_argument(
        "--clutter-fractions",
        type=float,
        nargs="+",
        default=DEFAULT_CLUTTER_FRACTIONS,
    )
    parser.add_argument("--object-counts", type=int, nargs="+", default=[1])
    parser.add_argument("--trials-per-count", type=int, default=500)
    parser.add_argument("--min-distance-m", type=float, default=0.15)
    parser.add_argument("--max-distance-m", type=float, default=3.4)
    parser.add_argument("--reflectivity-min", type=float, default=0.1)
    parser.add_argument("--reflectivity-max", type=float, default=0.8)
    parser.add_argument("--tolerance-steps", type=float, default=2.0)
    parser.add_argument("--min-separation-steps", type=float, default=0.0)
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--threshold-sigma", type=float, default=8.0)
    parser.add_argument("--relative-peak-threshold", type=float, default=0.35)
    parser.add_argument("--max-detections", type=int, default=8)
    parser.add_argument("--num-clutter-paths", type=int, default=50)
    parser.add_argument("--clutter-min-distance-m", type=float, default=0.05)
    parser.add_argument("--clutter-max-distance-m", type=float, default=None)
    parser.add_argument("--clutter-reflectivity-min", type=float, default=0.2)
    parser.add_argument("--clutter-reflectivity-max", type=float, default=0.6)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FmcwConfig(
        threshold_sigma=args.threshold_sigma,
        relative_peak_threshold=args.relative_peak_threshold,
        max_detections=args.max_detections,
        num_clutter_paths=args.num_clutter_paths,
        clutter_min_distance_m=args.clutter_min_distance_m,
        clutter_max_distance_m=args.clutter_max_distance_m,
        clutter_reflectivity_range=(
            args.clutter_reflectivity_min,
            args.clutter_reflectivity_max,
        ),
    )
    tolerance_m = cfg.steps_to_distance_m(args.tolerance_steps)
    min_separation_m = cfg.steps_to_distance_m(args.min_separation_steps)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_rows, summary_rows = run_clutter_sweep(
        clutter_fractions=args.clutter_fractions,
        object_counts=args.object_counts,
        trials_per_count=args.trials_per_count,
        min_distance_m=args.min_distance_m,
        max_distance_m=args.max_distance_m,
        reflectivity_range=(args.reflectivity_min, args.reflectivity_max),
        min_separation_m=min_separation_m,
        tolerance_m=tolerance_m,
        snr_db=args.snr_db,
        seed=args.seed,
        cfg=cfg,
    )

    summary_path = output_dir / "fmcw_clutter_detection_error_summary.csv"
    trial_path = output_dir / "fmcw_clutter_detection_error_trials.csv"
    figure_path = output_dir / "fig_fmcw_clutter_detection_error.png"

    print_summary(summary_rows, cfg)
    save_summary_csv(summary_rows, summary_path)
    save_trial_csv(trial_rows, trial_path)
    if not args.no_plot:
        plot_detection_error_vs_clutter(summary_rows, figure_path)

    print(f"\nSaved summary CSV to {summary_path}")
    print(f"Saved trial CSV to {trial_path}")
    if not args.no_plot:
        print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
