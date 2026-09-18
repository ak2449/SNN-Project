"""Classical FMCW SNR sweep: detection error vs SNR.

This is the FMCW counterpart to `grid_range_old.py`: it keeps the same
range-scale defaults, SNR grid, trial count, and detection-error definition,
but replaces the FM-FM SNN delay bank with a conventional dechirp + FFT
range detector.

Outputs, by default, are written next to this script:
  - fmcw_snr_detection_error_summary.csv
  - fmcw_snr_detection_error_trials.csv
  - fig_fmcw_snr_detection_error.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_SNR_DB = [-20, -15, -10, -5, 0, 5, 10, 15]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


@dataclass
class FmcwConfig:
    sample_rate_hz: float = 1_000_000.0
    num_steps: int = 250
    step_duration_ms: float = 0.1
    f_start_hz: float = 90_000.0
    f_end_hz: float = 30_000.0
    speed_of_sound_m_s: float = 343.0
    threshold_sigma: float = 8.0
    relative_peak_threshold: float = 0.35
    min_peak_separation_m: float = 0.025
    max_detections: int = 8
    num_clutter_paths: int = 50
    clutter_min_distance_m: float = 0.05
    clutter_max_distance_m: Optional[float] = None
    clutter_reflectivity_range: Tuple[float, float] = (0.2, 0.6)

    @property
    def duration_s(self) -> float:
        return self.num_steps * self.step_duration_ms * 1e-3

    @property
    def n_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.duration_s))

    @property
    def bandwidth_hz(self) -> float:
        return abs(self.f_end_hz - self.f_start_hz)

    @property
    def sweep_slope_hz_s(self) -> float:
        return (self.f_end_hz - self.f_start_hz) / self.duration_s

    @property
    def range_resolution_m(self) -> float:
        return self.speed_of_sound_m_s / (2.0 * self.bandwidth_hz)

    def steps_to_distance_m(self, delay_steps: float) -> float:
        return (
            0.5
            * self.speed_of_sound_m_s
            * float(delay_steps)
            * self.step_duration_ms
            * 1e-3
        )


@dataclass
class MatchMetrics:
    misses: int
    false_positives: int
    abs_distance_errors_m: List[float]


@dataclass
class FmcwTrialResult:
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
class SnrSweepResult:
    snr_db: float
    num_objects: int
    trials: int
    detection_error_trials: int
    detection_error_probability: float
    miss_rate: float
    false_positive_rate: float
    mean_num_detections: float
    mean_abs_distance_error_m: float
    range_tolerance_m: float


def _print_progress(completed: int, total: int) -> None:
    if total <= 0:
        return

    bar_width = 32
    ratio = min(max(completed / float(total), 0.0), 1.0)
    filled = int(round(bar_width * ratio))
    bar = "#" * filled + "-" * (bar_width - filled)
    sys.stdout.write(f"\rProgress [{bar}] {completed}/{total} ({ratio * 100:5.1f}%)")
    sys.stdout.flush()


def _build_chirp(cfg: FmcwConfig) -> Tuple[np.ndarray, np.ndarray]:
    t = np.arange(cfg.n_samples, dtype=np.float64) / cfg.sample_rate_hz
    phase = 2.0 * np.pi * (
        cfg.f_start_hz * t + 0.5 * cfg.sweep_slope_hz_s * t * t
    )
    return t, np.exp(1j * phase).astype(np.complex128)


def _chirp_at_time(t: np.ndarray, cfg: FmcwConfig) -> np.ndarray:
    phase = 2.0 * np.pi * (
        cfg.f_start_hz * t + 0.5 * cfg.sweep_slope_hz_s * t * t
    )
    return np.exp(1j * phase).astype(np.complex128)


def _sample_distances_m(
    rng: np.random.Generator,
    num_objects: int,
    min_distance_m: float,
    max_distance_m: float,
    min_separation_m: float,
) -> List[float]:
    accepted: List[float] = []
    attempts = 0
    max_attempts = 512

    while len(accepted) < int(num_objects) and attempts < max_attempts:
        attempts += 1
        candidate = float(rng.uniform(min_distance_m, max_distance_m))
        if any(abs(candidate - existing) < min_separation_m for existing in accepted):
            continue
        accepted.append(candidate)

    if len(accepted) != int(num_objects):
        raise RuntimeError(
            "Could not sample a valid FMCW scene. Reduce min separation or widen "
            "the distance range."
        )

    return sorted(accepted)


def _delayed_fmcw_path(
    t: np.ndarray,
    distance_m: float,
    amplitude_scale: float,
    rng: np.random.Generator,
    cfg: FmcwConfig,
) -> np.ndarray:
    path = np.zeros(t.shape, dtype=np.complex128)
    round_trip_m = 2.0 * float(distance_m)
    tau_s = round_trip_m / cfg.speed_of_sound_m_s
    delayed_t = t - tau_s
    active = (delayed_t >= 0.0) & (delayed_t < cfg.duration_s)
    if not np.any(active):
        return path

    random_phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
    path[active] = (
        float(amplitude_scale)
        * random_phase
        * _chirp_at_time(delayed_t[active], cfg)
    )
    return path


def _generate_fmcw_clutter(
    t: np.ndarray,
    target_rx: np.ndarray,
    clutter_fraction: float,
    rng: np.random.Generator,
    cfg: FmcwConfig,
) -> np.ndarray:
    if clutter_fraction <= 0.0 or cfg.num_clutter_paths <= 0:
        return np.zeros_like(target_rx)

    max_observable_distance_m = 0.5 * cfg.speed_of_sound_m_s * cfg.duration_s
    clutter_max_distance_m = (
        max_observable_distance_m
        if cfg.clutter_max_distance_m is None
        else float(cfg.clutter_max_distance_m)
    )
    clutter_max_distance_m = max(cfg.clutter_min_distance_m, clutter_max_distance_m)
    reflectivity_min, reflectivity_max = cfg.clutter_reflectivity_range

    clutter = np.zeros_like(target_rx)
    for _ in range(int(cfg.num_clutter_paths)):
        distance_m = float(
            rng.uniform(cfg.clutter_min_distance_m, clutter_max_distance_m)
        )
        round_trip_m = 2.0 * distance_m
        reflectivity = float(rng.uniform(reflectivity_min, reflectivity_max))
        geometric_attenuation = 1.0 / (1.0 + round_trip_m)
        path = _delayed_fmcw_path(
            t=t,
            distance_m=distance_m,
            amplitude_scale=reflectivity * geometric_attenuation,
            rng=rng,
            cfg=cfg,
        )
        if rng.uniform() < 0.5:
            path = -path
        clutter += path

    clutter_peak = float(np.max(np.abs(clutter))) if clutter.size else 0.0
    target_peak = float(np.max(np.abs(target_rx))) if target_rx.size else 0.0
    if clutter_peak <= 1e-12 or target_peak <= 1e-12:
        return np.zeros_like(target_rx)

    return clutter * (target_peak * float(clutter_fraction) / clutter_peak)


def _simulate_fmcw_rx(
    tx: np.ndarray,
    t: np.ndarray,
    distances_m: Sequence[float],
    reflectivity_range: Tuple[float, float],
    snr_db: Optional[float],
    rng: np.random.Generator,
    cfg: FmcwConfig,
    clutter_fraction: float = 0.0,
) -> np.ndarray:
    clean_rx = np.zeros_like(tx)

    for distance_m in distances_m:
        round_trip_m = 2.0 * float(distance_m)
        reflectivity = float(rng.uniform(*reflectivity_range))
        geometric_attenuation = 1.0 / (1.0 + round_trip_m)
        clean_rx += _delayed_fmcw_path(
            t=t,
            distance_m=distance_m,
            amplitude_scale=reflectivity * geometric_attenuation,
            rng=rng,
            cfg=cfg,
        )

    target_rx = clean_rx.copy()
    clean_rx = clean_rx + _generate_fmcw_clutter(
        t=t,
        target_rx=target_rx,
        clutter_fraction=float(clutter_fraction),
        rng=rng,
        cfg=cfg,
    )

    if snr_db is None or not np.isfinite(float(snr_db)):
        return clean_rx

    active_reference = target_rx[np.abs(target_rx) > 1e-12]
    if active_reference.size == 0:
        return clean_rx

    signal_power = float(np.mean(np.abs(active_reference) ** 2))
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise_std = np.sqrt(noise_power / 2.0)
    noise = noise_std * (
        rng.standard_normal(clean_rx.shape) + 1j * rng.standard_normal(clean_rx.shape)
    )
    return clean_rx + noise


def _local_peak_indices(values: np.ndarray, threshold: float) -> List[int]:
    peaks: List[int] = []
    if values.size < 3:
        return peaks

    for idx in range(1, values.size - 1):
        center = values[idx]
        if center < threshold:
            continue
        if center >= values[idx - 1] and center >= values[idx + 1]:
            if center > values[idx - 1] or center > values[idx + 1]:
                peaks.append(idx)

    return peaks


def _robust_spectrum_threshold(magnitude: np.ndarray, cfg: FmcwConfig) -> float:
    if magnitude.size == 0:
        return float("inf")

    median = float(np.median(magnitude))
    mad = float(np.median(np.abs(magnitude - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= 0.0:
        robust_sigma = float(np.std(magnitude)) + 1e-12

    absolute_threshold = median + cfg.threshold_sigma * robust_sigma
    relative_threshold = cfg.relative_peak_threshold * float(np.max(magnitude))
    return max(absolute_threshold, relative_threshold)


def detect_fmcw_ranges(
    tx: np.ndarray,
    rx: np.ndarray,
    min_distance_m: float,
    max_distance_m: float,
    cfg: FmcwConfig,
) -> List[float]:
    if cfg.sweep_slope_hz_s == 0.0:
        raise ValueError("FMCW sweep slope cannot be zero.")

    if cfg.sweep_slope_hz_s > 0:
        dechirped = tx * np.conj(rx)
    else:
        dechirped = np.conj(tx) * rx

    window = np.hanning(dechirped.size)
    spectrum = np.fft.fft(dechirped * window)
    freqs_hz = np.fft.fftfreq(dechirped.size, d=1.0 / cfg.sample_rate_hz)

    positive = freqs_hz > 0.0
    beat_freq_hz = freqs_hz[positive]
    magnitude = np.abs(spectrum[positive])
    ranges_m = (
        cfg.speed_of_sound_m_s
        * beat_freq_hz
        / (2.0 * abs(cfg.sweep_slope_hz_s))
    )

    valid = (ranges_m >= min_distance_m) & (ranges_m <= max_distance_m)
    beat_freq_hz = beat_freq_hz[valid]
    magnitude = magnitude[valid]
    ranges_m = ranges_m[valid]
    if magnitude.size == 0:
        return []

    threshold = _robust_spectrum_threshold(magnitude, cfg)
    peak_indices = _local_peak_indices(magnitude, threshold)
    if not peak_indices:
        return []

    freq_bin_hz = (
        float(beat_freq_hz[1] - beat_freq_hz[0])
        if beat_freq_hz.size > 1
        else cfg.sample_rate_hz / float(cfg.n_samples)
    )
    ranked = sorted(peak_indices, key=lambda idx: magnitude[idx], reverse=True)
    selected: List[Tuple[float, float]] = []

    for idx in ranked:
        refined_idx = float(idx)
        if 0 < idx < magnitude.size - 1:
            left = magnitude[idx - 1]
            center = magnitude[idx]
            right = magnitude[idx + 1]
            denom = left - 2.0 * center + right
            if denom != 0.0:
                refined_idx += float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))

        refined_freq_hz = beat_freq_hz[idx] + (refined_idx - idx) * freq_bin_hz
        refined_range_m = (
            cfg.speed_of_sound_m_s
            * refined_freq_hz
            / (2.0 * abs(cfg.sweep_slope_hz_s))
        )

        if any(
            abs(refined_range_m - existing_range_m) < cfg.min_peak_separation_m
            for existing_range_m, _ in selected
        ):
            continue

        selected.append((float(refined_range_m), float(magnitude[idx])))
        if len(selected) >= int(cfg.max_detections):
            break

    return sorted(range_m for range_m, _ in selected)


def match_ranges(
    true_distances_m: Sequence[float],
    detected_distances_m: Sequence[float],
    tolerance_m: float,
) -> MatchMetrics:
    true_sorted = [float(value) for value in sorted(true_distances_m)]
    detected_sorted = [float(value) for value in sorted(detected_distances_m)]
    candidate_pairs: List[Tuple[float, int, int]] = []

    for true_idx, true_distance_m in enumerate(true_sorted):
        for detected_idx, detected_distance_m in enumerate(detected_sorted):
            abs_error_m = abs(true_distance_m - detected_distance_m)
            if abs_error_m <= float(tolerance_m):
                candidate_pairs.append((abs_error_m, true_idx, detected_idx))

    candidate_pairs.sort(key=lambda item: item[0])
    matched_true = set()
    matched_detected = set()
    abs_distance_errors_m: List[float] = []

    for abs_error_m, true_idx, detected_idx in candidate_pairs:
        if true_idx in matched_true or detected_idx in matched_detected:
            continue
        matched_true.add(true_idx)
        matched_detected.add(detected_idx)
        abs_distance_errors_m.append(float(abs_error_m))

    return MatchMetrics(
        misses=len(true_sorted) - len(matched_true),
        false_positives=len(detected_sorted) - len(matched_detected),
        abs_distance_errors_m=abs_distance_errors_m,
    )


def run_fmcw_trial(
    snr_db: Optional[float],
    trial_index: int,
    num_objects: int,
    min_distance_m: float,
    max_distance_m: float,
    reflectivity_range: Tuple[float, float],
    min_separation_m: float,
    tolerance_m: float,
    rng: np.random.Generator,
    cfg: FmcwConfig,
    chirp: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    clutter_fraction: float = 0.0,
) -> FmcwTrialResult:
    t, tx = _build_chirp(cfg) if chirp is None else chirp
    true_distances_m = _sample_distances_m(
        rng=rng,
        num_objects=num_objects,
        min_distance_m=min_distance_m,
        max_distance_m=max_distance_m,
        min_separation_m=min_separation_m,
    )
    rx = _simulate_fmcw_rx(
        tx=tx,
        t=t,
        distances_m=true_distances_m,
        reflectivity_range=reflectivity_range,
        snr_db=snr_db,
        rng=rng,
        cfg=cfg,
        clutter_fraction=clutter_fraction,
    )
    detected_distances_m = detect_fmcw_ranges(
        tx=tx,
        rx=rx,
        min_distance_m=min_distance_m,
        max_distance_m=max_distance_m,
        cfg=cfg,
    )
    metrics = match_ranges(
        true_distances_m=true_distances_m,
        detected_distances_m=detected_distances_m,
        tolerance_m=tolerance_m,
    )

    return FmcwTrialResult(
        snr_db=None if snr_db is None else float(snr_db),
        trial_index=int(trial_index),
        num_objects=int(num_objects),
        true_distances_m=list(true_distances_m),
        detected_distances_m=list(detected_distances_m),
        misses=metrics.misses,
        false_positives=metrics.false_positives,
        abs_distance_errors_m=metrics.abs_distance_errors_m,
        has_detection_error=int(metrics.misses > 0 or metrics.false_positives > 0),
    )


def _summarise_group(
    snr_db: float,
    num_objects: int,
    group: Sequence[FmcwTrialResult],
    tolerance_m: float,
) -> SnrSweepResult:
    if not group:
        raise RuntimeError("Cannot summarise an empty FMCW result group.")

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

    return SnrSweepResult(
        snr_db=float(snr_db),
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


def run_snr_sweep(
    snr_db_values: Sequence[float],
    object_counts: Sequence[int],
    trials_per_count: int,
    min_distance_m: float,
    max_distance_m: float,
    reflectivity_range: Tuple[float, float],
    min_separation_m: float,
    tolerance_m: float,
    seed: int,
    cfg: FmcwConfig,
) -> Tuple[List[FmcwTrialResult], List[SnrSweepResult]]:
    rng = np.random.default_rng(seed)
    trial_rows: List[FmcwTrialResult] = []
    summary_rows: List[SnrSweepResult] = []
    chirp = _build_chirp(cfg)
    total_trials = (
        len(snr_db_values)
        * len(object_counts)
        * int(trials_per_count)
    )
    completed_trials = 0
    progress_interval = max(1, total_trials // 64)

    _print_progress(0, total_trials)
    for snr_db in snr_db_values:
        for num_objects in object_counts:
            group: List[FmcwTrialResult] = []
            for trial_index in range(1, int(trials_per_count) + 1):
                result = run_fmcw_trial(
                    snr_db=float(snr_db),
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
                )
                trial_rows.append(result)
                group.append(result)
                completed_trials += 1
                if (
                    completed_trials == total_trials
                    or completed_trials % progress_interval == 0
                ):
                    _print_progress(completed_trials, total_trials)

            summary_rows.append(
                _summarise_group(
                    snr_db=float(snr_db),
                    num_objects=int(num_objects),
                    group=group,
                    tolerance_m=tolerance_m,
                )
            )

    if total_trials > 0:
        print()

    return trial_rows, summary_rows


def _format_float_list(values: Sequence[float], precision: int = 4) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def save_trial_csv(rows: Sequence[FmcwTrialResult], output_path: Path) -> None:
    fieldnames = [
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
                    "snr_db": row.snr_db,
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


def save_summary_csv(rows: Sequence[SnrSweepResult], output_path: Path) -> None:
    fieldnames = [
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
                    "snr_db": row.snr_db,
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


def plot_detection_error_vs_snr(
    rows: Sequence[SnrSweepResult],
    output_path: Path,
) -> None:
    if not rows:
        raise RuntimeError("No FMCW SNR sweep rows were generated.")

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
            key=lambda row: row.snr_db,
        )
        snr_db = np.asarray([row.snr_db for row in group], dtype=float)
        detection_error = np.asarray(
            [row.detection_error_probability for row in group],
            dtype=float,
        )
        label = f"{num_objects} object" + ("" if num_objects == 1 else "s")
        line, = ax.plot(
            snr_db,
            detection_error,
            marker="o",
            markersize=4,
            linewidth=2,
            label=label,
        )
        ax.scatter(
            snr_db,
            detection_error,
            s=48,
            color=line.get_color(),
            zorder=3,
        )
        for x_value, y_value in zip(snr_db, detection_error):
            y_offset = -14 if y_value >= 0.95 else 8
            ax.annotate(
                f"{y_value:.3f}",
                xy=(x_value, y_value),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax.set_title("FMCW Detection Error vs SNR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Detection error probability")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(sorted({row.snr_db for row in rows}))
    ax.grid(True, alpha=0.3)
    ax.text(
        0.5,
        -0.22,
        (
            f"Dechirp + FFT detector, range tolerance={rows[0].range_tolerance_m:.3f} m, "
            f"n={rows[0].trials} trials/SNR"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(rows: Sequence[SnrSweepResult], cfg: FmcwConfig) -> None:
    print("\n" + "=" * 118)
    print("FMCW SNR SWEEP")
    print("=" * 118)
    print(
        f"range_resolution={cfg.range_resolution_m:.4f} m | "
        f"duration={cfg.duration_s * 1e3:.1f} ms | "
        f"bandwidth={cfg.bandwidth_hz / 1e3:.1f} kHz"
    )
    print(
        "snr_db | objects | trials | errors/trials | det_err | miss_rate | "
        "fp_rate | mean_detections | mean_abs_err_m"
    )

    for row in sorted(rows, key=lambda item: (item.num_objects, item.snr_db)):
        error_count_text = f"{row.detection_error_trials}/{row.trials}"
        print(
            f"{row.snr_db:6.1f} | {row.num_objects:7d} | {row.trials:6d} | "
            f"{error_count_text:13s} | {row.detection_error_probability:7.3f} | "
            f"{row.miss_rate:9.3f} | {row.false_positive_rate:7.3f} | "
            f"{row.mean_num_detections:15.2f} | {row.mean_abs_distance_error_m:14.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classical FMCW detection-error probability vs SNR."
    )
    parser.add_argument("--snrs", type=float, nargs="+", default=DEFAULT_SNR_DB)
    parser.add_argument("--object-counts", type=int, nargs="+", default=[1])
    parser.add_argument("--trials-per-count", type=int, default=500)
    parser.add_argument("--min-distance-m", type=float, default=0.15)
    parser.add_argument("--max-distance-m", type=float, default=3.4)
    parser.add_argument("--reflectivity-min", type=float, default=0.1)
    parser.add_argument("--reflectivity-max", type=float, default=0.8)
    parser.add_argument("--tolerance-steps", type=float, default=2.0)
    parser.add_argument("--min-separation-steps", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--threshold-sigma", type=float, default=8.0)
    parser.add_argument("--relative-peak-threshold", type=float, default=0.35)
    parser.add_argument("--max-detections", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FmcwConfig(
        threshold_sigma=args.threshold_sigma,
        relative_peak_threshold=args.relative_peak_threshold,
        max_detections=args.max_detections,
    )
    tolerance_m = cfg.steps_to_distance_m(args.tolerance_steps)
    min_separation_m = cfg.steps_to_distance_m(args.min_separation_steps)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_rows, summary_rows = run_snr_sweep(
        snr_db_values=args.snrs,
        object_counts=args.object_counts,
        trials_per_count=args.trials_per_count,
        min_distance_m=args.min_distance_m,
        max_distance_m=args.max_distance_m,
        reflectivity_range=(args.reflectivity_min, args.reflectivity_max),
        min_separation_m=min_separation_m,
        tolerance_m=tolerance_m,
        seed=args.seed,
        cfg=cfg,
    )

    summary_path = output_dir / "fmcw_snr_detection_error_summary.csv"
    trial_path = output_dir / "fmcw_snr_detection_error_trials.csv"
    figure_path = output_dir / "fig_fmcw_snr_detection_error.png"

    print_summary(summary_rows, cfg)
    save_summary_csv(summary_rows, summary_path)
    save_trial_csv(trial_rows, trial_path)
    if not args.no_plot:
        plot_detection_error_vs_snr(summary_rows, figure_path)

    print(f"\nSaved summary CSV to {summary_path}")
    print(f"Saved trial CSV to {trial_path}")
    if not args.no_plot:
        print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
