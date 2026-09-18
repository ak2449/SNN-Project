"""Classical FMCW SNR sweep: mean absolute velocity error vs SNR.

This is the FMCW counterpart to `snr_mean_velocity_error.py`. It uses a
triangular FMCW measurement: one up-chirp and one down-chirp are dechirped,
their signed beat frequencies are estimated by FFT peak picking, and the
Doppler component is converted to radial velocity.

Outputs, by default, are written to `Velocity_v1/evaluation_outputs/`:
  - fmcw_snr_mean_velocity_error_summary.csv
  - fmcw_snr_mean_velocity_error_trials.csv
  - fig_fmcw_snr_mean_velocity_error.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "snn_velocity_mplcache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SNR_BLUE = "#3B6CA8"
ERROR_RED = "#D86B6B"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "evaluation_outputs"
DEFAULT_SNR_VALUES = [-20, -15, -10, -5, 0, 5, 10, 15, 20]


@dataclass
class FmcwVelocityConfig:
    sample_rate_hz: float = 1_000_000.0
    chirp_duration_s: float = 0.025
    f_min_hz: float = 30_000.0
    f_max_hz: float = 90_000.0
    speed_of_sound_m_s: float = 343.0
    min_distance_m: float = 0.15
    max_distance_m: float = 3.4
    max_abs_velocity_m_s: float = 12.0
    peak_search_margin_hz: float = 1500.0

    @property
    def n_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.chirp_duration_s))

    @property
    def bandwidth_hz(self) -> float:
        return self.f_max_hz - self.f_min_hz

    @property
    def center_frequency_hz(self) -> float:
        return 0.5 * (self.f_min_hz + self.f_max_hz)

    @property
    def slope_hz_s(self) -> float:
        return self.bandwidth_hz / self.chirp_duration_s

    @property
    def velocity_resolution_m_s(self) -> float:
        freq_bin_hz = self.sample_rate_hz / float(self.n_samples)
        return 0.5 * freq_bin_hz * self.speed_of_sound_m_s / (
            2.0 * self.center_frequency_hz
        )


@dataclass
class FmcwVelocityTrialResult:
    trial_index: int
    snr_db: float
    true_distance_m: float
    true_velocity_m_s: float
    decoded_distance_m: float
    decoded_velocity_m_s: float
    range_abs_error_m: float
    velocity_error_m_s: float
    velocity_abs_error_m_s: float
    beat_up_hz: float
    beat_down_hz: float
    doppler_hz: float


@dataclass
class FmcwVelocitySummaryRow:
    sweep_name: str
    x_value: float
    trials: int
    mean_decoded_velocity_m_s: float
    std_decoded_velocity_m_s: float
    mean_velocity_error_m_s: float
    mean_abs_velocity_error_m_s: float
    std_abs_velocity_error_m_s: float
    rmse_velocity_m_s: float
    mean_range_abs_error_m: float


def _print_progress(completed: int, total: int) -> None:
    if total <= 0:
        return

    bar_width = 32
    ratio = min(max(completed / float(total), 0.0), 1.0)
    filled = int(round(bar_width * ratio))
    bar = "#" * filled + "-" * (bar_width - filled)
    sys.stdout.write(f"\rProgress [{bar}] {completed}/{total} ({ratio * 100:5.1f}%)")
    sys.stdout.flush()
    if completed >= total:
        sys.stdout.write("\n")


def _chirp_phase(t: np.ndarray, f_start_hz: float, slope_hz_s: float) -> np.ndarray:
    return 2.0 * np.pi * (f_start_hz * t + 0.5 * slope_hz_s * t * t)


def _build_chirp(
    cfg: FmcwVelocityConfig,
    f_start_hz: float,
    slope_hz_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    t = np.arange(cfg.n_samples, dtype=np.float64) / cfg.sample_rate_hz
    tx = np.exp(1j * _chirp_phase(t, f_start_hz, slope_hz_s))
    return t, tx.astype(np.complex128)


def _simulate_chirp_echo(
    t: np.ndarray,
    distance_m: float,
    velocity_m_s: float,
    reflectivity: float,
    snr_db: float,
    f_start_hz: float,
    slope_hz_s: float,
    rng: np.random.Generator,
    cfg: FmcwVelocityConfig,
) -> np.ndarray:
    tau_s = 2.0 * float(distance_m) / cfg.speed_of_sound_m_s
    delayed_t = t - tau_s
    active = (delayed_t >= 0.0) & (delayed_t < cfg.chirp_duration_s)
    clean_rx = np.zeros(t.shape, dtype=np.complex128)

    if np.any(active):
        round_trip_m = 2.0 * float(distance_m)
        geometric_attenuation = 1.0 / ((1.0 + round_trip_m) ** 2)
        amplitude = float(reflectivity) * geometric_attenuation
        doppler_hz = (
            2.0
            * float(velocity_m_s)
            * cfg.center_frequency_hz
            / cfg.speed_of_sound_m_s
        )
        random_phase = rng.uniform(0.0, 2.0 * np.pi)
        clean_rx[active] = amplitude * np.exp(
            1j
            * (
                _chirp_phase(delayed_t[active], f_start_hz, slope_hz_s)
                + 2.0 * np.pi * doppler_hz * t[active]
                + random_phase
            )
        )

    signal_power = float(np.mean(np.abs(clean_rx[active]) ** 2)) if np.any(active) else 0.0
    if signal_power <= 0.0:
        return clean_rx

    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise_std = np.sqrt(noise_power / 2.0)
    noise = noise_std * (
        rng.standard_normal(clean_rx.shape) + 1j * rng.standard_normal(clean_rx.shape)
    )
    return clean_rx + noise


def _refine_peak_frequency(
    magnitudes: np.ndarray,
    frequencies_hz: np.ndarray,
    peak_index: int,
) -> float:
    refined_index = float(peak_index)
    if 0 < peak_index < magnitudes.size - 1:
        left = magnitudes[peak_index - 1]
        center = magnitudes[peak_index]
        right = magnitudes[peak_index + 1]
        denom = left - 2.0 * center + right
        if denom != 0.0:
            refined_index += float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))

    freq_bin_hz = (
        float(frequencies_hz[1] - frequencies_hz[0])
        if frequencies_hz.size > 1
        else 0.0
    )
    return float(frequencies_hz[peak_index] + (refined_index - peak_index) * freq_bin_hz)


def estimate_signed_beat_frequency(
    tx: np.ndarray,
    rx: np.ndarray,
    cfg: FmcwVelocityConfig,
) -> float:
    dechirped = np.conj(tx) * rx
    window = np.hanning(dechirped.size)
    spectrum = np.fft.fftshift(np.fft.fft(dechirped * window))
    frequencies_hz = np.fft.fftshift(
        np.fft.fftfreq(dechirped.size, d=1.0 / cfg.sample_rate_hz)
    )
    magnitudes = np.abs(spectrum)

    tau_max_s = 2.0 * cfg.max_distance_m / cfg.speed_of_sound_m_s
    max_doppler_hz = (
        2.0
        * cfg.max_abs_velocity_m_s
        * cfg.center_frequency_hz
        / cfg.speed_of_sound_m_s
    )
    max_abs_beat_hz = (
        abs(cfg.slope_hz_s) * tau_max_s
        + max_doppler_hz
        + cfg.peak_search_margin_hz
    )
    valid = np.abs(frequencies_hz) <= max_abs_beat_hz
    if not np.any(valid):
        raise RuntimeError("No valid FMCW beat-frequency bins were available.")

    valid_magnitudes = magnitudes[valid]
    valid_frequencies = frequencies_hz[valid]
    peak_index = int(np.argmax(valid_magnitudes))
    return _refine_peak_frequency(valid_magnitudes, valid_frequencies, peak_index)


def run_fmcw_velocity_trial(
    trial_index: int,
    snr_db: float,
    distance_m: float,
    velocity_m_s: float,
    reflectivity_range: Tuple[float, float],
    rng: np.random.Generator,
    cfg: FmcwVelocityConfig,
    chirps: Optional[
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    ] = None,
) -> FmcwVelocityTrialResult:
    if chirps is None:
        up = _build_chirp(cfg, cfg.f_min_hz, cfg.slope_hz_s)
        down = _build_chirp(cfg, cfg.f_max_hz, -cfg.slope_hz_s)
    else:
        up, down = chirps

    reflectivity = float(rng.uniform(*reflectivity_range))
    t_up, tx_up = up
    t_down, tx_down = down
    rx_up = _simulate_chirp_echo(
        t=t_up,
        distance_m=distance_m,
        velocity_m_s=velocity_m_s,
        reflectivity=reflectivity,
        snr_db=snr_db,
        f_start_hz=cfg.f_min_hz,
        slope_hz_s=cfg.slope_hz_s,
        rng=rng,
        cfg=cfg,
    )
    rx_down = _simulate_chirp_echo(
        t=t_down,
        distance_m=distance_m,
        velocity_m_s=velocity_m_s,
        reflectivity=reflectivity,
        snr_db=snr_db,
        f_start_hz=cfg.f_max_hz,
        slope_hz_s=-cfg.slope_hz_s,
        rng=rng,
        cfg=cfg,
    )

    beat_up_hz = estimate_signed_beat_frequency(tx_up, rx_up, cfg)
    beat_down_hz = estimate_signed_beat_frequency(tx_down, rx_down, cfg)
    doppler_hz = 0.5 * (beat_up_hz + beat_down_hz)
    delay_s = (beat_down_hz - beat_up_hz) / (2.0 * cfg.slope_hz_s)
    decoded_distance_m = 0.5 * cfg.speed_of_sound_m_s * delay_s
    decoded_velocity_m_s = (
        doppler_hz
        * cfg.speed_of_sound_m_s
        / (2.0 * cfg.center_frequency_hz)
    )

    velocity_error_m_s = decoded_velocity_m_s - float(velocity_m_s)
    return FmcwVelocityTrialResult(
        trial_index=int(trial_index),
        snr_db=float(snr_db),
        true_distance_m=float(distance_m),
        true_velocity_m_s=float(velocity_m_s),
        decoded_distance_m=float(decoded_distance_m),
        decoded_velocity_m_s=float(decoded_velocity_m_s),
        range_abs_error_m=abs(float(decoded_distance_m) - float(distance_m)),
        velocity_error_m_s=float(velocity_error_m_s),
        velocity_abs_error_m_s=abs(float(velocity_error_m_s)),
        beat_up_hz=float(beat_up_hz),
        beat_down_hz=float(beat_down_hz),
        doppler_hz=float(doppler_hz),
    )


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _std(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=0)) if values else float("nan")


def summarize_results(
    snr_db: float,
    results: Sequence[FmcwVelocityTrialResult],
) -> FmcwVelocitySummaryRow:
    decoded_velocities = [result.decoded_velocity_m_s for result in results]
    velocity_errors = [result.velocity_error_m_s for result in results]
    velocity_abs_errors = [result.velocity_abs_error_m_s for result in results]
    range_abs_errors = [result.range_abs_error_m for result in results]
    rmse_velocity = (
        float(np.sqrt(np.mean(np.asarray(velocity_errors, dtype=float) ** 2)))
        if velocity_errors
        else float("nan")
    )

    return FmcwVelocitySummaryRow(
        sweep_name="snr_db",
        x_value=float(snr_db),
        trials=len(results),
        mean_decoded_velocity_m_s=_mean(decoded_velocities),
        std_decoded_velocity_m_s=_std(decoded_velocities),
        mean_velocity_error_m_s=_mean(velocity_errors),
        mean_abs_velocity_error_m_s=_mean(velocity_abs_errors),
        std_abs_velocity_error_m_s=_std(velocity_abs_errors),
        rmse_velocity_m_s=rmse_velocity,
        mean_range_abs_error_m=_mean(range_abs_errors),
    )


def run_snr_sweep(
    snr_values: Sequence[float],
    trials_per_snr: int,
    distance_m: float,
    velocity_m_s: float,
    reflectivity_range: Tuple[float, float],
    seed: int,
    cfg: FmcwVelocityConfig,
    progress: bool = True,
) -> Tuple[List[FmcwVelocityTrialResult], List[FmcwVelocitySummaryRow]]:
    rng = np.random.default_rng(seed)
    trial_rows: List[FmcwVelocityTrialResult] = []
    summary_rows: List[FmcwVelocitySummaryRow] = []
    chirps = (
        _build_chirp(cfg, cfg.f_min_hz, cfg.slope_hz_s),
        _build_chirp(cfg, cfg.f_max_hz, -cfg.slope_hz_s),
    )
    total = len(snr_values) * int(trials_per_snr)
    completed = 0
    progress_interval = max(1, total // 64)
    if progress:
        _print_progress(completed, total)

    for snr_db in snr_values:
        group: List[FmcwVelocityTrialResult] = []
        for trial_index in range(1, int(trials_per_snr) + 1):
            result = run_fmcw_velocity_trial(
                trial_index=trial_index,
                snr_db=float(snr_db),
                distance_m=distance_m,
                velocity_m_s=velocity_m_s,
                reflectivity_range=reflectivity_range,
                rng=rng,
                cfg=cfg,
                chirps=chirps,
            )
            trial_rows.append(result)
            group.append(result)
            completed += 1
            if progress and (
                completed == total
                or completed % progress_interval == 0
            ):
                _print_progress(completed, total)

        summary_rows.append(summarize_results(snr_db=float(snr_db), results=group))

    return trial_rows, summary_rows


def write_trial_csv(rows: Sequence[FmcwVelocityTrialResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary_csv(rows: Sequence[FmcwVelocitySummaryRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def plot_snr_mean_velocity_error(
    rows: Sequence[FmcwVelocitySummaryRow],
    output_path: Path,
    show: bool = False,
) -> None:
    if not rows:
        raise RuntimeError("No FMCW velocity SNR rows were generated.")

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
        color=SNR_BLUE,
        ecolor=ERROR_RED,
        elinewidth=1.5,
        label="FMCW velocity MAE +/- SD",
    )
    ax.scatter(snr_db, mean_abs_error, s=52, color=SNR_BLUE, zorder=3)

    ax.set_title("FMCW SNR Against Mean Velocity Error")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Mean absolute velocity error (m/s)")
    ax.set_xticks(snr_db)
    ax.set_yscale("linear")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def print_summary(rows: Sequence[FmcwVelocitySummaryRow], cfg: FmcwVelocityConfig) -> None:
    print("\n" + "=" * 112)
    print("FMCW SNR AGAINST MEAN VELOCITY ERROR")
    print("=" * 112)
    print(
        f"chirp_duration={cfg.chirp_duration_s * 1e3:.1f} ms | "
        f"bandwidth={cfg.bandwidth_hz / 1e3:.1f} kHz | "
        f"velocity_bin~={cfg.velocity_resolution_m_s:.3f} m/s"
    )
    print(
        "snr_db | trials | mean_error_m_s | mean_abs_error_m_s | "
        "rmse_m_s | mean_range_abs_error_m"
    )
    for row in sorted(rows, key=lambda item: item.x_value):
        print(
            f"{row.x_value:6.1f} | {row.trials:6d} | "
            f"{row.mean_velocity_error_m_s:14.3f} | "
            f"{row.mean_abs_velocity_error_m_s:18.3f} | "
            f"{row.rmse_velocity_m_s:8.3f} | "
            f"{row.mean_range_abs_error_m:22.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a classical FMCW SNR sweep for mean velocity error."
    )
    parser.add_argument("--snrs", type=float, nargs="+", default=DEFAULT_SNR_VALUES)
    parser.add_argument("--trials-per-snr", type=int, default=100)
    parser.add_argument("--distance-m", type=float, default=1.15)
    parser.add_argument("--velocity-m-s", type=float, default=3.0)
    parser.add_argument("--reflectivity-min", type=float, default=0.85)
    parser.add_argument("--reflectivity-max", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = FmcwVelocityConfig()

    trial_rows, summary_rows = run_snr_sweep(
        snr_values=[float(value) for value in args.snrs],
        trials_per_snr=args.trials_per_snr,
        distance_m=args.distance_m,
        velocity_m_s=args.velocity_m_s,
        reflectivity_range=(args.reflectivity_min, args.reflectivity_max),
        seed=args.seed,
        cfg=cfg,
        progress=True,
    )

    summary_path = output_dir / "fmcw_snr_mean_velocity_error_summary.csv"
    trial_path = output_dir / "fmcw_snr_mean_velocity_error_trials.csv"
    figure_path = output_dir / "fig_fmcw_snr_mean_velocity_error.png"

    write_summary_csv(summary_rows, summary_path)
    write_trial_csv(trial_rows, trial_path)
    plot_snr_mean_velocity_error(
        rows=summary_rows,
        output_path=figure_path,
        show=args.show_plot,
    )
    print_summary(summary_rows, cfg)
    print(f"\nSaved summary CSV to {summary_path}")
    print(f"Saved trial CSV to {trial_path}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
