"""Matched-filter baseline: range MAE vs SNR, SNN vs matched filter.

This script implements the textbook classical baseline described in
Section 4.4 of the report. For each trial it generates the same
pulse and noisy echo waveforms that the SNN sees (identical seeds, same
echo physics), runs both the matched-filter estimator and the SNN's
range pathway on those waveforms, and aggregates per-SNR results into a
single figure with both curves on the same axes.

The matched-filter pipeline:

  1. Cross-correlate the transmitted pulse against the received echo
     via FFT.
  2. Estimate the noise floor from the pre-echo region of the echo
     waveform (samples before the earliest possible round-trip arrival
     at the closest allowed object distance).
  3. Detect candidate delays as local maxima of the absolute correlation
     exceeding the larger of (a) a CFAR threshold of `k * sigma_noise`
     with `k` chosen for a per-sample false-alarm probability of order
     10^-9 under Gaussian noise, and (b) a relative-peak threshold set
     as a fraction of the global correlation maximum, which suppresses
     sidelobes of the dominant peak that pass the CFAR test but are
     in fact part of the same echo. This relative threshold mirrors
     the lateral-inhibition behaviour of the SNN's FM-FM bank.
  4. Refine each peak by three-point parabolic interpolation around the
     local maximum, giving fractional-sample delay resolution.
  5. Convert sample-domain delay to delay-steps and to round-trip
     distance using the same constants the SNN uses.
  6. Match against ground truth using the framework's existing greedy
     matcher with the same tolerance (10 delay steps).

Scope note. The current configuration is tuned for the single-object
SNR sweep (S5 of Section 4.2). For a multi-object resolution comparison
the `relative_peak_threshold` would need to be lowered so that a dim
secondary echo is not suppressed by a bright primary one; this is left
as a configurable parameter at the cost of more spurious detections
under noise.

Outputs:
  - `range_baseline_trials.csv`     -- one row per trial (both estimators)
  - `range_baseline_summary.csv`    -- one row per SNR point with MAE
                                        and Wilson CI for miss rate
  - `range_baseline.png`            -- two curves on the same axes
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_model_range import FMSweepEvaluationFramework


# ---------------------------------------------------------------------------
# Matched-filter estimator.
# ---------------------------------------------------------------------------


@dataclass
class MatchedFilterConfig:
    """Configuration for the matched-filter detector."""

    cfar_k: float = 6.0
    # k * sigma_noise threshold; k=6.0 corresponds to per-sample P_FA ~ 1e-9
    # for Gaussian noise (very conservative) and avoids false detections
    # on small noise excursions. Lower values would trade specificity for
    # sensitivity.

    min_peak_separation_samples: int = 100
    # Local-maximum gating: peaks closer than this in the correlation
    # output are merged (keep the larger). 100 samples at 1 MHz = 100 us
    # = one SNN delay step (~17.2 mm of round-trip distance), so two
    # detected peaks closer than this are unresolvable by the SNN anyway
    # and merging them gives a fair like-for-like comparison.

    relative_peak_threshold: float = 0.5
    # Peaks below this fraction of the global correlation maximum are
    # rejected as sidelobes of the dominant peak. The CF tone in the
    # transmitted pulse autocorrelates with substantial sidelobe
    # structure at its CF period (16.7 us at 60 kHz), so a CFAR
    # threshold alone allows many spurious detections that are in fact
    # echoes of a single object. The relative threshold matches the
    # SNN's lateral-inhibition behaviour, which similarly suppresses
    # weaker neighbours of a dominant peak.

    noise_region_max_samples: Optional[int] = None
    # If set, only the first N samples of the echo are used to estimate
    # the noise floor. If None, the region is computed from
    # min_object_distance_m at construction time.


def matched_filter_detect(
    pulse_waveform: np.ndarray,
    echo_waveform: np.ndarray,
    *,
    pulse_active_slice: Tuple[int, int],
    sample_rate_hz: float,
    speed_of_sound_m_s: float,
    step_duration_s: float,
    min_object_distance_m: float,
    cfg: MatchedFilterConfig,
) -> Tuple[List[float], List[float], np.ndarray]:
    """Run the matched filter on a single trial.

    Returns
    -------
    estimated_delays_steps : list of float
        Refined delay estimates in **delay steps** (the SNN's native
        units) so the values are directly comparable to ground truth and
        usable by the framework's matcher.
    estimated_distances_m : list of float
        Estimated round-trip distances in metres.
    correlation : np.ndarray
        The full absolute cross-correlation, returned so the caller can
        inspect or plot it if desired.
    """
    pulse = np.asarray(pulse_waveform, dtype=np.float64)
    echo = np.asarray(echo_waveform, dtype=np.float64)

    # --- Cross-correlation via FFT (only over the active pulse window).
    pa_lo, pa_hi = pulse_active_slice
    active_pulse = pulse[pa_lo:pa_hi]
    if active_pulse.size == 0:
        return [], [], np.zeros_like(echo)

    n_fft = int(2 ** np.ceil(np.log2(echo.size + active_pulse.size - 1)))
    P = np.fft.rfft(active_pulse, n=n_fft)
    E = np.fft.rfft(echo, n=n_fft)
    full_corr = np.fft.irfft(np.conj(P) * E, n=n_fft)[: echo.size + active_pulse.size - 1]
    # corr[k] aligns with echo[k + pa_lo]: peak index k means the active
    # pulse starts at sample k + pa_lo of the echo, so the round-trip
    # delay in samples (relative to the transmitted pulse onset, which
    # is at sample pa_lo of the trial) is k.
    abs_corr = np.abs(full_corr)

    # --- Noise-floor estimate from the guaranteed-empty pre-echo region.
    min_delay_samples = int(
        round((2.0 * min_object_distance_m / speed_of_sound_m_s) * sample_rate_hz)
    )
    noise_region_end = (
        cfg.noise_region_max_samples
        if cfg.noise_region_max_samples is not None
        else min_delay_samples
    )
    noise_region_end = max(32, min(noise_region_end, abs_corr.size))
    noise_samples = abs_corr[:noise_region_end]
    # Use the MAD-derived sigma for robustness against any leakage from
    # the active pulse into the early samples.
    sigma_noise = 1.4826 * np.median(np.abs(noise_samples - np.median(noise_samples)))
    if sigma_noise <= 0:
        sigma_noise = float(np.std(noise_samples)) + 1e-12
    threshold = cfg.cfar_k * sigma_noise

    # --- Combined CFAR + relative-peak threshold.
    #     CFAR alone is satisfied by many sidelobes of the dominant peak
    #     because the CF tone has substantial autocorrelation structure
    #     at its period; the relative threshold drops anything below
    #     `relative_peak_threshold` * (global max) and so removes those
    #     sidelobes while leaving genuinely distinct echoes intact.
    relative_floor = cfg.relative_peak_threshold * float(abs_corr.max())
    effective_threshold = max(threshold, relative_floor)
    above = abs_corr > effective_threshold
    if not np.any(above):
        return [], [], abs_corr

    # Find local maxima: idx where above[idx] and abs_corr[idx] is larger
    # than its immediate neighbours.
    candidate_indices: List[int] = []
    for idx in range(1, abs_corr.size - 1):
        if not above[idx]:
            continue
        if abs_corr[idx] >= abs_corr[idx - 1] and abs_corr[idx] >= abs_corr[idx + 1]:
            candidate_indices.append(idx)

    if not candidate_indices:
        return [], [], abs_corr

    # Greedy non-maximum suppression by separation.
    candidate_indices.sort(key=lambda i: -abs_corr[i])
    selected: List[int] = []
    for idx in candidate_indices:
        if all(abs(idx - s) >= cfg.min_peak_separation_samples for s in selected):
            selected.append(idx)

    # --- Parabolic sub-sample refinement.
    estimated_delays_samples: List[float] = []
    for idx in selected:
        y_m = abs_corr[idx - 1]
        y_0 = abs_corr[idx]
        y_p = abs_corr[idx + 1]
        denom = (y_m - 2.0 * y_0 + y_p)
        if denom != 0.0:
            delta = 0.5 * (y_m - y_p) / denom
            delta = float(np.clip(delta, -0.5, 0.5))
        else:
            delta = 0.0
        estimated_delays_samples.append(idx + delta)

    # --- Convert sample-domain delays to delay-steps and distances.
    #     Correlation peak index = round-trip delay in samples relative to
    #     the transmitted pulse onset, so the conversion is
    #         delay_steps = (samples / f_s) / step_duration_s
    #         distance_m  = 0.5 * c * (samples / f_s)
    estimated_delays_steps: List[float] = []
    estimated_distances_m: List[float] = []
    for delay_samples in estimated_delays_samples:
        delay_s = delay_samples / sample_rate_hz
        estimated_delays_steps.append(delay_s / step_duration_s)
        estimated_distances_m.append(0.5 * speed_of_sound_m_s * delay_s)

    return estimated_delays_steps, estimated_distances_m, abs_corr


# ---------------------------------------------------------------------------
# Framework extension: per-trial paired sweep over both estimators.
# ---------------------------------------------------------------------------


class PairedSnrSweepFramework(FMSweepEvaluationFramework):
    """Drives the SNN and matched filter on the exact same trials."""

    def run_paired_snr_trial(
        self,
        distance_m: float,
        reflectivity_range: Tuple[float, float],
        roughness_range: Tuple[float, float],
        snr_db: float,
        clutter_fraction: float,
        tolerance_steps: int,
        mf_cfg: MatchedFilterConfig,
    ) -> dict:
        """Run one trial; return SNN and matched-filter metrics together."""
        from sim_modules.echo_range import generate_fm_echo_from_waveform

        # --- Build pulse exactly as the framework does in run_trial.
        pulse_waveform, pulse_spikes = self._build_pulse()

        # --- Sample object properties for this trial.
        obj_echo_cfg = dict(self.echo_default_cfg)
        obj_echo_cfg["distance_m"] = float(distance_m)
        obj_echo_cfg["reflectivity"] = float(self.rng.uniform(*reflectivity_range))
        obj_echo_cfg["surface_roughness"] = float(self.rng.uniform(*roughness_range))
        obj_echo_cfg["snr_db"] = float(snr_db)
        obj_echo_cfg["clutter_fraction"] = float(clutter_fraction)

        # --- Generate the echo. This is the single source of truth for
        #     both estimators: both will see the resulting `echo_waveform`
        #     (matched filter) and `echo_spikes` (SNN), and they are
        #     produced from the same noise realisation.
        echo_spikes, echo_waveform, true_delay = generate_fm_echo_from_waveform(
            pulse_waveform=pulse_waveform,
            encoder=self.encoder,
            num_steps=self.sim_cfg["num_steps"],
            distance_m=obj_echo_cfg["distance_m"],
            reflectivity=obj_echo_cfg["reflectivity"],
            max_spike_prob=obj_echo_cfg["max_spike_prob"],
            sample_rate_hz=self.sim_cfg["sample_rate_hz"],
            step_duration_ms=self.sim_cfg["step_duration_ms"],
            speed_of_sound_m_s=obj_echo_cfg["speed_of_sound_m_s"],
            air_absorption_db_per_m_100khz=obj_echo_cfg["air_absorption_db_per_m_100khz"],
            surface_roughness=obj_echo_cfg["surface_roughness"],
            surface_texture_scale_m=obj_echo_cfg["surface_texture_scale_m"],
            filter_taps=obj_echo_cfg["filter_taps"],
            bandwidth_khz=obj_echo_cfg["bandwidth_khz"],
            smoothing_frac=obj_echo_cfg["smoothing_frac"],
            snr_db=obj_echo_cfg["snr_db"],
            clutter_fraction=obj_echo_cfg["clutter_fraction"],
            num_clutter_paths=obj_echo_cfg["num_clutter_paths"],
            clutter_min_distance_m=obj_echo_cfg["clutter_min_distance_m"],
            clutter_max_distance_m=obj_echo_cfg["clutter_max_distance_m"],
            clutter_reflectivity_range=obj_echo_cfg["clutter_reflectivity_range"],
            clutter_roughness_range=obj_echo_cfg["clutter_roughness_range"],
            clutter_texture_scale_range_m=obj_echo_cfg["clutter_texture_scale_range_m"],
            device=self.device,
        )
        true_delays = [int(true_delay)]

        # --- SNN forward pass and population decode.
        bank_spikes = self.net.forward(pulse_spikes, echo_spikes)
        spike_counts_np = bank_spikes.sum(dim=0)[0].detach().cpu().numpy()
        decoded_objects = self.net.decode_population_response(
            spike_counts_np,
            min_peak_spikes=self.detect_cfg["min_peak_spikes"],
            min_object_spikes=self.detect_cfg["min_object_spikes"],
        )
        snn_delays = [float(obj.delay_steps) for obj in decoded_objects]
        snn_metrics = self._match_estimates(
            true_delays=true_delays,
            estimated_delays=snn_delays,
            tolerance_steps=tolerance_steps,
        )

        # --- Matched-filter pass on the same waveforms.
        pulse_np = pulse_waveform.detach().cpu().numpy().astype(np.float64)
        echo_np = echo_waveform.detach().cpu().numpy().astype(np.float64)

        # Active-pulse slice from the framework's pulse configuration.
        step_duration_s = self.sim_cfg["step_duration_ms"] * 1e-3
        f_s = float(self.sim_cfg["sample_rate_hz"])
        pulse_start_s = float(self.pulse_cfg["pulse_start_time"]) * step_duration_s
        pulse_dur_s = float(self.pulse_cfg["duration"]) * step_duration_s
        pa_lo = int(round(pulse_start_s * f_s))
        pa_hi = int(round((pulse_start_s + pulse_dur_s) * f_s))
        pa_hi = min(pa_hi, pulse_np.size)

        mf_delays_steps, mf_distances_m, _ = matched_filter_detect(
            pulse_waveform=pulse_np,
            echo_waveform=echo_np,
            pulse_active_slice=(pa_lo, pa_hi),
            sample_rate_hz=f_s,
            speed_of_sound_m_s=float(self.echo_default_cfg["speed_of_sound_m_s"]),
            step_duration_s=step_duration_s,
            min_object_distance_m=float(self.echo_default_cfg.get("min_distance_m", 0.15)),
            cfg=mf_cfg,
        )
        mf_metrics = self._match_estimates(
            true_delays=true_delays,
            estimated_delays=mf_delays_steps,
            tolerance_steps=tolerance_steps,
        )

        true_distance_m = self._delay_steps_to_distance_m(float(true_delay))

        return {
            "true_distance_m": float(true_distance_m),
            "true_delay": int(true_delay),
            "snn_decoded_delays": snn_delays,
            "snn_misses": snn_metrics.misses,
            "snn_false_positives": snn_metrics.false_positives,
            "snn_abs_distance_errors_m": snn_metrics.abs_distance_errors_m,
            "mf_decoded_delays": mf_delays_steps,
            "mf_decoded_distances_m": mf_distances_m,
            "mf_misses": mf_metrics.misses,
            "mf_false_positives": mf_metrics.false_positives,
            "mf_abs_distance_errors_m": mf_metrics.abs_distance_errors_m,
        }


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------


@dataclass
class SnrSummary:
    snr_db: float
    num_trials: int

    snn_mae_mm: float
    snn_mae_ci_half_mm: float
    snn_miss_rate: float
    snn_miss_ci_half: float
    snn_n_matched: int

    mf_mae_mm: float
    mf_mae_ci_half_mm: float
    mf_miss_rate: float
    mf_miss_ci_half: float
    mf_n_matched: int


def wilson_half_width(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return float("nan")
    p_hat = k / n
    denom = 1.0 + z**2 / n
    return float(
        z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    )


def _summarise_estimator(
    trial_rows: List[dict],
    miss_key: str,
    err_key: str,
) -> Tuple[float, float, float, float, int]:
    """Return (mae_mm, mae_ci_half_mm, miss_rate, miss_ci_half, n_matched)."""
    n = len(trial_rows)
    misses = sum(row[miss_key] for row in trial_rows)
    errors_mm = [
        1000.0 * e for row in trial_rows for e in row[err_key]
    ]
    miss_rate = misses / n if n else float("nan")
    miss_ci_half = wilson_half_width(misses, n)
    if errors_mm:
        mae_mm = float(np.mean(errors_mm))
        sem_mm = float(np.std(errors_mm, ddof=1)) / np.sqrt(max(1, len(errors_mm)))
        mae_ci_half = 1.96 * sem_mm
    else:
        mae_mm = float("nan")
        mae_ci_half = float("nan")
    return mae_mm, mae_ci_half, miss_rate, miss_ci_half, len(errors_mm)


def summarise_snr(snr_db: float, trial_rows: List[dict]) -> SnrSummary:
    snn = _summarise_estimator(trial_rows, "snn_misses", "snn_abs_distance_errors_m")
    mf = _summarise_estimator(trial_rows, "mf_misses", "mf_abs_distance_errors_m")
    return SnrSummary(
        snr_db=snr_db,
        num_trials=len(trial_rows),
        snn_mae_mm=snn[0], snn_mae_ci_half_mm=snn[1],
        snn_miss_rate=snn[2], snn_miss_ci_half=snn[3], snn_n_matched=snn[4],
        mf_mae_mm=mf[0], mf_mae_ci_half_mm=mf[1],
        mf_miss_rate=mf[2], mf_miss_ci_half=mf[3], mf_n_matched=mf[4],
    )


# ---------------------------------------------------------------------------
# Sweep runner.
# ---------------------------------------------------------------------------


def run_snr_sweep(
    snr_db_list: Sequence[float],
    num_trials: int,
    target_distance_m: float,
    reflectivity_range: Tuple[float, float],
    roughness_range: Tuple[float, float],
    clutter_fraction: float,
    tolerance_steps: int,
    mf_cfg: MatchedFilterConfig,
    seed: int,
) -> Tuple[List[dict], List[SnrSummary]]:
    framework = PairedSnrSweepFramework(seed=seed)
    all_trial_rows: List[dict] = []
    summaries: List[SnrSummary] = []

    total = len(snr_db_list) * num_trials
    done = 0
    for snr_db in snr_db_list:
        trial_rows_for_snr: List[dict] = []
        for trial_idx in range(num_trials):
            row = framework.run_paired_snr_trial(
                distance_m=target_distance_m,
                reflectivity_range=reflectivity_range,
                roughness_range=roughness_range,
                snr_db=float(snr_db),
                clutter_fraction=clutter_fraction,
                tolerance_steps=tolerance_steps,
                mf_cfg=mf_cfg,
            )
            row["snr_db"] = float(snr_db)
            row["trial_index"] = trial_idx
            trial_rows_for_snr.append(row)
            all_trial_rows.append(row)

            done += 1
            if done % 25 == 0 or done == total:
                pct = 100.0 * done / total
                sys.stdout.write(
                    f"\r  [snr={snr_db:>6.1f} dB] "
                    f"trial {trial_idx + 1}/{num_trials}  "
                    f"(overall {done}/{total}, {pct:5.1f}%)"
                )
                sys.stdout.flush()

        summary = summarise_snr(float(snr_db), trial_rows_for_snr)
        summaries.append(summary)
        print(
            f"\n    SNN: MAE = {summary.snn_mae_mm:.2f} +/- "
            f"{summary.snn_mae_ci_half_mm:.2f} mm, "
            f"miss = {summary.snn_miss_rate:.3f}"
        )
        print(
            f"    MF : MAE = {summary.mf_mae_mm:.2f} +/- "
            f"{summary.mf_mae_ci_half_mm:.2f} mm, "
            f"miss = {summary.mf_miss_rate:.3f}"
        )

    return all_trial_rows, summaries


# ---------------------------------------------------------------------------
# CSV + figure writers.
# ---------------------------------------------------------------------------


def write_trial_csv(rows: List[dict], path: Path) -> None:
    fieldnames = [
        "snr_db", "trial_index", "true_distance_m", "true_delay",
        "snn_decoded_delays", "snn_misses", "snn_false_positives",
        "mf_decoded_delays", "mf_decoded_distances_m",
        "mf_misses", "mf_false_positives",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: (",".join(f"{x:g}" for x in row[k]) if isinstance(row[k], list) else row[k])
                for k in fieldnames
            })


def write_summary_csv(summaries: List[SnrSummary], path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "snr_db", "num_trials",
            "snn_mae_mm", "snn_mae_ci_half_mm",
            "snn_miss_rate", "snn_miss_ci_half", "snn_n_matched",
            "mf_mae_mm", "mf_mae_ci_half_mm",
            "mf_miss_rate", "mf_miss_ci_half", "mf_n_matched",
        ])
        for s in summaries:
            writer.writerow([
                s.snr_db, s.num_trials,
                s.snn_mae_mm, s.snn_mae_ci_half_mm,
                s.snn_miss_rate, s.snn_miss_ci_half, s.snn_n_matched,
                s.mf_mae_mm, s.mf_mae_ci_half_mm,
                s.mf_miss_rate, s.mf_miss_ci_half, s.mf_n_matched,
            ])


def plot_range_baseline(summaries: List[SnrSummary], path: Path) -> None:
    snrs = np.array([s.snr_db for s in summaries])
    snn_mae = np.array([s.snn_mae_mm for s in summaries])
    snn_ci = np.array([s.snn_mae_ci_half_mm for s in summaries])
    snn_miss = np.array([s.snn_miss_rate for s in summaries])
    mf_mae = np.array([s.mf_mae_mm for s in summaries])
    mf_ci = np.array([s.mf_mae_ci_half_mm for s in summaries])
    mf_miss = np.array([s.mf_miss_rate for s in summaries])

    fig, (ax_mae, ax_miss) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- Left: MAE vs SNR, both estimators -------------------------------
    ax_mae.errorbar(
        snrs, snn_mae, yerr=snn_ci, fmt="o-",
        label="SNN (population centroid)", color="#1f77b4", capsize=3,
    )
    ax_mae.errorbar(
        snrs, mf_mae, yerr=mf_ci, fmt="s-",
        label="Matched filter (parabolic)", color="#2ca02c", capsize=3,
    )
    ax_mae.set_xlabel("SNR (dB)")
    ax_mae.set_ylabel("Range MAE (mm)")
    ax_mae.set_title("Range accuracy: SNN vs matched filter")
    ax_mae.grid(alpha=0.3)
    ax_mae.legend(loc="upper right", fontsize=9)
    if np.nanmax(np.concatenate([snn_mae, mf_mae])) > 100:
        ax_mae.set_yscale("log")

    # --- Right: miss rate vs SNR for context -----------------------------
    ax_miss.plot(snrs, snn_miss, "o-", label="SNN miss rate", color="#1f77b4")
    ax_miss.plot(snrs, mf_miss, "s-", label="Matched-filter miss rate", color="#2ca02c")
    ax_miss.set_xlabel("SNR (dB)")
    ax_miss.set_ylabel("Miss rate")
    ax_miss.set_title("Detection miss rate")
    ax_miss.set_ylim(-0.05, 1.05)
    ax_miss.grid(alpha=0.3)
    ax_miss.legend(loc="upper right", fontsize=9)

    fig.suptitle("Range pathway: classical-baseline comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Range MAE vs SNR: SNN vs matched-filter baseline"
    )
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument(
        "--snrs", type=float, nargs="+",
        default=[-20, -15, -10, -5, 0, 5, 10, 15, 20],
        help="SNR values to sweep, in dB.",
    )
    parser.add_argument("--target-distance-m", type=float, default=1.15)
    parser.add_argument(
        "--reflectivity", type=float, nargs=2,
        default=(0.1, 0.8), metavar=("LO", "HI"),
    )
    parser.add_argument(
        "--roughness", type=float, nargs=2,
        default=(0.1, 0.35), metavar=("LO", "HI"),
    )
    parser.add_argument("--clutter-fraction", type=float, default=0.0)
    parser.add_argument("--tolerance-steps", type=int, default=10)
    parser.add_argument("--cfar-k", type=float, default=6.0,
                        help="CFAR threshold multiplier on sigma_noise.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/range_baseline"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mf_cfg = MatchedFilterConfig(cfar_k=args.cfar_k)

    trial_rows, summaries = run_snr_sweep(
        snr_db_list=args.snrs,
        num_trials=args.num_trials,
        target_distance_m=args.target_distance_m,
        reflectivity_range=tuple(args.reflectivity),
        roughness_range=tuple(args.roughness),
        clutter_fraction=args.clutter_fraction,
        tolerance_steps=args.tolerance_steps,
        mf_cfg=mf_cfg,
        seed=args.seed,
    )

    write_trial_csv(trial_rows, args.output_dir / "range_baseline_trials.csv")
    write_summary_csv(summaries, args.output_dir / "range_baseline_summary.csv")
    plot_range_baseline(summaries, args.output_dir / "range_baseline.png")

    # --- Console summary table -------------------------------------------
    print("\n" + "=" * 100)
    print("RANGE BASELINE SUMMARY (SNN vs matched filter)")
    print("=" * 100)
    print(
        f"{'snr_db':>7} | {'trials':>6} | "
        f"{'snn_mae_mm':>11} | {'snn_miss':>8} | "
        f"{'mf_mae_mm':>10} | {'mf_miss':>7}"
    )
    print("-" * 100)
    for s in summaries:
        print(
            f"{s.snr_db:>7.1f} | "
            f"{s.num_trials:>6} | "
            f"{s.snn_mae_mm:>5.2f} +/- {s.snn_mae_ci_half_mm:>3.2f} | "
            f"{s.snn_miss_rate:>8.3f} | "
            f"{s.mf_mae_mm:>4.2f} +/- {s.mf_mae_ci_half_mm:>3.2f} | "
            f"{s.mf_miss_rate:>7.3f}"
        )
    print("=" * 100)
    print(f"\nOutputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()