"""Two-object separation sweep for Section 5.3 (range resolution).

This script tests the FM-FM bank's ability to resolve two closely-spaced
objects, as the true separation between them is varied from 1 delay step
(the single-step quantisation limit) up to 20 delay steps (trivially
resolved). For each separation, ``num_trials`` independent trials are run;
each trial places the first object at a uniformly random anchor and the
second object at ``anchor + separation_steps * delta_r``. Object
reflectivity and roughness are drawn per trial from the same ranges used
in S1 (single-object accuracy).

For each separation, both the detection (active-neuron) read-out and the
population centroid read-out are evaluated against the ground truth using
the framework's existing matched-metric logic, with a tolerance of one
delay step. The primary metric is ``resolution probability``: the fraction
of trials in which exactly two true objects were matched, with no misses
and no false positives. The secondary metric is the per-object mean
absolute distance error on resolved trials, which exposes any accuracy
cost paid for resolution under lateral inhibition.

Outputs:
    - ``separation_sweep_trials.csv``  --- one row per trial
    - ``separation_sweep_summary.csv`` --- one row per separation point
    - ``separation_sweep.png``         --- two-panel figure
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_model_range import FMSweepEvaluationFramework


# ---------------------------------------------------------------------------
# Framework extension: separation-controlled scene simulation.
# ---------------------------------------------------------------------------


class SeparationSweepFramework(FMSweepEvaluationFramework):
    """Adds a two-object resolution probe with exact separation control."""

    def _simulate_pair_with_pulse(
        self,
        pulse_waveform: torch.Tensor,
        pulse_spikes: torch.Tensor,
        distances_m: Sequence[float],
        reflectivity_range: Tuple[float, float],
        roughness_range: Tuple[float, float],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[int]]:
        """Simulate a scene with the two object distances given explicitly.

        This is a thin variant of ``_simulate_scene_with_pulse`` that
        bypasses ``_sample_distances_m`` so we can place the two objects at
        an exact target separation, rather than using rejection sampling
        with only a separation floor.
        """
        from sim_modules.echo_range import generate_fm_echo_from_waveform

        object_echoes = []
        true_distances_m: List[float] = []
        true_delays: List[int] = []

        for distance_m in distances_m:
            obj_echo_cfg = dict(self.echo_default_cfg)
            obj_echo_cfg["distance_m"] = float(distance_m)
            obj_echo_cfg["reflectivity"] = float(self.rng.uniform(*reflectivity_range))
            obj_echo_cfg["surface_roughness"] = float(self.rng.uniform(*roughness_range))

            echo_i, _, true_delay_i = generate_fm_echo_from_waveform(
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
            object_echoes.append(echo_i)
            true_distances_m.append(float(distance_m))
            true_delays.append(int(true_delay_i))

        echo_spikes = torch.clamp(sum(object_echoes), 0, 1)
        order = np.argsort(true_distances_m)
        return (
            pulse_spikes,
            echo_spikes,
            [true_distances_m[i] for i in order],
            [true_delays[i] for i in order],
        )

    def run_pair_trial(
        self,
        anchor_distance_m: float,
        separation_steps: int,
        reflectivity_range: Tuple[float, float],
        roughness_range: Tuple[float, float],
        tolerance_steps: int,
    ) -> dict:
        """Run a single two-object trial with an exact target separation."""
        delta_r = self._delay_steps_to_distance_m(1.0)
        anchor_distance_m = float(anchor_distance_m)
        partner_distance_m = anchor_distance_m + separation_steps * delta_r

        pulse_waveform, pulse_spikes = self._build_pulse()
        pulse_spikes, echo_spikes, true_distances_m, true_delays = (
            self._simulate_pair_with_pulse(
                pulse_waveform=pulse_waveform,
                pulse_spikes=pulse_spikes,
                distances_m=[anchor_distance_m, partner_distance_m],
                reflectivity_range=reflectivity_range,
                roughness_range=roughness_range,
            )
        )

        bank_spikes = self.net.forward(pulse_spikes, echo_spikes)
        spike_counts = bank_spikes.sum(dim=0)
        spike_counts_np = spike_counts[0].detach().cpu().numpy()

        # --- Detection read-out: active neurons above min_spikes -----------
        active_indices = np.flatnonzero(
            spike_counts_np >= float(self.detect_cfg["min_spikes"])
        )
        detected_delays = [int(self.candidate_delays[idx]) for idx in active_indices]

        # --- Population read-out: centroid-decoded objects -----------------
        decoded_objects = self.net.decode_population_response(
            spike_counts_np,
            min_peak_spikes=self.detect_cfg["min_peak_spikes"],
            min_object_spikes=self.detect_cfg["min_object_spikes"],
        )
        decoded_delays = [float(obj.delay_steps) for obj in decoded_objects]

        det_metrics = self._match_estimates(
            true_delays=true_delays,
            estimated_delays=detected_delays,
            tolerance_steps=tolerance_steps,
            allow_multiple_estimates_per_true=bool(
                self.detect_cfg.get("allow_multiple_estimates_per_true", False)
            ),
        )
        pop_metrics = self._match_estimates(
            true_delays=true_delays,
            estimated_delays=decoded_delays,
            tolerance_steps=tolerance_steps,
        )

        det_resolved = (
            det_metrics.misses == 0
            and det_metrics.false_positives == 0
            and len(detected_delays) == 2
        )
        pop_resolved = (
            pop_metrics.misses == 0
            and pop_metrics.false_positives == 0
            and len(decoded_delays) == 2
        )

        return {
            "anchor_distance_m": anchor_distance_m,
            "partner_distance_m": partner_distance_m,
            "true_distances_m": true_distances_m,
            "true_delays": true_delays,
            "detected_delays": sorted(detected_delays),
            "decoded_delays": sorted(decoded_delays),
            "det_misses": det_metrics.misses,
            "det_false_positives": det_metrics.false_positives,
            "pop_misses": pop_metrics.misses,
            "pop_false_positives": pop_metrics.false_positives,
            "pop_abs_distance_errors_m": pop_metrics.abs_distance_errors_m,
            "det_resolved": int(det_resolved),
            "pop_resolved": int(pop_resolved),
        }


# ---------------------------------------------------------------------------
# Aggregation helpers.
# ---------------------------------------------------------------------------


@dataclass
class SeparationSummary:
    separation_steps: int
    separation_mm: float
    num_trials: int
    det_resolution_prob: float
    pop_resolution_prob: float
    det_resolution_ci_half: float
    pop_resolution_ci_half: float
    pop_mae_resolved_mm: float
    pop_mae_resolved_ci_half_mm: float
    n_resolved_pop: int


def wilson_half_width(k: int, n: int, z: float = 1.96) -> float:
    """Half-width of the Wilson 95% CI for a binomial proportion."""
    if n <= 0:
        return float("nan")
    p_hat = k / n
    denom = 1.0 + z**2 / n
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return float(half)


def summarise_separation(
    separation_steps: int,
    delta_r_mm: float,
    trial_rows: List[dict],
) -> SeparationSummary:
    n = len(trial_rows)
    det_k = sum(row["det_resolved"] for row in trial_rows)
    pop_k = sum(row["pop_resolved"] for row in trial_rows)

    # Per-object MAE on trials where the population decoder resolved both
    # objects; mixing in unresolved trials would conflate resolution and
    # accuracy, so we don't.
    pop_resolved_errors_mm = [
        1000.0 * e
        for row in trial_rows
        if row["pop_resolved"]
        for e in row["pop_abs_distance_errors_m"]
    ]
    if pop_resolved_errors_mm:
        mae_mm = float(np.mean(pop_resolved_errors_mm))
        sem_mm = float(np.std(pop_resolved_errors_mm, ddof=1)) / np.sqrt(
            max(1, len(pop_resolved_errors_mm))
        )
        mae_ci_half = 1.96 * sem_mm
    else:
        mae_mm = float("nan")
        mae_ci_half = float("nan")

    return SeparationSummary(
        separation_steps=separation_steps,
        separation_mm=separation_steps * delta_r_mm,
        num_trials=n,
        det_resolution_prob=det_k / n if n else float("nan"),
        pop_resolution_prob=pop_k / n if n else float("nan"),
        det_resolution_ci_half=wilson_half_width(det_k, n),
        pop_resolution_ci_half=wilson_half_width(pop_k, n),
        pop_mae_resolved_mm=mae_mm,
        pop_mae_resolved_ci_half_mm=mae_ci_half,
        n_resolved_pop=len(pop_resolved_errors_mm) // 2,
    )


# ---------------------------------------------------------------------------
# Sweep runner.
# ---------------------------------------------------------------------------


def run_separation_sweep(
    separation_steps_list: Sequence[int],
    num_trials: int,
    anchor_range_m: Tuple[float, float],
    reflectivity_range: Tuple[float, float],
    roughness_range: Tuple[float, float],
    tolerance_steps: int,
    seed: int,
    output_dir: Path,
) -> Tuple[List[dict], List[SeparationSummary]]:
    framework = SeparationSweepFramework(seed=seed)
    delta_r_mm = 1000.0 * framework._delay_steps_to_distance_m(1.0)
    print(
        f"Single-step range resolution delta_r = {delta_r_mm:.3f} mm; "
        f"sweep separations (steps): {list(separation_steps_list)}"
    )

    all_trial_rows: List[dict] = []
    summaries: List[SeparationSummary] = []

    total_trials = len(separation_steps_list) * num_trials
    trials_done = 0

    for sep_steps in separation_steps_list:
        sep_mm = sep_steps * delta_r_mm
        trial_rows_for_sep: List[dict] = []

        # Restrict the anchor window so the partner lies inside the
        # operating range with margin equal to one tolerance band.
        max_anchor = anchor_range_m[1] - sep_steps * (delta_r_mm / 1000.0)
        max_anchor = max(anchor_range_m[0], max_anchor)

        for trial_idx in range(num_trials):
            anchor_distance_m = float(
                framework.rng.uniform(anchor_range_m[0], max_anchor)
            )
            row = framework.run_pair_trial(
                anchor_distance_m=anchor_distance_m,
                separation_steps=int(sep_steps),
                reflectivity_range=reflectivity_range,
                roughness_range=roughness_range,
                tolerance_steps=tolerance_steps,
            )
            row["separation_steps"] = int(sep_steps)
            row["separation_mm"] = float(sep_mm)
            row["trial_index"] = trial_idx
            trial_rows_for_sep.append(row)
            all_trial_rows.append(row)

            trials_done += 1
            if trials_done % 25 == 0 or trials_done == total_trials:
                pct = 100.0 * trials_done / total_trials
                sys.stdout.write(
                    f"\r  [sep={sep_steps:>2} steps / {sep_mm:6.2f} mm] "
                    f"trial {trial_idx + 1}/{num_trials}  "
                    f"(overall {trials_done}/{total_trials}, {pct:5.1f}%)"
                )
                sys.stdout.flush()

        summary = summarise_separation(
            separation_steps=int(sep_steps),
            delta_r_mm=delta_r_mm,
            trial_rows=trial_rows_for_sep,
        )
        summaries.append(summary)
        print(
            f"\n    detection P(resolve)  = {summary.det_resolution_prob:.3f} "
            f"+/- {summary.det_resolution_ci_half:.3f}"
        )
        print(
            f"    population P(resolve) = {summary.pop_resolution_prob:.3f} "
            f"+/- {summary.pop_resolution_ci_half:.3f}"
        )
        if not np.isnan(summary.pop_mae_resolved_mm):
            print(
                f"    pop MAE on resolved   = {summary.pop_mae_resolved_mm:.2f} mm "
                f"+/- {summary.pop_mae_resolved_ci_half_mm:.2f} mm"
            )

    return all_trial_rows, summaries


# ---------------------------------------------------------------------------
# CSV + figure writers.
# ---------------------------------------------------------------------------


def write_trial_csv(rows: List[dict], path: Path) -> None:
    fieldnames = [
        "separation_steps",
        "separation_mm",
        "trial_index",
        "anchor_distance_m",
        "partner_distance_m",
        "true_delays",
        "detected_delays",
        "decoded_delays",
        "det_misses",
        "det_false_positives",
        "pop_misses",
        "pop_false_positives",
        "det_resolved",
        "pop_resolved",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (
                        ",".join(str(x) for x in row[k])
                        if isinstance(row[k], list)
                        else row[k]
                    )
                    for k in fieldnames
                }
            )


def write_summary_csv(summaries: List[SeparationSummary], path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "separation_steps",
                "separation_mm",
                "num_trials",
                "det_resolution_prob",
                "det_resolution_ci_half",
                "pop_resolution_prob",
                "pop_resolution_ci_half",
                "pop_mae_resolved_mm",
                "pop_mae_resolved_ci_half_mm",
                "n_resolved_pop",
            ]
        )
        for s in summaries:
            writer.writerow(
                [
                    s.separation_steps,
                    s.separation_mm,
                    s.num_trials,
                    s.det_resolution_prob,
                    s.det_resolution_ci_half,
                    s.pop_resolution_prob,
                    s.pop_resolution_ci_half,
                    s.pop_mae_resolved_mm,
                    s.pop_mae_resolved_ci_half_mm,
                    s.n_resolved_pop,
                ]
            )


def plot_separation_sweep(
    summaries: List[SeparationSummary],
    delta_r_mm: float,
    classical_limit_mm: Optional[float],
    path: Path,
) -> None:
    seps_mm = np.array([s.separation_mm for s in summaries])
    seps_steps = np.array([s.separation_steps for s in summaries])
    det_p = np.array([s.det_resolution_prob for s in summaries])
    pop_p = np.array([s.pop_resolution_prob for s in summaries])
    det_ci = np.array([s.det_resolution_ci_half for s in summaries])
    pop_ci = np.array([s.pop_resolution_ci_half for s in summaries])
    pop_mae = np.array([s.pop_mae_resolved_mm for s in summaries])
    pop_mae_ci = np.array([s.pop_mae_resolved_ci_half_mm for s in summaries])

    fig, (ax_p, ax_e) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- Left: resolution probability vs separation ----------------------
    ax_p.errorbar(
        seps_mm, det_p, yerr=det_ci, fmt="o-",
        label="Detection (active neurons)", color="#1f77b4", capsize=3,
    )
    ax_p.errorbar(
        seps_mm, pop_p, yerr=pop_ci, fmt="s-",
        label="Population (centroid)", color="#d62728", capsize=3,
    )
    ax_p.axhline(0.5, color="k", linewidth=0.6, linestyle=":", alpha=0.5)
    ax_p.axvline(
        delta_r_mm, color="grey", linewidth=0.8, linestyle="--", alpha=0.7,
        label=f"Single delay step ({delta_r_mm:.1f} mm)",
    )
    if classical_limit_mm is not None and classical_limit_mm > 0:
        ax_p.axvline(
            classical_limit_mm, color="green", linewidth=0.8, linestyle="--",
            alpha=0.7, label=f"Classical limit c/2B ({classical_limit_mm:.1f} mm)",
        )
    ax_p.set_xlabel("True separation (mm)")
    ax_p.set_ylabel("P(both objects resolved)")
    ax_p.set_title("Resolution probability vs separation")
    ax_p.set_ylim(-0.05, 1.05)
    ax_p.grid(alpha=0.3)
    ax_p.legend(loc="lower right", fontsize=9)

    # Secondary x-axis showing separation in delay steps.
    ax_p_top = ax_p.secondary_xaxis(
        "top",
        functions=(lambda mm: mm / delta_r_mm, lambda st: st * delta_r_mm),
    )
    ax_p_top.set_xlabel("Separation (delay steps)")

    # --- Right: per-object MAE on resolved trials ------------------------
    ax_e.errorbar(
        seps_mm, pop_mae, yerr=pop_mae_ci, fmt="s-",
        color="#d62728", capsize=3, label="Population MAE on resolved trials",
    )
    ax_e.set_xlabel("True separation (mm)")
    ax_e.set_ylabel("Per-object MAE (mm)")
    ax_e.set_title("Accuracy on resolved trials")
    ax_e.grid(alpha=0.3)
    ax_e.legend(loc="upper right", fontsize=9)

    ax_e_top = ax_e.secondary_xaxis(
        "top",
        functions=(lambda mm: mm / delta_r_mm, lambda st: st * delta_r_mm),
    )
    ax_e_top.set_xlabel("Separation (delay steps)")

    fig.suptitle("Two-Object Range Resolution Sweep", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-object separation sweep")
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument(
        "--separations",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 7, 10, 15, 20],
        help="Separations to sweep, in delay steps.",
    )
    parser.add_argument("--anchor-min-m", type=float, default=0.5)
    parser.add_argument("--anchor-max-m", type=float, default=2.5)
    parser.add_argument(
        "--reflectivity",
        type=float,
        nargs=2,
        default=(0.1, 0.8),
        metavar=("LO", "HI"),
    )
    parser.add_argument(
        "--roughness",
        type=float,
        nargs=2,
        default=(0.1, 0.35),
        metavar=("LO", "HI"),
    )
    parser.add_argument("--tolerance-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classical-limit-mm",
        type=float,
        default=2.86,
        help="Classical c/(2B) limit, for the dashed reference line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/separation_sweep"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    trial_rows, summaries = run_separation_sweep(
        separation_steps_list=args.separations,
        num_trials=args.num_trials,
        anchor_range_m=(args.anchor_min_m, args.anchor_max_m),
        reflectivity_range=tuple(args.reflectivity),
        roughness_range=tuple(args.roughness),
        tolerance_steps=args.tolerance_steps,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    write_trial_csv(trial_rows, args.output_dir / "separation_sweep_trials.csv")
    write_summary_csv(summaries, args.output_dir / "separation_sweep_summary.csv")

    framework = SeparationSweepFramework(seed=args.seed)
    delta_r_mm = 1000.0 * framework._delay_steps_to_distance_m(1.0)
    plot_separation_sweep(
        summaries=summaries,
        delta_r_mm=delta_r_mm,
        classical_limit_mm=args.classical_limit_mm,
        path=args.output_dir / "separation_sweep.png",
    )

    # --- Console summary table -------------------------------------------
    print("\n" + "=" * 90)
    print("SEPARATION SWEEP SUMMARY")
    print("=" * 90)
    print(
        f"{'sep_steps':>9} | {'sep_mm':>7} | {'trials':>6} | "
        f"{'det_P(res)':>11} | {'pop_P(res)':>11} | "
        f"{'pop_MAE_mm':>10}"
    )
    print("-" * 90)
    for s in summaries:
        mae_str = (
            f"{s.pop_mae_resolved_mm:>6.2f}"
            if not np.isnan(s.pop_mae_resolved_mm)
            else "   n/a"
        )
        print(
            f"{s.separation_steps:>9} | "
            f"{s.separation_mm:>7.2f} | "
            f"{s.num_trials:>6} | "
            f"{s.det_resolution_prob:>5.3f} +/- {s.det_resolution_ci_half:>3.3f} | "
            f"{s.pop_resolution_prob:>5.3f} +/- {s.pop_resolution_ci_half:>3.3f} | "
            f"{mae_str}"
        )
    print("=" * 90)
    print(f"\nOutputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()