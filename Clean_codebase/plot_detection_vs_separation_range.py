from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import csv
import os

MPL_CACHE_DIR = Path(__file__).with_name(".mplcache")
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from grid_search_evaluation_range import FMSweepGridSearchFramework
from sim_modules.echo_range import generate_fm_echo_from_waveform
from sim_modules.network_range import FMSweepBankNetwork, WTAReadoutLayer


@dataclass
class SeparationSweepRow:
    separation_cm: int
    separation_m: float
    separation_steps: float
    true_distances_m: List[float]
    true_delays: List[float]
    rounded_true_delays: List[int]
    true_delay_gap_steps: float
    trials: int
    resolved_probability: float
    exact_detection_probability: float
    detection_error_probability: float
    miss_rate: float
    false_positive_rate: float
    false_positive_trial_probability: float
    mean_num_detections: float


EXPERIMENT_CFG = {
    "r1_m": 1.50,
    "separation_cm_min": 1,
    "separation_cm_max": 50,
    "trials_per_separation": 64,
    "trials_per_batch": 32,
    "reflectivity": 1.0,
    "surface_roughness": 0.0,
    "surface_texture_scale_m": 0.0,
    "air_absorption_db_per_m_100khz": 0.0,
    "snr_db": None,
    "clutter_fraction": 0.0,
    "num_clutter_paths": 0,
    "tolerance_steps": 1.0,
    "seed": 421,
}

NETWORK_CFG = {
    "weight": 0.1,
    "beta": 0.1,
    "threshold": 0.7,
    "rs": 2,
    "lis": 3,
    "lir": 8,
    "inhib_decay": 0.1,
}

READOUT_CFG = {
    "beta": 0.4,
    "threshold": 0.4,
    "is": 0,
    "ir": 4,
    "inhibition_falloff": "gaussian",
    "sigma": 2.0,
}

OUTPUT_FIG_PATH = Path(__file__).with_name("fig_detection_vs_separation_range.png")
OUTPUT_CSV_PATH = Path(__file__).with_name("detection_vs_separation_range.csv")

BLUE = "#3B6CA8"
LIGHT_RED = "#D86B6B"
GREEN = "#3E8F65"
DARK_GREY = "#333333"


def separation_m_to_steps(framework: FMSweepGridSearchFramework, separation_m: float) -> float:
    step_duration_s = framework.sim_cfg["step_duration_ms"] * 1e-3
    speed = framework.echo_default_cfg["speed_of_sound_m_s"]
    return (2.0 * float(separation_m) / speed) / step_duration_s


def distance_m_to_delay_steps(framework: FMSweepGridSearchFramework, distance_m: float) -> float:
    step_duration_s = framework.sim_cfg["step_duration_ms"] * 1e-3
    speed = framework.echo_default_cfg["speed_of_sound_m_s"]
    return (2.0 * float(distance_m) / speed) / step_duration_s


def build_clean_echo_cfg(framework: FMSweepGridSearchFramework, cfg: Dict[str, object]) -> Dict[str, object]:
    echo_cfg = dict(framework.echo_default_cfg)
    echo_cfg.update(
        {
            "reflectivity": cfg["reflectivity"],
            "surface_roughness": cfg["surface_roughness"],
            "surface_texture_scale_m": cfg["surface_texture_scale_m"],
            "air_absorption_db_per_m_100khz": cfg["air_absorption_db_per_m_100khz"],
            "snr_db": cfg["snr_db"],
            "clutter_fraction": cfg["clutter_fraction"],
            "num_clutter_paths": cfg["num_clutter_paths"],
        }
    )
    return echo_cfg


def compute_echo_probabilities(
    framework: FMSweepGridSearchFramework,
    pulse_waveform: torch.Tensor,
    distance_m: float,
    echo_cfg: Dict[str, object],
) -> Tuple[torch.Tensor, int]:
    _, echo_waveform, true_delay = generate_fm_echo_from_waveform(
        pulse_waveform=pulse_waveform,
        encoder=framework.encoder,
        num_steps=framework.sim_cfg["num_steps"],
        distance_m=distance_m,
        reflectivity=echo_cfg["reflectivity"],
        max_spike_prob=echo_cfg["max_spike_prob"],
        sample_rate_hz=framework.sim_cfg["sample_rate_hz"],
        step_duration_ms=framework.sim_cfg["step_duration_ms"],
        speed_of_sound_m_s=echo_cfg["speed_of_sound_m_s"],
        air_absorption_db_per_m_100khz=echo_cfg["air_absorption_db_per_m_100khz"],
        surface_roughness=echo_cfg["surface_roughness"],
        surface_texture_scale_m=echo_cfg["surface_texture_scale_m"],
        filter_taps=echo_cfg["filter_taps"],
        bandwidth_khz=echo_cfg["bandwidth_khz"],
        smoothing_frac=echo_cfg["smoothing_frac"],
        snr_db=echo_cfg["snr_db"],
        clutter_fraction=echo_cfg["clutter_fraction"],
        num_clutter_paths=echo_cfg["num_clutter_paths"],
        clutter_min_distance_m=echo_cfg["clutter_min_distance_m"],
        clutter_max_distance_m=echo_cfg["clutter_max_distance_m"],
        clutter_reflectivity_range=echo_cfg["clutter_reflectivity_range"],
        clutter_roughness_range=echo_cfg["clutter_roughness_range"],
        clutter_texture_scale_range_m=echo_cfg["clutter_texture_scale_range_m"],
        device=framework.device,
    )

    echo_probs = framework.encoder.compute_spike_probabilities_from_waveform(
        waveform=echo_waveform,
        num_steps=framework.sim_cfg["num_steps"],
        max_spike_prob=echo_cfg["max_spike_prob"],
        sample_rate_hz=framework.sim_cfg["sample_rate_hz"],
        step_duration_ms=framework.sim_cfg["step_duration_ms"],
        filter_taps=echo_cfg["filter_taps"],
        bandwidth_khz=echo_cfg["bandwidth_khz"],
        smoothing_frac=echo_cfg["smoothing_frac"],
        device=framework.device,
    )
    return echo_probs, int(true_delay)


def build_fixed_scene_batch(
    framework: FMSweepGridSearchFramework,
    distances_m: Sequence[float],
    trials: int,
    echo_cfg: Dict[str, object],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    pulse_waveform, pulse_spike_probs = framework._get_pulse_template()
    pulse_spikes = framework.encoder.sample_spikes_from_probabilities(
        spike_probs=pulse_spike_probs,
        batch_size=trials,
        device=framework.device,
    )

    object_echo_batches = []
    true_delays = []
    for distance_m in distances_m:
        echo_probs, true_delay = compute_echo_probabilities(
            framework=framework,
            pulse_waveform=pulse_waveform,
            distance_m=float(distance_m),
            echo_cfg=echo_cfg,
        )
        object_echo_batches.append(
            framework.encoder.sample_spikes_from_probabilities(
                spike_probs=echo_probs,
                batch_size=trials,
                device=framework.device,
            )
        )
        true_delays.append(true_delay)

    echo_spikes = torch.clamp(sum(object_echo_batches), 0, 1)
    return pulse_spikes, echo_spikes, sorted(true_delays)


def evaluate_detection_batch(
    framework: FMSweepGridSearchFramework,
    net: FMSweepBankNetwork,
    readout: WTAReadoutLayer,
    pulse_spikes: torch.Tensor,
    echo_spikes: torch.Tensor,
    true_delays: Sequence[float],
    tolerance_steps: float,
) -> Dict[str, float]:
    bank_spikes = net.forward(pulse_spikes, echo_spikes)
    readout_spikes = readout(bank_spikes)
    spike_counts = readout_spikes.sum(dim=0)
    if spike_counts.ndim != 2:
        raise ValueError("Expected spike counts with shape [batch, neurons].")

    spike_counts_np = spike_counts.detach().cpu().numpy()
    trials = spike_counts_np.shape[0]
    resolved_trials = 0
    exact_trials = 0
    false_positive_trials = 0
    total_misses = 0
    total_false_positives = 0
    total_detections = 0

    for counts in spike_counts_np:
        # Peak-decoded object hypotheses expose fusion at small separations; raw
        # active bins can overstate resolution because one broad response may
        # light up several neighboring delay neurons.
        decoded_objects = net.decode_population_response(
            counts,
            min_peak_spikes=framework.detect_cfg["min_peak_spikes"],
            min_object_spikes=framework.detect_cfg["min_object_spikes"],
        )
        detected_delays = [float(obj.delay_steps) for obj in decoded_objects]

        metrics = framework._match_estimates(
            true_delays=true_delays,
            estimated_delays=detected_delays,
            tolerance_steps=tolerance_steps,
            allow_multiple_estimates_per_true=False,
        )

        total_misses += metrics.misses
        total_false_positives += metrics.false_positives
        total_detections += len(detected_delays)
        if metrics.misses == 0:
            resolved_trials += 1
        if metrics.misses == 0 and metrics.false_positives == 0:
            exact_trials += 1
        if metrics.false_positives > 0:
            false_positive_trials += 1

    total_true = max(1, len(true_delays) * trials)
    return {
        "resolved_probability": resolved_trials / float(trials),
        "exact_detection_probability": exact_trials / float(trials),
        "detection_error_probability": 1.0 - exact_trials / float(trials),
        "miss_rate": total_misses / float(total_true),
        "false_positive_rate": (
            total_false_positives / float(total_detections) if total_detections > 0 else 0.0
        ),
        "false_positive_trial_probability": false_positive_trials / float(trials),
        "mean_num_detections": total_detections / float(trials),
    }


def run_separation_sweep(
    cfg: Dict[str, object] = EXPERIMENT_CFG,
) -> Tuple[List[SeparationSweepRow], FMSweepGridSearchFramework]:
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    framework = FMSweepGridSearchFramework(seed=seed)
    echo_cfg = build_clean_echo_cfg(framework=framework, cfg=cfg)
    net = framework._build_network(NETWORK_CFG)
    readout = framework._build_readout(READOUT_CFG)

    rows: List[SeparationSweepRow] = []
    trials_per_separation = int(cfg["trials_per_separation"])
    trials_per_batch = max(1, int(cfg["trials_per_batch"]))
    separation_values_cm = range(
        int(cfg["separation_cm_min"]),
        int(cfg["separation_cm_max"]) + 1,
    )

    for separation_cm in separation_values_cm:
        separation_m = separation_cm / 100.0
        distances_m = [float(cfg["r1_m"]), float(cfg["r1_m"]) + separation_m]
        accumulated = {
            "resolved_probability": 0.0,
            "exact_detection_probability": 0.0,
            "detection_error_probability": 0.0,
            "miss_rate": 0.0,
            "false_positive_rate": 0.0,
            "false_positive_trial_probability": 0.0,
            "mean_num_detections": 0.0,
        }
        completed_trials = 0
        true_delays = sorted(
            distance_m_to_delay_steps(framework=framework, distance_m=distance_m)
            for distance_m in distances_m
        )
        rounded_true_delays: List[int] = []

        for batch_start in range(0, trials_per_separation, trials_per_batch):
            batch_trials = min(trials_per_batch, trials_per_separation - batch_start)
            pulse_spikes, echo_spikes, batch_rounded_true_delays = build_fixed_scene_batch(
                framework=framework,
                distances_m=distances_m,
                trials=batch_trials,
                echo_cfg=echo_cfg,
            )
            rounded_true_delays = batch_rounded_true_delays
            batch_stats = evaluate_detection_batch(
                framework=framework,
                net=net,
                readout=readout,
                pulse_spikes=pulse_spikes,
                echo_spikes=echo_spikes,
                true_delays=true_delays,
                tolerance_steps=float(cfg["tolerance_steps"]),
            )

            for key, value in batch_stats.items():
                accumulated[key] += value * batch_trials
            completed_trials += batch_trials

        averaged = {
            key: value / float(completed_trials)
            for key, value in accumulated.items()
        }
        rows.append(
            SeparationSweepRow(
                separation_cm=int(separation_cm),
                separation_m=separation_m,
                separation_steps=separation_m_to_steps(framework, separation_m),
                true_distances_m=distances_m,
                true_delays=sorted(true_delays),
                rounded_true_delays=sorted(rounded_true_delays),
                true_delay_gap_steps=float(max(true_delays) - min(true_delays)),
                trials=completed_trials,
                **averaged,
            )
        )

    return rows, framework


def save_rows_csv(rows: Sequence[SeparationSweepRow], output_path: Path = OUTPUT_CSV_PATH) -> None:
    fieldnames = [
        "separation_cm",
        "separation_m",
        "separation_steps",
        "r1_m",
        "r2_m",
        "true_delay_1",
        "true_delay_2",
        "rounded_true_delay_1",
        "rounded_true_delay_2",
        "true_delay_gap_steps",
        "trials",
        "resolved_probability",
        "exact_detection_probability",
        "detection_error_probability",
        "miss_rate",
        "false_positive_rate",
        "false_positive_trial_probability",
        "mean_num_detections",
    ]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "separation_cm": row.separation_cm,
                    "separation_m": row.separation_m,
                    "separation_steps": row.separation_steps,
                    "r1_m": row.true_distances_m[0],
                    "r2_m": row.true_distances_m[1],
                    "true_delay_1": row.true_delays[0],
                    "true_delay_2": row.true_delays[1],
                    "rounded_true_delay_1": row.rounded_true_delays[0],
                    "rounded_true_delay_2": row.rounded_true_delays[1],
                    "true_delay_gap_steps": row.true_delay_gap_steps,
                    "trials": row.trials,
                    "resolved_probability": row.resolved_probability,
                    "exact_detection_probability": row.exact_detection_probability,
                    "detection_error_probability": row.detection_error_probability,
                    "miss_rate": row.miss_rate,
                    "false_positive_rate": row.false_positive_rate,
                    "false_positive_trial_probability": row.false_positive_trial_probability,
                    "mean_num_detections": row.mean_num_detections,
                }
            )


def first_crossing(rows: Sequence[SeparationSweepRow], threshold: float) -> SeparationSweepRow | None:
    for row in rows:
        if row.resolved_probability >= threshold:
            return row
    return None


def add_bank_step_axis(ax: plt.Axes, framework: FMSweepGridSearchFramework) -> None:
    step_duration_s = framework.sim_cfg["step_duration_ms"] * 1e-3
    speed = framework.echo_default_cfg["speed_of_sound_m_s"]

    def cm_to_steps(cm_values):
        return np.asarray(cm_values) * 0.01 * 2.0 / speed / step_duration_s

    def steps_to_cm(step_values):
        return np.asarray(step_values) * speed * step_duration_s * 0.5 * 100.0

    top_ax = ax.secondary_xaxis("top", functions=(cm_to_steps, steps_to_cm))
    top_ax.set_xlabel("Separation in bank steps")


def plot_sweep(
    rows: Sequence[SeparationSweepRow],
    framework: FMSweepGridSearchFramework,
    output_path: Path = OUTPUT_FIG_PATH,
) -> None:
    if not rows:
        raise RuntimeError("No separation sweep rows were generated.")

    separation_cm = np.asarray([row.separation_cm for row in rows], dtype=float)
    resolved = np.asarray([row.resolved_probability for row in rows], dtype=float)
    exact = np.asarray([row.exact_detection_probability for row in rows], dtype=float)
    miss_rate = np.asarray([row.miss_rate for row in rows], dtype=float)
    fp_trial = np.asarray([row.false_positive_trial_probability for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True)
    resolution_ax, error_ax = axes

    resolution_ax.plot(
        separation_cm,
        resolved,
        marker="o",
        markersize=4,
        linewidth=2,
        color=BLUE,
        label="Resolved both targets",
    )
    resolution_ax.plot(
        separation_cm,
        exact,
        marker="s",
        markersize=3.5,
        linewidth=1.7,
        color=GREEN,
        label="Resolved with no extras",
    )
    resolution_ax.axhline(0.95, color=DARK_GREY, linestyle="--", linewidth=1.0, alpha=0.75)

    limit_95 = first_crossing(rows, threshold=0.95)
    if limit_95 is not None:
        resolution_ax.axvline(limit_95.separation_cm, color=BLUE, linestyle=":", linewidth=1.2)
        resolution_ax.annotate(
            f"95% at {limit_95.separation_cm} cm",
            xy=(limit_95.separation_cm, 0.95),
            xytext=(8, -22),
            textcoords="offset points",
            fontsize=9,
            color=BLUE,
        )

    resolution_ax.set_title("Two-Object Resolution")
    resolution_ax.set_xlabel("Range separation, Delta r (cm)")
    resolution_ax.set_ylabel("Probability")
    resolution_ax.set_ylim(-0.03, 1.03)
    resolution_ax.set_xlim(separation_cm.min(), separation_cm.max())
    resolution_ax.grid(True, alpha=0.3)
    resolution_ax.legend(frameon=False, loc="lower right")
    add_bank_step_axis(resolution_ax, framework)

    error_ax.plot(
        separation_cm,
        miss_rate,
        marker="o",
        markersize=4,
        linewidth=2,
        color=LIGHT_RED,
        label="Miss rate per object",
    )
    error_ax.plot(
        separation_cm,
        fp_trial,
        marker="s",
        markersize=3.5,
        linewidth=1.7,
        color=DARK_GREY,
        label="Trials with extra detections",
    )
    error_ax.set_title("Failure Modes")
    error_ax.set_xlabel("Range separation, Delta r (cm)")
    error_ax.set_ylabel("Rate / probability")
    error_ax.set_ylim(-0.03, 1.03)
    error_ax.grid(True, alpha=0.3)
    error_ax.legend(frameon=False, loc="upper right")
    add_bank_step_axis(error_ax, framework)

    fig.suptitle("Detection as a Function of Range Separation")
    fig.text(
        0.5,
        0.01,
        (
            f"r1={EXPERIMENT_CFG['r1_m']:.2f} m, r2=r1+Delta r, "
            f"unit reflectivity, no clutter, no added noise, "
            f"n={rows[0].trials} trials/separation"
        ),
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")

    if "agg" not in matplotlib.get_backend().lower():
        plt.show()


def print_summary(rows: Sequence[SeparationSweepRow]) -> None:
    limit_50 = first_crossing(rows, threshold=0.50)
    limit_95 = first_crossing(rows, threshold=0.95)

    print("=" * 112)
    print("DETECTION AS A FUNCTION OF RANGE SEPARATION")
    print("=" * 112)
    print(
        "Delta cm | bank steps | true delay steps | P_resolved | P_exact | miss_rate | fp_trial | mean detections"
    )
    for row in rows:
        true_delay_text = f"[{row.true_delays[0]:.2f}, {row.true_delays[1]:.2f}]"
        print(
            f"{row.separation_cm:8d} | {row.separation_steps:10.2f} | "
            f"{true_delay_text:16s} | {row.resolved_probability:10.3f} | "
            f"{row.exact_detection_probability:7.3f} | {row.miss_rate:9.3f} | "
            f"{row.false_positive_trial_probability:8.3f} | {row.mean_num_detections:15.2f}"
        )

    print("-" * 112)
    if limit_50 is None:
        print("50% resolution threshold: not reached")
    else:
        print(
            "50% resolution threshold: "
            f"{limit_50.separation_cm} cm ({limit_50.separation_steps:.2f} bank steps)"
        )

    if limit_95 is None:
        print("95% resolution threshold: not reached")
    else:
        print(
            "95% resolution threshold: "
            f"{limit_95.separation_cm} cm ({limit_95.separation_steps:.2f} bank steps)"
        )


def main() -> None:
    rows, framework = run_separation_sweep()
    print_summary(rows)
    save_rows_csv(rows)
    print(f"Saved CSV to {OUTPUT_CSV_PATH}")
    plot_sweep(rows=rows, framework=framework)


if __name__ == "__main__":
    main()
