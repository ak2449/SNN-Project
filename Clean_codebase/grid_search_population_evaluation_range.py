from dataclasses import dataclass
from itertools import product
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from evaluate_model_range import FMSweepEvaluationFramework
from sim_modules.network_range import FMSweepBankNetwork


@dataclass
class PopulationDecodingRow:
    decoded_delay_steps: float
    delay_ms: float
    distance_m: float
    total_spikes: float
    active_neurons: int
    max_spike_count: float


@dataclass
class PopulationGridTrialResult:
    config_id: int
    trial_index: int
    num_objects: int
    network_cfg: Dict[str, float]
    decoder_cfg: Dict[str, float]
    true_distances_m: List[float]
    true_delays: List[int]
    decoded_delays: List[float]
    decoded_distances_m: List[float]
    decoded_rows: List[PopulationDecodingRow]
    population_misses: int
    population_false_positives: int
    population_abs_delay_errors: List[float]
    population_abs_distance_errors_m: List[float]
    has_population_error: int


@dataclass
class CachedTrialBatch:
    trial_indices: List[int]
    num_objects: int
    pulse_spikes: torch.Tensor
    echo_spikes: torch.Tensor
    true_distances_batch: List[List[float]]
    true_delays_batch: List[List[int]]


class FMSweepPopulationGridSearchFramework(FMSweepEvaluationFramework):
    """Grid search over network and single-object population-decoder parameters."""

    def __init__(self, seed: int = 0):
        super().__init__(seed=seed)
        self.base_network_cfg = dict(self.network_cfg)
        self.base_decoder_cfg = {
            "min_total_spikes": self.detect_cfg["min_object_spikes"],
        }

    @staticmethod
    def _print_progress(completed: int, total: int) -> None:
        if total <= 0:
            return

        bar_width = 32
        ratio = min(max(completed / float(total), 0.0), 1.0)
        filled = int(round(bar_width * ratio))
        bar = "#" * filled + "-" * (bar_width - filled)
        sys.stdout.write(f"\rProgress [{bar}] {completed}/{total} ({ratio * 100:5.1f}%)")
        sys.stdout.flush()

    def _build_network(self, network_cfg: Dict[str, float]) -> FMSweepBankNetwork:
        return FMSweepBankNetwork(
            delays=self.candidate_delays,
            freq_channels=self.encoder_cfg["freq_channels"],
            weight=network_cfg["weight"],
            beta=network_cfg["beta"],
            threshold=network_cfg["threshold"],
            refractory_steps=network_cfg["refractory_steps"],
            lateral_inhibition_strength=network_cfg["lateral_inhibition_strength"],
            lateral_inhibition_radius=network_cfg["lateral_inhibition_radius"],
            inhibition_decay=network_cfg["inhibition_decay"],
        ).to(self.device)

    def _decode_single_object(self, spike_counts_np: np.ndarray, decoder_cfg: Dict[str, float]) -> PopulationDecodingRow | None:
        counts = np.asarray(spike_counts_np, dtype=float)
        if counts.ndim != 1:
            raise ValueError("Single-object population decoding expects a 1-D spike count vector.")
        if counts.size != len(self.candidate_delays):
            raise ValueError("Spike count vector must match the number of delay neurons.")

        total_spikes = float(counts.sum())
        if total_spikes < float(decoder_cfg["min_total_spikes"]):
            return None

        decoded_delay_steps = float(
            np.sum(self.candidate_delays.astype(float) * counts) / max(total_spikes, 1e-8)
        )
        return PopulationDecodingRow(
            decoded_delay_steps=decoded_delay_steps,
            delay_ms=decoded_delay_steps * self.sim_cfg["step_duration_ms"],
            distance_m=self._delay_steps_to_distance_m(decoded_delay_steps),
            total_spikes=total_spikes,
            active_neurons=int(np.count_nonzero(counts > 0.0)),
            max_spike_count=float(counts.max()) if counts.size else 0.0,
        )

    def _build_trial_result(
        self,
        config_id: int,
        trial_index: int,
        num_objects: int,
        network_cfg: Dict[str, float],
        decoder_cfg: Dict[str, float],
        true_distances_m: Sequence[float],
        true_delays: Sequence[int],
        spike_counts_np: np.ndarray,
        tolerance_steps: int,
        net: FMSweepBankNetwork,
    ) -> PopulationGridTrialResult:
        decoded_row = self._decode_single_object(spike_counts_np=spike_counts_np, decoder_cfg=decoder_cfg)
        decoded_rows = [] if decoded_row is None else [decoded_row]
        decoded_delays = [] if decoded_row is None else [decoded_row.decoded_delay_steps]
        decoded_distances_m = [] if decoded_row is None else [decoded_row.distance_m]

        population_metrics = self._match_estimates(
            true_delays=true_delays,
            estimated_delays=decoded_delays,
            tolerance_steps=tolerance_steps,
        )

        return PopulationGridTrialResult(
            config_id=config_id,
            trial_index=trial_index,
            num_objects=num_objects,
            network_cfg=dict(network_cfg),
            decoder_cfg=dict(decoder_cfg),
            true_distances_m=list(true_distances_m),
            true_delays=list(true_delays),
            decoded_delays=sorted(decoded_delays),
            decoded_distances_m=sorted(decoded_distances_m),
            decoded_rows=decoded_rows,
            population_misses=population_metrics.misses,
            population_false_positives=population_metrics.false_positives,
            population_abs_delay_errors=population_metrics.abs_delay_errors,
            population_abs_distance_errors_m=population_metrics.abs_distance_errors_m,
            has_population_error=int(population_metrics.misses > 0 or population_metrics.false_positives > 0),
        )

    def _simulate_trial_batch(
        self,
        num_trials: int,
        num_objects: int,
        min_distance_m: float,
        max_distance_m: float,
        reflectivity_range: Tuple[float, float],
        roughness_range: Tuple[float, float],
        min_separation_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[float]], List[List[int]]]:
        pulse_batch: List[torch.Tensor] = []
        echo_batch: List[torch.Tensor] = []
        true_distances_batch: List[List[float]] = []
        true_delays_batch: List[List[int]] = []

        for _ in range(int(num_trials)):
            pulse_waveform, pulse_spikes = self._build_pulse()
            _, echo_spikes, true_distances_m, true_delays = self._simulate_scene_with_pulse(
                pulse_waveform=pulse_waveform,
                pulse_spikes=pulse_spikes,
                num_objects=num_objects,
                min_distance_m=min_distance_m,
                max_distance_m=max_distance_m,
                reflectivity_range=reflectivity_range,
                roughness_range=roughness_range,
                min_separation_steps=min_separation_steps,
            )
            pulse_batch.append(pulse_spikes)
            echo_batch.append(echo_spikes)
            true_distances_batch.append(list(true_distances_m))
            true_delays_batch.append(list(true_delays))

        return (
            torch.cat(pulse_batch, dim=1),
            torch.cat(echo_batch, dim=1),
            true_distances_batch,
            true_delays_batch,
        )

    def _run_trial_batch_with_config(
        self,
        config_id: int,
        trial_indices: Sequence[int],
        num_objects: int,
        network_cfg: Dict[str, float],
        decoder_cfg: Dict[str, float],
        min_distance_m: float,
        max_distance_m: float,
        reflectivity_range: Tuple[float, float],
        roughness_range: Tuple[float, float],
        tolerance_steps: int,
        min_separation_steps: int,
        net: FMSweepBankNetwork,
    ) -> List[PopulationGridTrialResult]:
        pulse_spikes, echo_spikes, true_distances_batch, true_delays_batch = self._simulate_trial_batch(
            num_trials=len(trial_indices),
            num_objects=num_objects,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
            reflectivity_range=reflectivity_range,
            roughness_range=roughness_range,
            min_separation_steps=min_separation_steps,
        )

        return self._evaluate_trial_batch_with_config(
            config_id=config_id,
            trial_indices=trial_indices,
            num_objects=num_objects,
            network_cfg=network_cfg,
            decoder_cfg=decoder_cfg,
            pulse_spikes=pulse_spikes,
            echo_spikes=echo_spikes,
            true_distances_batch=true_distances_batch,
            true_delays_batch=true_delays_batch,
            tolerance_steps=tolerance_steps,
            net=net,
        )

    def _evaluate_trial_batch_with_config(
        self,
        config_id: int,
        trial_indices: Sequence[int],
        num_objects: int,
        network_cfg: Dict[str, float],
        decoder_cfg: Dict[str, float],
        pulse_spikes: torch.Tensor,
        echo_spikes: torch.Tensor,
        true_distances_batch: Sequence[Sequence[float]],
        true_delays_batch: Sequence[Sequence[int]],
        tolerance_steps: int,
        net: FMSweepBankNetwork,
    ) -> List[PopulationGridTrialResult]:
        bank_spikes = net.forward(pulse_spikes, echo_spikes)
        spike_counts = bank_spikes.sum(dim=0)
        if spike_counts.ndim != 2 or spike_counts.shape[0] != len(trial_indices):
            raise ValueError("Grid search batch size does not match the requested trials.")

        spike_counts_np = spike_counts.detach().cpu().numpy()
        return [
            self._build_trial_result(
                config_id=config_id,
                trial_index=trial_index,
                num_objects=num_objects,
                network_cfg=network_cfg,
                decoder_cfg=decoder_cfg,
                true_distances_m=true_distances_batch[batch_idx],
                true_delays=true_delays_batch[batch_idx],
                spike_counts_np=spike_counts_np[batch_idx],
                tolerance_steps=tolerance_steps,
                net=net,
            )
            for batch_idx, trial_index in enumerate(trial_indices)
        ]

    def _precompute_trial_batches(
        self,
        object_counts: Sequence[int],
        trials_per_count: int,
        min_distance_m: float,
        max_distance_m: float,
        reflectivity_range: Tuple[float, float],
        roughness_range: Tuple[float, float],
        min_separation_steps: int,
        trials_per_batch: int,
    ) -> List[CachedTrialBatch]:
        cached_batches: List[CachedTrialBatch] = []

        for num_objects in object_counts:
            for trial_offset in range(0, int(trials_per_count), trials_per_batch):
                batch_trial_indices = list(
                    range(
                        trial_offset + 1,
                        min(int(trials_per_count), trial_offset + trials_per_batch) + 1,
                    )
                )
                pulse_spikes, echo_spikes, true_distances_batch, true_delays_batch = self._simulate_trial_batch(
                    num_trials=len(batch_trial_indices),
                    num_objects=int(num_objects),
                    min_distance_m=min_distance_m,
                    max_distance_m=max_distance_m,
                    reflectivity_range=reflectivity_range,
                    roughness_range=roughness_range,
                    min_separation_steps=min_separation_steps,
                )
                cached_batches.append(
                    CachedTrialBatch(
                        trial_indices=batch_trial_indices,
                        num_objects=int(num_objects),
                        pulse_spikes=pulse_spikes,
                        echo_spikes=echo_spikes,
                        true_distances_batch=true_distances_batch,
                        true_delays_batch=true_delays_batch,
                    )
                )

        return cached_batches

    def run_trial_with_config(
        self,
        config_id: int,
        trial_index: int,
        num_objects: int,
        network_cfg: Dict[str, float],
        decoder_cfg: Dict[str, float],
        min_distance_m: float,
        max_distance_m: float,
        reflectivity_range: Tuple[float, float],
        roughness_range: Tuple[float, float],
        tolerance_steps: int,
        min_separation_steps: int,
    ) -> PopulationGridTrialResult:
        net = self._build_network(network_cfg)
        return self._run_trial_batch_with_config(
            config_id=config_id,
            trial_indices=[trial_index],
            num_objects=num_objects,
            network_cfg=network_cfg,
            decoder_cfg=decoder_cfg,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
            reflectivity_range=reflectivity_range,
            roughness_range=roughness_range,
            tolerance_steps=tolerance_steps,
            min_separation_steps=min_separation_steps,
            net=net,
        )[0]

    def iter_grid_configs(
        self,
        network_grid: Dict[str, Sequence[float]],
        decoder_grid: Dict[str, Sequence[float]],
    ) -> Iterable[Tuple[Dict[str, float], Dict[str, float]]]:
        network_keys = list(network_grid.keys())
        decoder_keys = list(decoder_grid.keys())

        network_values = [network_grid[key] for key in network_keys]
        decoder_values = [decoder_grid[key] for key in decoder_keys]

        for network_choice in product(*network_values):
            full_network_cfg = dict(self.base_network_cfg)
            for key, value in zip(network_keys, network_choice):
                full_network_cfg[key] = value

            for decoder_choice in product(*decoder_values):
                full_decoder_cfg = dict(self.base_decoder_cfg)
                for key, value in zip(decoder_keys, decoder_choice):
                    full_decoder_cfg[key] = value

                yield full_network_cfg, full_decoder_cfg

    def evaluate_grid(
        self,
        network_grid: Dict[str, Sequence[float]],
        decoder_grid: Dict[str, Sequence[float]],
        object_counts: Sequence[int],
        trials_per_count: int,
        min_distance_m: float,
        max_distance_m: float,
        reflectivity_range: Tuple[float, float],
        roughness_range: Tuple[float, float],
        tolerance_steps: int,
        min_separation_steps: int,
        trials_per_batch: int = 1,
    ) -> List[PopulationGridTrialResult]:
        results: List[PopulationGridTrialResult] = []
        grid_configs = list(self.iter_grid_configs(network_grid=network_grid, decoder_grid=decoder_grid))
        total_trials = len(grid_configs) * len(object_counts) * int(trials_per_count)
        completed_trials = 0
        trials_per_batch = max(1, int(trials_per_batch))
        cached_batches = self._precompute_trial_batches(
            object_counts=object_counts,
            trials_per_count=trials_per_count,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
            reflectivity_range=reflectivity_range,
            roughness_range=roughness_range,
            min_separation_steps=min_separation_steps,
            trials_per_batch=trials_per_batch,
        )

        self._print_progress(0, total_trials)

        for config_id, (network_cfg, decoder_cfg) in enumerate(grid_configs, start=1):
            net = self._build_network(network_cfg)
            for cached_batch in cached_batches:
                results.extend(
                    self._evaluate_trial_batch_with_config(
                        config_id=config_id,
                        trial_indices=cached_batch.trial_indices,
                        num_objects=cached_batch.num_objects,
                        network_cfg=network_cfg,
                        decoder_cfg=decoder_cfg,
                        pulse_spikes=cached_batch.pulse_spikes,
                        echo_spikes=cached_batch.echo_spikes,
                        true_distances_batch=cached_batch.true_distances_batch,
                        true_delays_batch=cached_batch.true_delays_batch,
                        tolerance_steps=tolerance_steps,
                        net=net,
                    )
                )
                completed_trials += len(cached_batch.trial_indices)
                self._print_progress(completed_trials, total_trials)

        if total_trials > 0:
            print()

        return results


def _format_cfg_subset(cfg: Dict[str, float], keys: Sequence[str]) -> str:
    return ", ".join(f"{key}={cfg[key]}" for key in keys)


def _print_population_table(result: PopulationGridTrialResult) -> None:
    print("-" * 112)
    print(
        f"Config {result.config_id:03d} | trial {result.trial_index:03d} | "
        f"objects={result.num_objects}"
    )
    print(
        "network: "
        + _format_cfg_subset(result.network_cfg, ["weight", "beta", "threshold", "refractory_steps"])
    )
    print(
        "decoder: " + _format_cfg_subset(result.decoder_cfg, ["min_total_spikes"])
    )
    print(f"true delays: {result.true_delays}")
    print(f"decoded delays: {[round(delay, 3) for delay in result.decoded_delays]}")
    # print("Population decoding table")
    # print("decoded_delay_steps | distance_m | total_spikes | active_neurons | max_spike_count")
    # if not result.decoded_rows:
    #     print("(none)")
    # else:
    #     for row in result.decoded_rows:
    #         print(
    #             f"{row.decoded_delay_steps:19.3f} | {row.distance_m:10.4f} | {row.total_spikes:12.2f} | "
    #             f"{row.active_neurons:14d} | {row.max_spike_count:15.2f}"
    #         )


def _print_summary(results: Sequence[PopulationGridTrialResult]) -> None:
    grouped: Dict[int, List[PopulationGridTrialResult]] = {}
    for result in results:
        grouped.setdefault(result.config_id, []).append(result)

    summary_rows = []
    for config_id, group in grouped.items():
        trials = len(group)
        total_true = sum(len(result.true_delays) for result in group)
        total_decoded = sum(len(result.decoded_delays) for result in group)
        population_misses = sum(result.population_misses for result in group)
        population_fp = sum(result.population_false_positives for result in group)
        population_delay_errors = [
            err for result in group for err in result.population_abs_delay_errors
        ]
        population_distance_errors_m = [
            err for result in group for err in result.population_abs_distance_errors_m
        ]
        sample = group[0]

        summary_rows.append(
            {
                "config_id": config_id,
                "trials": trials,
                "pop_err": sum(result.has_population_error for result in group) / float(trials),
                "pop_miss": population_misses / float(total_true) if total_true else 0.0,
                "pop_fp": population_fp / float(total_decoded) if total_decoded else 0.0,
                "mean_pop_delay": float(np.mean(population_delay_errors)) if population_delay_errors else 0.0,
                "mean_pop_dist": (
                    float(np.mean(population_distance_errors_m)) if population_distance_errors_m else 0.0
                ),
                "network_params": _format_cfg_subset(sample.network_cfg, ["weight", "beta", "threshold"]),
                "decoder_params": _format_cfg_subset(
                    sample.decoder_cfg,
                    ["min_total_spikes"],
                ),
            }
        )

    print("\n" + "=" * 164)
    print("POPULATION DECODING GRID SEARCH SUMMARY")
    print("=" * 164)
    print(
        "config | trials | pop_err | pop_miss | pop_fp | mean_delay_err | mean_dist_err_m | "
        "network params                      | decoder params"
    )
    for row in sorted(
        summary_rows,
        key=lambda row: (
            row["pop_err"],
            row["pop_miss"],
            row["pop_fp"],
            row["mean_pop_dist"],
            row["config_id"],
        ),
    ):
        print(
            f"{row['config_id']:6d} | {row['trials']:6d} | {row['pop_err']:.3f}   | "
            f"{row['pop_miss']:.3f}    | {row['pop_fp']:.3f} | {row['mean_pop_delay']:.3f}         | "
            f"{row['mean_pop_dist']:.4f}          | {row['network_params']} | {row['decoder_params']}"
        )


def main() -> None:
    grid_cfg = {
        "object_counts": [1],
        "trials_per_count": 500,
        "trials_per_batch": 16,
        "min_distance_m": 0.15,
        "max_distance_m": 3.4,
        "reflectivity_range": (0.1, 0.8),
        "roughness_range": (0.1, 0.35),
        "tolerance_steps": 3,
        "min_separation_steps": 2,
        "seed": np.random.randint(0, 10000),
    }
    network_grid = {
        "weight": [0.3, 0.4],
        "beta": [0.5],
        "threshold": [0.5],
        "refractory_steps": [0],
    }
    decoder_grid = {
        "min_total_spikes": [2, 3, 4],
    }

    torch.manual_seed(grid_cfg["seed"])
    np.random.seed(grid_cfg["seed"])

    framework = FMSweepPopulationGridSearchFramework(seed=grid_cfg["seed"])
    results = framework.evaluate_grid(
        network_grid=network_grid,
        decoder_grid=decoder_grid,
        object_counts=grid_cfg["object_counts"],
        trials_per_count=grid_cfg["trials_per_count"],
        min_distance_m=grid_cfg["min_distance_m"],
        max_distance_m=grid_cfg["max_distance_m"],
        reflectivity_range=grid_cfg["reflectivity_range"],
        roughness_range=grid_cfg["roughness_range"],
        tolerance_steps=grid_cfg["tolerance_steps"],
        min_separation_steps=grid_cfg["min_separation_steps"],
        trials_per_batch=grid_cfg["trials_per_batch"],
    )

    # for result in results:
    #     _print_population_table(result)

    _print_summary(results)


if __name__ == "__main__":
    main()
