from dataclasses import dataclass
from itertools import product
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from evaluate_model_range import FMSweepEvaluationFramework
from sim_modules.network_range import FMSweepBankNetwork, WTAReadoutLayer


@dataclass
class DetectionRow:
    bank_index: int
    delay_steps: int
    delay_ms: float
    distance_m: float
    spike_count: float


@dataclass
class GridTrialResult:
    config_id: int
    trial_index: int
    num_objects: int
    scene_cfg: Dict[str, object]
    network_cfg: Dict[str, float]
    readout_cfg: Dict[str, float]
    true_distances_m: List[float]
    true_separation_m: Optional[float]
    true_separation_steps: Optional[int]
    true_delays: List[int]
    detected_delays: List[int]
    decoded_delays: List[float]
    decoded_distances_m: List[float]
    detection_rows: List[DetectionRow]
    detection_misses: int
    detection_false_positives: int
    detection_abs_delay_errors: List[float]
    detection_abs_distance_errors_m: List[float]
    population_misses: int
    population_false_positives: int
    population_abs_delay_errors: List[float]
    population_abs_distance_errors_m: List[float]
    has_detection_error: int
    has_population_error: int


@dataclass
class CachedTrialBatch:
    trial_indices: List[int]
    num_objects: int
    scene_cfg: Dict[str, object]
    pulse_spikes: torch.Tensor
    echo_spikes: torch.Tensor
    true_distances_batch: List[List[float]]
    true_delays_batch: List[List[int]]


class FMSweepGridSearchFramework(FMSweepEvaluationFramework):
    """Grid-search wrapper around the existing FM sweep evaluation framework."""

    def __init__(self, seed):
        super().__init__(seed=seed)
        self.base_network_cfg = dict(self.network_cfg)
        self.base_readout_cfg = {
            "beta": 0.7,
            "threshold": 0.3,
            "is": 0.4,
            "ir": 4,
            "inhibition_falloff": "gaussian",
            "sigma": 2.0,
        }
        self.readout = self._build_readout(self.base_readout_cfg)

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
            refractory_steps=network_cfg["rs"],
            lateral_inhibition_strength=network_cfg["lis"],
            lateral_inhibition_radius=network_cfg["lir"],

            inhibition_decay=network_cfg["inhib_decay"],
        ).to(self.device)

    def _build_readout(self, readout_cfg: Dict[str, float]) -> WTAReadoutLayer:
        return WTAReadoutLayer(
            n_neurons=len(self.candidate_delays),
            beta=readout_cfg["beta"],
            threshold=readout_cfg["threshold"],
            inhibition_strength=readout_cfg["is"],
            inhibition_radius=readout_cfg["ir"],
            inhibition_falloff=readout_cfg["inhibition_falloff"],
            sigma=readout_cfg["sigma"],
        ).to(self.device)

    def _delay_steps_to_detection_row(self, delay_steps: int, spike_count: float) -> DetectionRow:
        delay_ms = float(delay_steps) * self.sim_cfg["step_duration_ms"]
        distance_m = self._delay_steps_to_distance_m(delay_steps)
        bank_index = int(np.where(self.candidate_delays == int(delay_steps))[0][0])
        return DetectionRow(
            bank_index=bank_index,
            delay_steps=int(delay_steps),
            delay_ms=delay_ms,
            distance_m=distance_m,
            spike_count=float(spike_count),
        )

    @staticmethod
    def _cfg_key(cfg: Dict[str, object]) -> Tuple[Tuple[str, object], ...]:
        return tuple(sorted(cfg.items()))

    def _default_scene_cfg(self) -> Dict[str, object]:
        return {
            "snr_db": self.echo_default_cfg.get("snr_db"),
            "clutter_fraction": self.echo_default_cfg.get("clutter_fraction"),
        }

    def _build_trial_result(
        self,
        config_id: int,
        trial_index: int,
        num_objects: int,
        scene_cfg: Dict[str, object],
        network_cfg: Dict[str, float],
        readout_cfg: Dict[str, float],
        true_distances_m: Sequence[float],
        true_delays: Sequence[int],
        spike_counts_np: np.ndarray,
        readout_spike_counts_np: np.ndarray,
        tolerance_steps: int,
        net: FMSweepBankNetwork,
    ) -> GridTrialResult:
        active_indices = np.flatnonzero(spike_counts_np >= float(self.detect_cfg["min_spikes"]))
        detected_delays = [int(self.candidate_delays[idx]) for idx in active_indices.tolist()]
        detection_rows = [
            self._delay_steps_to_detection_row(delay_steps, spike_counts_np[idx])
            for idx, delay_steps in zip(active_indices.tolist(), detected_delays)
        ]

        decoded_objects = net.decode_population_response(
            readout_spike_counts_np,
            min_peak_spikes=self.detect_cfg["min_peak_spikes"],
            min_object_spikes=self.detect_cfg["min_object_spikes"],
        )
        decoded_delays = sorted(float(obj.delay_steps) for obj in decoded_objects)
        decoded_distances_m = [
            self._delay_steps_to_distance_m(delay_steps)
            for delay_steps in decoded_delays
        ]

        detection_metrics = self._match_estimates(
            true_delays=true_delays,
            estimated_delays=detected_delays,
            tolerance_steps=tolerance_steps,
            allow_multiple_estimates_per_true=bool(
                self.detect_cfg.get("allow_multiple_estimates_per_true", False)
            ),
        )
        population_metrics = self._match_estimates(
            true_delays=true_delays,
            estimated_delays=decoded_delays,
            tolerance_steps=tolerance_steps,
        )

        return GridTrialResult(
            config_id=config_id,
            trial_index=trial_index,
            num_objects=num_objects,
            scene_cfg=dict(scene_cfg),
            network_cfg=dict(network_cfg),
            readout_cfg=dict(readout_cfg),
            true_distances_m=list(true_distances_m),
            true_separation_m=_two_object_separation_m(true_distances_m),
            true_separation_steps=_two_object_separation_steps(true_delays),
            true_delays=list(true_delays),
            detected_delays=sorted(detected_delays),
            decoded_delays=decoded_delays,
            decoded_distances_m=decoded_distances_m,
            detection_rows=detection_rows,
            detection_misses=detection_metrics.misses,
            detection_false_positives=detection_metrics.false_positives,
            detection_abs_delay_errors=detection_metrics.abs_delay_errors,
            detection_abs_distance_errors_m=detection_metrics.abs_distance_errors_m,
            population_misses=population_metrics.misses,
            population_false_positives=population_metrics.false_positives,
            population_abs_delay_errors=population_metrics.abs_delay_errors,
            population_abs_distance_errors_m=population_metrics.abs_distance_errors_m,
            has_detection_error=int(detection_metrics.misses > 0 or detection_metrics.false_positives > 0),
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
        scene_cfg: Dict[str, object],
        network_cfg: Dict[str, float],
        readout_cfg: Dict[str, float],
        min_distance_m: float,
        max_distance_m: float,
        reflectivity_range: Tuple[float, float],
        roughness_range: Tuple[float, float],
        tolerance_steps: int,
        min_separation_steps: int,
        net: FMSweepBankNetwork,
        readout: WTAReadoutLayer,
    ) -> List[GridTrialResult]:
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
            scene_cfg=scene_cfg,
            network_cfg=network_cfg,
            readout_cfg=readout_cfg,
            pulse_spikes=pulse_spikes,
            echo_spikes=echo_spikes,
            true_distances_batch=true_distances_batch,
            true_delays_batch=true_delays_batch,
            tolerance_steps=tolerance_steps,
            net=net,
            readout=readout,
        )

    def _evaluate_trial_batch_with_config(
        self,
        config_id: int,
        trial_indices: Sequence[int],
        num_objects: int,
        scene_cfg: Dict[str, object],
        network_cfg: Dict[str, float],
        readout_cfg: Dict[str, float],
        pulse_spikes: torch.Tensor,
        echo_spikes: torch.Tensor,
        true_distances_batch: Sequence[Sequence[float]],
        true_delays_batch: Sequence[Sequence[int]],
        tolerance_steps: int,
        net: FMSweepBankNetwork,
        readout: WTAReadoutLayer,
    ) -> List[GridTrialResult]:
        bank_spikes = net.forward(pulse_spikes, echo_spikes)
        readout_spikes = readout(bank_spikes)
        spike_counts = readout_spikes.sum(dim=0)
        if spike_counts.ndim != 2 or spike_counts.shape[0] != len(trial_indices):
            raise ValueError("Grid search batch size does not match the requested trials.")

        spike_counts_np = spike_counts.detach().cpu().numpy()
        return [
            self._build_trial_result(
                config_id=config_id,
                trial_index=trial_index,
                num_objects=num_objects,
                scene_cfg=scene_cfg,
                network_cfg=network_cfg,
                readout_cfg=readout_cfg,
                true_distances_m=true_distances_batch[batch_idx],
                true_delays=true_delays_batch[batch_idx],
                spike_counts_np=spike_counts_np[batch_idx],
                readout_spike_counts_np=spike_counts_np[batch_idx],
                tolerance_steps=tolerance_steps,
                net=net,
            )
            for batch_idx, trial_index in enumerate(trial_indices)
        ]

    def _precompute_trial_batches(
        self,
        scene_configs: Sequence[Dict[str, object]],
        object_counts: Sequence[int],
        trials_per_count: int,
        min_distance_m: float,
        max_distance_m: float,
        reflectivity_range: Tuple[float, float],
        roughness_range: Tuple[float, float],
        min_separation_steps: int,
        trials_per_batch: int,
    ) -> Dict[Tuple[Tuple[str, object], ...], List[CachedTrialBatch]]:
        cached_batches_by_scene: Dict[Tuple[Tuple[str, object], ...], List[CachedTrialBatch]] = {}
        base_echo_cfg = dict(self.echo_default_cfg)

        try:
            for scene_cfg in scene_configs:
                scene_key = self._cfg_key(scene_cfg)
                self.echo_default_cfg = dict(base_echo_cfg)
                self.echo_default_cfg.update(scene_cfg)
                cached_batches_by_scene[scene_key] = []

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
                        cached_batches_by_scene[scene_key].append(
                            CachedTrialBatch(
                                trial_indices=batch_trial_indices,
                                num_objects=int(num_objects),
                                scene_cfg=dict(scene_cfg),
                                pulse_spikes=pulse_spikes,
                                echo_spikes=echo_spikes,
                                true_distances_batch=true_distances_batch,
                                true_delays_batch=true_delays_batch,
                            )
                        )
        finally:
            self.echo_default_cfg = base_echo_cfg

        return cached_batches_by_scene

    def run_trial_with_config(
        self,
        config_id: int,
        trial_index: int,
        num_objects: int,
        network_cfg: Dict[str, float],
        readout_cfg: Dict[str, float],
        min_distance_m: float,
        max_distance_m: float,
        reflectivity_range: Tuple[float, float],
        roughness_range: Tuple[float, float],
        tolerance_steps: int,
        min_separation_steps: int,
    ) -> GridTrialResult:
        net = self._build_network(network_cfg)
        readout = self._build_readout(readout_cfg)
        return self._run_trial_batch_with_config(
            config_id=config_id,
            trial_indices=[trial_index],
            num_objects=num_objects,
            scene_cfg=self._default_scene_cfg(),
            network_cfg=network_cfg,
            readout_cfg=readout_cfg,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
            reflectivity_range=reflectivity_range,
            roughness_range=roughness_range,
            tolerance_steps=tolerance_steps,
            min_separation_steps=min_separation_steps,
            net=net,
            readout=readout,
        )[0]

    def iter_grid_configs(
        self,
        network_grid: Dict[str, Sequence[float]],
        readout_grid: Dict[str, Sequence[float]],
        scene_grid: Optional[Dict[str, Sequence[object]]] = None,
    ) -> Iterable[Tuple[Dict[str, float], Dict[str, float], Dict[str, object]]]:
        network_keys = list(network_grid.keys())
        readout_keys = list(readout_grid.keys())
        scene_grid = {} if scene_grid is None else dict(scene_grid)
        scene_keys = list(scene_grid.keys())

        network_values = [network_grid[key] for key in network_keys]
        readout_values = [readout_grid[key] for key in readout_keys]
        scene_values = [scene_grid[key] for key in scene_keys]
        scene_choices = list(product(*scene_values)) if scene_values else [()]
        strength_idx = network_keys.index("lat_inhib_strength") if "lat_inhib_strength" in network_keys else None
        radius_idx = network_keys.index("lat_inhib_radius") if "lat_inhib_radius" in network_keys else None
        decay_idx = network_keys.index("inhib_decay") if "inhib_decay" in network_keys else None
        default_radius = network_values[radius_idx][0] if radius_idx is not None else None
        default_decay = network_values[decay_idx][0] if decay_idx is not None else None

        for network_choice in product(*network_values):
            if strength_idx is not None and float(network_choice[strength_idx]) <= 0.0:
                if radius_idx is not None and network_choice[radius_idx] != default_radius:
                    continue
                if decay_idx is not None and network_choice[decay_idx] != default_decay:
                    continue

            full_network_cfg = dict(self.base_network_cfg)
            for key, value in zip(network_keys, network_choice):
                full_network_cfg[key] = value

            for readout_choice in product(*readout_values):
                full_readout_cfg = dict(self.base_readout_cfg)
                for key, value in zip(readout_keys, readout_choice):
                    full_readout_cfg[key] = value

                for scene_choice in scene_choices:
                    full_scene_cfg = self._default_scene_cfg()
                    for key, value in zip(scene_keys, scene_choice):
                        full_scene_cfg[key] = value

                    yield full_network_cfg, full_readout_cfg, full_scene_cfg

    def evaluate_grid(
        self,
        network_grid: Dict[str, Sequence[float]],
        readout_grid: Dict[str, Sequence[float]],
        object_counts: Sequence[int],
        trials_per_count: int,
        min_distance_m: float,
        max_distance_m: float,
        reflectivity_range: Tuple[float, float],
        roughness_range: Tuple[float, float],
        tolerance_steps: int,
        min_separation_steps: int,
        scene_grid: Optional[Dict[str, Sequence[object]]] = None,
        trials_per_batch: int = 32,
    ) -> List[GridTrialResult]:
        results: List[GridTrialResult] = []
        grid_configs = list(
            self.iter_grid_configs(
                network_grid=network_grid,
                readout_grid=readout_grid,
                scene_grid=scene_grid,
            )
        )
        scene_configs: List[Dict[str, object]] = []
        seen_scene_keys = set()
        for _, _, scene_cfg in grid_configs:
            scene_key = self._cfg_key(scene_cfg)
            if scene_key in seen_scene_keys:
                continue
            seen_scene_keys.add(scene_key)
            scene_configs.append(scene_cfg)

        total_trials = len(grid_configs) * len(object_counts) * int(trials_per_count)
        completed_trials = 0
        trials_per_batch = max(1, int(trials_per_batch))
        cached_batches_by_scene = self._precompute_trial_batches(
            scene_configs=scene_configs,
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

        for config_id, (network_cfg, readout_cfg, scene_cfg) in enumerate(grid_configs, start=1):
            net = self._build_network(network_cfg)
            readout = self._build_readout(readout_cfg)
            scene_key = self._cfg_key(scene_cfg)
            for cached_batch in cached_batches_by_scene[scene_key]:
                results.extend(
                    self._evaluate_trial_batch_with_config(
                        config_id=config_id,
                        trial_indices=cached_batch.trial_indices,
                        num_objects=cached_batch.num_objects,
                        scene_cfg=scene_cfg,
                        network_cfg=network_cfg,
                        readout_cfg=readout_cfg,
                        pulse_spikes=cached_batch.pulse_spikes,
                        echo_spikes=cached_batch.echo_spikes,
                        true_distances_batch=cached_batch.true_distances_batch,
                        true_delays_batch=cached_batch.true_delays_batch,
                        tolerance_steps=tolerance_steps,
                        net=net,
                        readout=readout,
                    )
                )
                completed_trials += len(cached_batch.trial_indices)
                self._print_progress(completed_trials, total_trials)

        if total_trials > 0:
            print()

        return results


def _format_cfg_subset(cfg: Dict[str, float], keys: Sequence[str]) -> str:
    return ", ".join(f"{key}={cfg[key]}" for key in keys)


def _two_object_separation_m(distances_m: Sequence[float]) -> Optional[float]:
    if len(distances_m) != 2:
        return None
    return abs(float(distances_m[1]) - float(distances_m[0]))


def _two_object_separation_steps(delays: Sequence[int]) -> Optional[int]:
    if len(delays) != 2:
        return None
    return abs(int(delays[1]) - int(delays[0]))


def _format_optional_float(value: Optional[float], precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _format_float_list(values: Sequence[float], precision: int = 2) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def _print_trial_separation_table(results: Sequence[GridTrialResult]) -> None:
    print("\n" + "=" * 156)
    print("TWO-OBJECT SEPARATION BY TRIAL")
    print("=" * 156)
    print(
        "config | trial | r1_m   | r2_m   | separation_m | separation_cm | separation_steps | "
        "true_delays | detected_delays          | centroid_delays | centroid_dist_m"
    )

    for result in sorted(results, key=lambda item: (item.config_id, item.trial_index, item.num_objects)):
        if result.num_objects != 2 or len(result.true_distances_m) != 2:
            continue

        r1_m, r2_m = result.true_distances_m
        separation_cm = (
            result.true_separation_m * 100.0
            if result.true_separation_m is not None
            else None
        )
        print(
            f"{result.config_id:6d} | {result.trial_index:5d} | "
            f"{r1_m:6.4f} | {r2_m:6.4f} | "
            f"{_format_optional_float(result.true_separation_m, precision=4):12s} | "
            f"{_format_optional_float(separation_cm, precision=2):13s} | "
            f"{str(result.true_separation_steps):16s} | "
            f"{str(result.true_delays):11s} | {str(result.detected_delays):24s} | "
            f"{_format_float_list(result.decoded_delays, precision=2):15s} | "
            f"{_format_float_list(result.decoded_distances_m, precision=3)}"
        )


def _print_detection_table(result: GridTrialResult) -> None:
    scene_keys = [key for key in ("snr_db", "clutter_fraction") if key in result.scene_cfg]
    print("-" * 96)
    print(
        f"Config {result.config_id:03d} | trial {result.trial_index:03d} | "
        f"objects={result.num_objects}"
    )
    print("scene: " + _format_cfg_subset(result.scene_cfg, scene_keys))
    print(
        "network: "
        + _format_cfg_subset(
            result.network_cfg,
            [
                "weight",
                "beta",
                "threshold",
                "rs",
                "lis",
                "lir",
                "inhib_decay",
            ],
        )

    )
    print(
        "readout: "
        + _format_cfg_subset(
            result.readout_cfg,
            ["beta", "threshold", "is", "ir", "inhibition_falloff", "sigma"],
        )
    )
    if result.true_separation_m is not None:
        print(
            "object separation: "
            f"{result.true_separation_m:.4f} m "
            f"({result.true_separation_m * 100.0:.2f} cm, "
            f"{result.true_separation_steps} delay steps)"
        )
    print(f"true delays: {result.true_delays}")
    print(f"detected delays: {result.detected_delays}")
    print(f"centroid decoded delays: {_format_float_list(result.decoded_delays, precision=2)}")
    print(f"centroid decoded distances m: {_format_float_list(result.decoded_distances_m, precision=3)}")
    # print("Detection table")
    # print("bank_index | delay_steps | distance_m | spike_count")
    # if not result.detection_rows:
    #     print("(none)")
    # else:
    #     for row in result.detection_rows:
    #         print(
    #             f"{row.bank_index:10d} | {row.delay_steps:11d} | "
    #             f"{row.distance_m:10.4f} | {row.spike_count:10.2f}"
    #         )


def _print_summary(results: Sequence[GridTrialResult]) -> None:
    grouped: Dict[Tuple[int, int], List[GridTrialResult]] = {}
    for result in results:
        grouped.setdefault((result.config_id, result.num_objects), []).append(result)

    summary_rows = []
    for (config_id, num_objects), group in grouped.items():
        trials = len(group)
        total_true = sum(len(result.true_delays) for result in group)
        total_detected = sum(len(result.detected_delays) for result in group)
        total_decoded = sum(len(result.decoded_delays) for result in group)
        sample = group[0]

        det_err = sum(result.has_detection_error for result in group) / float(trials)
        det_miss = sum(result.detection_misses for result in group) / float(total_true) if total_true else 0.0
        det_fp = (
            sum(result.detection_false_positives for result in group) / float(total_detected)
            if total_detected
            else 0.0
        )
        abs_delay_errors = [
            error for result in group for error in result.detection_abs_delay_errors
        ]
        abs_distance_errors_m = [
            error for result in group for error in result.detection_abs_distance_errors_m
        ]
        det_mae_steps = float(np.mean(abs_delay_errors)) if abs_delay_errors else float("nan")
        det_mae_m = float(np.mean(abs_distance_errors_m)) if abs_distance_errors_m else float("nan")
        pop_err = sum(result.has_population_error for result in group) / float(trials)
        pop_miss = sum(result.population_misses for result in group) / float(total_true) if total_true else 0.0
        pop_fp = (
            sum(result.population_false_positives for result in group) / float(total_decoded)
            if total_decoded
            else 0.0
        )
        pop_abs_delay_errors = [
            error for result in group for error in result.population_abs_delay_errors
        ]
        pop_abs_distance_errors_m = [
            error for result in group for error in result.population_abs_distance_errors_m
        ]
        pop_mae_steps = (
            float(np.mean(pop_abs_delay_errors)) if pop_abs_delay_errors else float("nan")
        )
        pop_mae_m = (
            float(np.mean(pop_abs_distance_errors_m))
            if pop_abs_distance_errors_m
            else float("nan")
        )

        summary_rows.append(
            {
                "config_id": config_id,
                "num_objects": num_objects,
                "trials": trials,
                "det_err": det_err,
                "det_miss": det_miss,
                "det_fp": det_fp,
                "det_mae_m": det_mae_m,
                "det_mae_steps": det_mae_steps,
                "pop_err": pop_err,
                "pop_miss": pop_miss,
                "pop_fp": pop_fp,
                "pop_mae_m": pop_mae_m,
                "pop_mae_steps": pop_mae_steps,
                "scene_params": _format_cfg_subset(
                    sample.scene_cfg,
                    [key for key in ("snr_db", "clutter_fraction") if key in sample.scene_cfg],
                ),
                "network_params": _format_cfg_subset(
                    sample.network_cfg,
                    [
                        "weight",
                        "beta",
                        "threshold",
                        "rs",
                        "lis",
                        "lir",
                        "inhib_decay",
                    ],
                ),
                "readout_params": _format_cfg_subset(
                    sample.readout_cfg,
                    ["beta", "threshold", "is", "ir"],
                ),
            }
        )

    print("\n" + "=" * 182)
    print("GRID SEARCH SUMMARY")
    print("=" * 182)
    print(
        "config | objects | trials | det_err | det_miss | det_fp | det_mae_m | "
        "pop_err | pop_miss | pop_fp | pop_mae_m | pop_mae_steps | "
        "scene params | network params                                                      | readout params"
    )
    for row in sorted(
        summary_rows,
        key=lambda row: (
            row["det_err"],
            row["det_miss"],
            row["det_fp"],
            row["det_mae_m"],
            row["pop_err"],
            row["pop_miss"],
            row["pop_fp"],
            row["pop_mae_m"],
            row["config_id"],
            row["num_objects"],
        ),
    ):
        print(
            f"{row['config_id']:6d} | {row['num_objects']:7d} | {row['trials']:6d} | "
            f"{row['det_err']:.3f}   | {row['det_miss']:.3f}    | {row['det_fp']:.3f} | "
            f"{row['det_mae_m']:.4f}    | {row['pop_err']:.3f}   | {row['pop_miss']:.3f}    | "
            f"{row['pop_fp']:.3f} | {row['pop_mae_m']:.4f}    | {row['pop_mae_steps']:.3f}         | "
            f"{row['scene_params']} | "
            f"{row['network_params']} | {row['readout_params']}"
        )


def main() -> None:
    grid_cfg = {
        "object_counts": [1],
        "trials_per_count": 100,
        "trials_per_batch": 32,
        "min_distance_m": 0.15,
        "max_distance_m": 3.4,
        "reflectivity_range": (0.1, 0.8),
        "roughness_range": (0.1, 0.35),
        "tolerance_steps": 2,
        "min_separation_steps": 0,
        "seed":420 #np.random.randint(0, 1000),
    }
    network_grid = {
        "weight": [0.1],
        "beta": [0.1],
        "threshold": [0.7],
        "rs": [2],
        "lis": [3],
        "lir": [8],
        "inhib_decay": [0.1],
    }
    readout_grid = {
        "beta": [0.4],
        "threshold": [0.4],
        "is": [0],
        "ir": [4],
        "inhibition_falloff": ["gaussian"],
        "sigma": [2.0],
    }
    scene_grid = {
        "snr_db": [-20,-15,-10,-5,0,5,10,15],
        "clutter_fraction": [0.0],
    }

    torch.manual_seed(grid_cfg["seed"])
    np.random.seed(grid_cfg["seed"])

    framework = FMSweepGridSearchFramework(seed=grid_cfg["seed"])
    results = framework.evaluate_grid(
        network_grid=network_grid,
        readout_grid=readout_grid,
        scene_grid=scene_grid,
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

    _print_trial_separation_table(results)

    _print_summary(results)


if __name__ == "__main__":
    main()
