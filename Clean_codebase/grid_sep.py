from dataclasses import dataclass
from itertools import product
from pathlib import Path
import csv
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from evaluate_model_range import FMSweepEvaluationFramework
from sim_modules.echo_range import generate_fm_echo_from_waveform
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
    true_delays: List[int]
    detected_delays: List[int]
    detection_rows: List[DetectionRow]
    detection_misses: int
    detection_false_positives: int
    detection_abs_delay_errors: List[float]
    detection_abs_distance_errors_m: List[float]
    has_detection_error: int


@dataclass
class CachedTrialBatch:
    trial_indices: List[int]
    num_objects: int
    scene_cfg: Dict[str, object]
    pulse_spikes: torch.Tensor
    echo_spikes: torch.Tensor
    true_distances_batch: List[List[float]]
    true_delays_batch: List[List[int]]


@dataclass
class SeparationSweepResult:
    separation_cm: int
    separation_m: float
    separation_steps: float
    true_distances_m: List[float]
    true_delays: List[int]
    trials: int
    detection_error_trials: int
    detection_error_probability: float
    miss_rate: float
    false_positive_rate: float
    mean_num_detections: float
    mean_abs_distance_error_m: float


@dataclass
class SeparationBatchStats:
    trials: int
    total_true: int
    total_detected: int
    detection_error_trials: int
    misses: int
    false_positives: int
    abs_distance_errors_m: List[float]


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
        tolerance_steps: int,
    ) -> GridTrialResult:
        active_indices = np.flatnonzero(spike_counts_np >= float(self.detect_cfg["min_spikes"]))
        detected_delays = [int(self.candidate_delays[idx]) for idx in active_indices.tolist()]
        detection_rows = [
            self._delay_steps_to_detection_row(delay_steps, spike_counts_np[idx])
            for idx, delay_steps in zip(active_indices.tolist(), detected_delays)
        ]

        detection_metrics = self._match_estimates(
            true_delays=true_delays,
            estimated_delays=detected_delays,
            tolerance_steps=tolerance_steps,
            allow_multiple_estimates_per_true=bool(
                self.detect_cfg.get("allow_multiple_estimates_per_true", False)
            ),
        )

        return GridTrialResult(
            config_id=config_id,
            trial_index=trial_index,
            num_objects=num_objects,
            scene_cfg=dict(scene_cfg),
            network_cfg=dict(network_cfg),
            readout_cfg=dict(readout_cfg),
            true_distances_m=list(true_distances_m),
            true_delays=list(true_delays),
            detected_delays=sorted(detected_delays),
            detection_rows=detection_rows,
            detection_misses=detection_metrics.misses,
            detection_false_positives=detection_metrics.false_positives,
            detection_abs_delay_errors=detection_metrics.abs_delay_errors,
            detection_abs_distance_errors_m=detection_metrics.abs_distance_errors_m,
            has_detection_error=int(detection_metrics.misses > 0 or detection_metrics.false_positives > 0),
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

    def _simulate_fixed_distance_trial_batch(
        self,
        num_trials: int,
        distances_m: Sequence[float],
        echo_cfg: Dict[str, object],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[float]], List[List[int]]]:
        pulse_waveform, pulse_spike_probs = self._get_pulse_template()
        pulse_spikes = self.encoder.sample_spikes_from_probabilities(
            spike_probs=pulse_spike_probs,
            batch_size=num_trials,
            device=self.device,
        )

        object_echo_batches: List[torch.Tensor] = []
        true_distances_m = sorted(float(distance_m) for distance_m in distances_m)
        true_delays: List[int] = []

        for distance_m in true_distances_m:
            obj_echo_cfg = dict(self.echo_default_cfg)
            obj_echo_cfg.update(echo_cfg)
            obj_echo_cfg["distance_m"] = float(distance_m)

            _, echo_waveform, true_delay_i = generate_fm_echo_from_waveform(
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
            echo_spike_probs = self.encoder.compute_spike_probabilities_from_waveform(
                waveform=echo_waveform,
                num_steps=self.sim_cfg["num_steps"],
                max_spike_prob=obj_echo_cfg["max_spike_prob"],
                sample_rate_hz=self.sim_cfg["sample_rate_hz"],
                step_duration_ms=self.sim_cfg["step_duration_ms"],
                filter_taps=obj_echo_cfg["filter_taps"],
                bandwidth_khz=obj_echo_cfg["bandwidth_khz"],
                smoothing_frac=obj_echo_cfg["smoothing_frac"],
                device=self.device,
            )
            object_echo_batches.append(
                self.encoder.sample_spikes_from_probabilities(
                    spike_probs=echo_spike_probs,
                    batch_size=num_trials,
                    device=self.device,
                )
            )
            true_delays.append(int(true_delay_i))

        echo_spikes = torch.clamp(sum(object_echo_batches), 0, 1)
        sorted_true_delays = sorted(true_delays)
        return (
            pulse_spikes,
            echo_spikes,
            [list(true_distances_m) for _ in range(num_trials)],
            [list(sorted_true_delays) for _ in range(num_trials)],
        )

    def _run_fixed_distance_batch_with_config(
        self,
        config_id: int,
        trial_indices: Sequence[int],
        distances_m: Sequence[float],
        scene_cfg: Dict[str, object],
        echo_cfg: Dict[str, object],
        network_cfg: Dict[str, float],
        readout_cfg: Dict[str, float],
        tolerance_steps: int,
        net: FMSweepBankNetwork,
        readout: WTAReadoutLayer,
    ) -> List[GridTrialResult]:
        pulse_spikes, echo_spikes, true_distances_batch, true_delays_batch = (
            self._simulate_fixed_distance_trial_batch(
                num_trials=len(trial_indices),
                distances_m=distances_m,
                echo_cfg=echo_cfg,
            )
        )

        return self._evaluate_trial_batch_with_config(
            config_id=config_id,
            trial_indices=trial_indices,
            num_objects=len(distances_m),
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
                tolerance_steps=tolerance_steps,
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
    print(f"true delays: {result.true_delays}")
    print(f"detected delays: {result.detected_delays}")
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

    print("\n" + "=" * 156)
    print("GRID SEARCH SUMMARY")
    print("=" * 156)
    print(
        "config | objects | trials | det_err | det_miss |det_fp | mae_m  | mae_steps | "
        "scene params | network params                                                      | readout params"
    )
    for row in sorted(
        summary_rows,
        key=lambda row: (
            row["det_err"],
            row["det_miss"],
            row["det_fp"],
            row["det_mae_m"],
            row["config_id"],
            row["num_objects"],
        ),
    ):
        print(
            f"{row['config_id']:6d} | {row['num_objects']:7d} | {row['trials']:6d} | {row['det_err']:.3f}   | "
            f"{row['det_miss']:.3f}    | {row['det_fp']:.3f} | {row['det_mae_m']:.4f} | "
            f"{row['det_mae_steps']:.3f}     | {row['scene_params']} | "
            f"{row['network_params']} | {row['readout_params']}"
        )


OUTPUT_FIG_PATH = Path(__file__).with_name("fig_detection_vs_separation_range.png")
OUTPUT_CSV_PATH = Path(__file__).with_name("detection_vs_separation_range.csv")


def _separation_m_to_steps(framework: FMSweepGridSearchFramework, separation_m: float) -> float:
    step_duration_s = framework.sim_cfg["step_duration_ms"] * 1e-3
    speed = framework.echo_default_cfg["speed_of_sound_m_s"]
    return (2.0 * float(separation_m) / speed) / step_duration_s


def _build_clean_two_object_echo_cfg(
    framework: FMSweepGridSearchFramework,
    sweep_cfg: Dict[str, object],
) -> Dict[str, object]:
    echo_cfg = dict(framework.echo_default_cfg)
    echo_cfg.update(
        {
            "reflectivity": sweep_cfg["reflectivity"],
            "surface_roughness": sweep_cfg["surface_roughness"],
            "surface_texture_scale_m": sweep_cfg["surface_texture_scale_m"],
            "air_absorption_db_per_m_100khz": sweep_cfg["air_absorption_db_per_m_100khz"],
            "snr_db": None,
            "clutter_fraction": 0.0,
            "num_clutter_paths": 0,
        }
    )
    return echo_cfg


def _evaluate_separation_detection_batch(
    framework: FMSweepGridSearchFramework,
    net: FMSweepBankNetwork,
    readout: WTAReadoutLayer,
    pulse_spikes: torch.Tensor,
    echo_spikes: torch.Tensor,
    true_delays: Sequence[int],
    tolerance_steps: int,
) -> SeparationBatchStats:
    bank_spikes = net.forward(pulse_spikes, echo_spikes)
    readout_spikes = readout(bank_spikes)
    spike_counts = readout_spikes.sum(dim=0)
    if spike_counts.ndim != 2:
        raise ValueError("Expected spike counts with shape [batch, neurons].")

    spike_counts_np = spike_counts.detach().cpu().numpy()
    trials = int(spike_counts_np.shape[0])
    detection_error_trials = 0
    misses = 0
    false_positives = 0
    total_detected = 0
    abs_distance_errors_m: List[float] = []

    for counts in spike_counts_np:
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

        total_detected += len(detected_delays)
        misses += metrics.misses
        false_positives += metrics.false_positives
        abs_distance_errors_m.extend(metrics.abs_distance_errors_m)
        if metrics.misses > 0 or metrics.false_positives > 0:
            detection_error_trials += 1

    return SeparationBatchStats(
        trials=trials,
        total_true=len(true_delays) * trials,
        total_detected=total_detected,
        detection_error_trials=detection_error_trials,
        misses=misses,
        false_positives=false_positives,
        abs_distance_errors_m=abs_distance_errors_m,
    )


def _summarize_separation_stats(
    framework: FMSweepGridSearchFramework,
    separation_cm: int,
    separation_m: float,
    true_distances_m: Sequence[float],
    true_delays: Sequence[int],
    batch_stats: Sequence[SeparationBatchStats],
) -> SeparationSweepResult:
    if not batch_stats:
        raise RuntimeError("No trial results were generated for the separation scenario.")

    trials = sum(stats.trials for stats in batch_stats)
    total_true = sum(stats.total_true for stats in batch_stats)
    total_detected = sum(stats.total_detected for stats in batch_stats)
    detection_error_trials = sum(stats.detection_error_trials for stats in batch_stats)
    misses = sum(stats.misses for stats in batch_stats)
    false_positives = sum(stats.false_positives for stats in batch_stats)
    abs_distance_errors_m = [
        error for stats in batch_stats for error in stats.abs_distance_errors_m
    ]

    return SeparationSweepResult(
        separation_cm=int(separation_cm),
        separation_m=float(separation_m),
        separation_steps=_separation_m_to_steps(framework, separation_m),
        true_distances_m=list(true_distances_m),
        true_delays=list(true_delays),
        trials=trials,
        detection_error_trials=detection_error_trials,
        detection_error_probability=detection_error_trials / float(trials),
        miss_rate=misses / float(total_true) if total_true else 0.0,
        false_positive_rate=(
            false_positives / float(total_detected) if total_detected else 0.0
        ),
        mean_num_detections=total_detected / float(trials),
        mean_abs_distance_error_m=(
            float(np.mean(abs_distance_errors_m)) if abs_distance_errors_m else 0.0
        ),
    )


def run_two_object_separation_sweep(
    framework: FMSweepGridSearchFramework,
    network_cfg: Dict[str, float],
    readout_cfg: Dict[str, float],
    sweep_cfg: Dict[str, object],
) -> List[SeparationSweepResult]:
    echo_cfg = _build_clean_two_object_echo_cfg(framework, sweep_cfg)
    net = framework._build_network(network_cfg)
    readout = framework._build_readout(readout_cfg)

    trials_per_separation = int(sweep_cfg["trials_per_separation"])
    trials_per_batch = max(1, int(sweep_cfg["trials_per_batch"]))
    separation_values_cm = list(
        range(
            int(sweep_cfg["separation_cm_min"]),
            int(sweep_cfg["separation_cm_max"]) + 1,
        )
    )

    rows: List[SeparationSweepResult] = []
    total_trials = len(separation_values_cm) * trials_per_separation
    completed_trials = 0
    framework._print_progress(0, total_trials)

    for separation_cm in separation_values_cm:
        separation_m = separation_cm / 100.0
        distances_m = [
            float(sweep_cfg["base_distance_m"]),
            float(sweep_cfg["base_distance_m"]) + separation_m,
        ]
        scenario_batch_stats: List[SeparationBatchStats] = []
        true_distances_m: List[float] = []
        true_delays: List[int] = []

        for batch_start in range(0, trials_per_separation, trials_per_batch):
            batch_trials = min(trials_per_batch, trials_per_separation - batch_start)
            pulse_spikes, echo_spikes, true_distances_batch, true_delays_batch = (
                framework._simulate_fixed_distance_trial_batch(
                    num_trials=batch_trials,
                    distances_m=distances_m,
                    echo_cfg=echo_cfg,
                )
            )
            true_distances_m = list(true_distances_batch[0])
            true_delays = list(true_delays_batch[0])
            batch_stats = _evaluate_separation_detection_batch(
                framework=framework,
                net=net,
                readout=readout,
                pulse_spikes=pulse_spikes,
                echo_spikes=echo_spikes,
                true_delays=true_delays,
                tolerance_steps=int(sweep_cfg["tolerance_steps"]),
            )
            scenario_batch_stats.append(batch_stats)
            completed_trials += batch_trials
            framework._print_progress(completed_trials, total_trials)

        rows.append(
            _summarize_separation_stats(
                framework=framework,
                separation_cm=separation_cm,
                separation_m=separation_m,
                true_distances_m=true_distances_m,
                true_delays=true_delays,
                batch_stats=scenario_batch_stats,
            )
        )

    if total_trials > 0:
        print()

    return rows


def save_separation_sweep_csv(
    rows: Sequence[SeparationSweepResult],
    output_path: Path = OUTPUT_CSV_PATH,
) -> None:
    fieldnames = [
        "separation_cm",
        "separation_m",
        "separation_steps",
        "r1_m",
        "r2_m",
        "true_delay_1",
        "true_delay_2",
        "trials",
        "detection_error_trials",
        "detection_error_probability",
        "miss_rate",
        "false_positive_rate",
        "mean_num_detections",
        "mean_abs_distance_error_m",
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
                    "trials": row.trials,
                    "detection_error_trials": row.detection_error_trials,
                    "detection_error_probability": row.detection_error_probability,
                    "miss_rate": row.miss_rate,
                    "false_positive_rate": row.false_positive_rate,
                    "mean_num_detections": row.mean_num_detections,
                    "mean_abs_distance_error_m": row.mean_abs_distance_error_m,
                }
            )


def plot_detection_error_vs_separation(
    rows: Sequence[SeparationSweepResult],
    output_path: Path = OUTPUT_FIG_PATH,
) -> None:
    if not rows:
        raise RuntimeError("No separation sweep rows were generated.")

    mpl_cache_dir = Path(__file__).with_name(".mplcache")
    mpl_cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    separation_cm = np.asarray([row.separation_cm for row in rows], dtype=float)
    detection_error = np.asarray(
        [row.detection_error_probability for row in rows],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(
        separation_cm,
        detection_error,
        marker="o",
        markersize=4,
        linewidth=2,
        color="#D86B6B",
    )
    ax.set_title("Detection Error vs Object Separation")
    ax.set_xlabel("Object separation (cm)")
    ax.set_ylabel("Detection error probability")
    ax.set_xlim(separation_cm.min(), separation_cm.max())
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.5,
        -0.22,
        (
            f"Two objects, r1={rows[0].true_distances_m[0]:.2f} m, "
            f"r2=r1+separation, snr_db=None, clutter=0, "
            f"n={rows[0].trials} trials/scenario"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_separation_sweep_summary(rows: Sequence[SeparationSweepResult]) -> None:
    print("\n" + "=" * 112)
    print("TWO-OBJECT CLEAN SEPARATION SWEEP")
    print("=" * 112)
    print(
        "sep_cm | bank_steps | true_delays | errors/trials | det_err | miss_rate | fp_rate | mean_detections"
    )
    for row in rows:
        true_delay_text = f"[{row.true_delays[0]}, {row.true_delays[1]}]"
        error_count_text = f"{row.detection_error_trials}/{row.trials}"
        print(
            f"{row.separation_cm:6d} | {row.separation_steps:10.2f} | "
            f"{true_delay_text:11s} | {error_count_text:13s} | "
            f"{row.detection_error_probability:7.3f} | {row.miss_rate:9.3f} | "
            f"{row.false_positive_rate:7.3f} | {row.mean_num_detections:15.2f}"
        )


def main() -> None:
    sweep_cfg = {
        "base_distance_m": 1.50,
        "separation_cm_min": 1,
        "separation_cm_max": 50,
        "trials_per_separation": 50,
        "trials_per_batch": 50,
        "reflectivity": 1.0,
        "surface_roughness": 0.0,
        "surface_texture_scale_m": 0.0,
        "air_absorption_db_per_m_100khz": 0.0,
        "tolerance_steps": 2,
        "seed": 420,
    }
    network_cfg = {
        "weight": 0.1,
        "beta": 0.1,
        "threshold": 0.7,
        "rs": 2,
        "lis": 3,
        "lir": 8,
        "inhib_decay": 0.1,
    }
    readout_cfg = {
        "beta": 0.4,
        "threshold": 0.4,
        "is": 0,
        "ir": 4,
        "inhibition_falloff": "gaussian",
        "sigma": 2.0,
    }

    torch.manual_seed(sweep_cfg["seed"])
    np.random.seed(sweep_cfg["seed"])

    framework = FMSweepGridSearchFramework(seed=sweep_cfg["seed"])
    rows = run_two_object_separation_sweep(
        framework=framework,
        network_cfg=network_cfg,
        readout_cfg=readout_cfg,
        sweep_cfg=sweep_cfg,
    )

    print_separation_sweep_summary(rows)
    save_separation_sweep_csv(rows)
    plot_detection_error_vs_separation(rows)
    print(f"\nSaved CSV to {OUTPUT_CSV_PATH}")
    print(f"Saved figure to {OUTPUT_FIG_PATH}")


if __name__ == "__main__":
    main()
