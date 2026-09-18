from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from sim_modules.echo_range import generate_fm_echo_from_waveform
from sim_modules.encoder_range import TonotopicEncoder
from sim_modules.network_range import FMSweepBankNetwork


@dataclass
class MatchMetrics:
    misses: int
    false_positives: int
    abs_delay_errors: List[float]
    abs_distance_errors_m: List[float]


@dataclass
class TrialResult:
    num_objects: int
    true_distances_m: List[float]
    true_delays: List[int]
    detected_delays: List[int]
    decoded_delays: List[float]
    decoded_distances_m: List[float]
    detection_misses: int
    detection_false_positives: int
    population_misses: int
    population_false_positives: int
    population_abs_delay_errors: List[float]
    population_abs_distance_errors_m: List[float]
    has_detection_error: int
    has_population_error: int


class FMSweepEvaluationFramework:
    """Monte Carlo evaluation for multi-object detection and localization."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

        self.sim_cfg = {
            "device": "cpu",
            "num_steps": 250,
            "sample_rate_hz": 1_000_000,
            "step_duration_ms": 0.1,
        }
        self.encoder_cfg = {
            "freq_channels": 30,
            "f_min": 30,
            "f_max": 90,
        }
        self.network_cfg = {
            "delays_start": 1,
            "delays_stop": 200,
            "weight": 0.2,
            "beta": 0.3,
            "threshold": 0.5,
            "refractory_steps": 0,
            "lateral_inhibition_strength": 0.0,
            "lateral_inhibition_radius": 3,
            "inhibition_decay": 0.9,
        }
        self.pulse_cfg = {
            "pulse_start_time": 5,
            "duration": 8,
            "f_start": 90,
            "f_end": 30,
            "amplitude": 1.0,
        }
        self.pulse_encoding_cfg = {
            "max_spike_prob": 0.9,
            "filter_taps": 63,
            "bandwidth_khz": 8.0,
            "smoothing_frac": 0.8,
        }
        self.echo_default_cfg = {
            "max_spike_prob": 0.9,
            "speed_of_sound_m_s": 343.0,
            "air_absorption_db_per_m_100khz": 0.5,
            "surface_roughness": 0.25,
            "surface_texture_scale_m": 0.002,
            "filter_taps": 63,
            "bandwidth_khz": 8.0,
            "smoothing_frac": 0.8,
            "snr_db": None,
            "clutter_fraction": 0.0,
            "num_clutter_paths": 50,
            "clutter_min_distance_m": 0.05,
            "clutter_max_distance_m": 10.0,
            "clutter_reflectivity_range": (0.2, 0.6),
            "clutter_roughness_range": (0.1, 0.3),
            "clutter_texture_scale_range_m": (0.001, 0.004),
        }
        self.detect_cfg = {
            "min_spikes": 1,
            "min_peak_spikes": 2,
            "min_object_spikes": 2,
            "allow_multiple_estimates_per_true": True,
        }
        self.scene_cfg = {
            "min_separation_steps": 2,
            "max_sampling_attempts": 256,
        }

        self.device = self.sim_cfg["device"]
        self.encoder = TonotopicEncoder(**self.encoder_cfg)
        self.candidate_delays = np.arange(self.network_cfg["delays_start"], self.network_cfg["delays_stop"])
        self._pulse_cache_key = None
        self._cached_pulse_waveform: Optional[torch.Tensor] = None
        self._cached_pulse_spike_probs: Optional[torch.Tensor] = None
        self.net = FMSweepBankNetwork(
            delays=self.candidate_delays,
            freq_channels=self.encoder_cfg["freq_channels"],
            weight=self.network_cfg["weight"],
            beta=self.network_cfg["beta"],
            threshold=self.network_cfg["threshold"],
            refractory_steps=self.network_cfg["refractory_steps"],
            lateral_inhibition_strength=self.network_cfg["lateral_inhibition_strength"],
            lateral_inhibition_radius=self.network_cfg["lateral_inhibition_radius"],
            inhibition_decay=self.network_cfg["inhibition_decay"],
        ).to(self.device)

    def _delay_steps_to_distance_m(self, delay_steps: float) -> float:
        step_duration_s = self.sim_cfg["step_duration_ms"] * 1e-3
        speed = self.echo_default_cfg["speed_of_sound_m_s"]
        return 0.5 * float(delay_steps) * step_duration_s * speed

    def _sample_distances_m(
        self,
        num_objects: int,
        min_distance_m: float,
        max_distance_m: float,
        min_separation_steps: int,
    ) -> np.ndarray:
        min_distance_m = max(0.0, float(min_distance_m))
        max_distance_m = max(min_distance_m, float(max_distance_m))
        min_separation_steps = max(0, int(min_separation_steps))

        accepted: List[float] = []
        accepted_delays: List[int] = []

        for _ in range(self.scene_cfg["max_sampling_attempts"]):
            if len(accepted) == int(num_objects):
                break

            distance_m = float(self.rng.uniform(min_distance_m, max_distance_m))
            delay_steps = self._distance_to_delay_steps(distance_m)
            if delay_steps not in self.candidate_delays:
                continue

            if any(abs(delay_steps - existing) < min_separation_steps for existing in accepted_delays):
                continue

            accepted.append(distance_m)
            accepted_delays.append(delay_steps)

        if len(accepted) != int(num_objects):
            raise RuntimeError(
                "Could not sample a valid multi-object scene. "
                "Reduce `min_separation_steps` or widen the distance range."
            )

        return np.sort(np.asarray(accepted, dtype=float))

    def _distance_to_delay_steps(self, distance_m: float) -> int:
        step_duration_s = self.sim_cfg["step_duration_ms"] * 1e-3
        speed = self.echo_default_cfg["speed_of_sound_m_s"]
        return int(round((2.0 * distance_m / speed) / step_duration_s))

    def _pulse_cache_signature(self) -> Tuple[object, ...]:
        return (
            self.device,
            self.encoder.freq_channels,
            self.encoder.f_min,
            self.encoder.f_max,
            self.encoder.tuning_width,
            self.sim_cfg["num_steps"],
            self.sim_cfg["sample_rate_hz"],
            self.sim_cfg["step_duration_ms"],
            self.pulse_cfg["pulse_start_time"],
            self.pulse_cfg["duration"],
            self.pulse_cfg["f_start"],
            self.pulse_cfg["f_end"],
            self.pulse_cfg["amplitude"],
            self.pulse_encoding_cfg["max_spike_prob"],
            self.pulse_encoding_cfg["filter_taps"],
            self.pulse_encoding_cfg["bandwidth_khz"],
            self.pulse_encoding_cfg["smoothing_frac"],
        )

    def _get_pulse_template(self) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = self._pulse_cache_signature()
        if (
            self._pulse_cache_key != cache_key
            or self._cached_pulse_waveform is None
            or self._cached_pulse_spike_probs is None
        ):
            pulse_waveform = self.encoder.generate_fm_sweep_waveform(
                num_steps=self.sim_cfg["num_steps"],
                start_time=self.pulse_cfg["pulse_start_time"],
                duration=self.pulse_cfg["duration"],
                f_start=self.pulse_cfg["f_start"],
                f_end=self.pulse_cfg["f_end"],
                sample_rate_hz=self.sim_cfg["sample_rate_hz"],
                step_duration_ms=self.sim_cfg["step_duration_ms"],
                amplitude=self.pulse_cfg["amplitude"],
                device=self.device,
            )
            pulse_spike_probs = self.encoder.compute_spike_probabilities_from_waveform(
                waveform=pulse_waveform,
                num_steps=self.sim_cfg["num_steps"],
                max_spike_prob=self.pulse_encoding_cfg["max_spike_prob"],
                sample_rate_hz=self.sim_cfg["sample_rate_hz"],
                step_duration_ms=self.sim_cfg["step_duration_ms"],
                filter_taps=self.pulse_encoding_cfg["filter_taps"],
                bandwidth_khz=self.pulse_encoding_cfg["bandwidth_khz"],
                smoothing_frac=self.pulse_encoding_cfg["smoothing_frac"],
                device=self.device,
            )
            self._pulse_cache_key = cache_key
            self._cached_pulse_waveform = pulse_waveform
            self._cached_pulse_spike_probs = pulse_spike_probs

        return self._cached_pulse_waveform, self._cached_pulse_spike_probs

    def _build_pulse(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        pulse_waveform, pulse_spike_probs = self._get_pulse_template()
        pulse_spikes = self.encoder.sample_spikes_from_probabilities(
            spike_probs=pulse_spike_probs,
            batch_size=batch_size,
            device=self.device,
        )
        return pulse_waveform, pulse_spikes

    def _match_estimates(
        self,
        true_delays: Sequence[float],
        estimated_delays: Sequence[float],
        tolerance_steps: int,
        allow_multiple_estimates_per_true: bool = False,
    ) -> MatchMetrics:
        true_sorted = [float(d) for d in sorted(true_delays)]
        est_sorted = [float(d) for d in sorted(estimated_delays)]
        matched_true = set()
        matched_est = set()
        pairs = []

        for i, true_delay in enumerate(true_sorted):
            for j, est_delay in enumerate(est_sorted):
                abs_err = abs(true_delay - est_delay)
                if abs_err <= float(tolerance_steps):
                    pairs.append((abs_err, i, j))
        pairs.sort(key=lambda item: item[0])

        abs_delay_errors: List[float] = []
        abs_distance_errors_m: List[float] = []
        for abs_err, i, j in pairs:
            if j in matched_est:
                continue
            if not allow_multiple_estimates_per_true and i in matched_true:
                continue
            matched_true.add(i)
            matched_est.add(j)
            abs_delay_errors.append(float(abs_err))
            abs_distance_errors_m.append(
                abs(self._delay_steps_to_distance_m(true_sorted[i]) - self._delay_steps_to_distance_m(est_sorted[j]))
            )

        return MatchMetrics(
            misses=len(true_sorted) - len(matched_true),
            false_positives=len(est_sorted) - len(matched_est),
            abs_delay_errors=abs_delay_errors,
            abs_distance_errors_m=abs_distance_errors_m,
        )

    def _simulate_scene_with_pulse(
        self,
        pulse_waveform: torch.Tensor,
        pulse_spikes: torch.Tensor,
        num_objects: int,
        min_distance_m: float,
        max_distance_m: float,
        reflectivity_range: Tuple[float, float],
        roughness_range: Tuple[float, float],
        min_separation_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[int]]:
        distances = self._sample_distances_m(
            num_objects=num_objects,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
            min_separation_steps=min_separation_steps,
        )

        object_echoes = []
        true_distances_m: List[float] = []
        true_delays: List[int] = []

        for distance_m in distances:
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
        return pulse_spikes, echo_spikes, sorted(true_distances_m), sorted(true_delays)

    def _simulate_scene(
        self,
        num_objects: int,
        min_distance_m: float,
        max_distance_m: float,
        reflectivity_range: Tuple[float, float],
        roughness_range: Tuple[float, float],
        min_separation_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[int]]:
        pulse_waveform, pulse_spikes = self._build_pulse()
        return self._simulate_scene_with_pulse(
            pulse_waveform=pulse_waveform,
            pulse_spikes=pulse_spikes,
            num_objects=num_objects,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
            reflectivity_range=reflectivity_range,
            roughness_range=roughness_range,
            min_separation_steps=min_separation_steps,
        )

    def run_trial(
        self,
        num_objects: int,
        min_distance_m: float,
        max_distance_m: float,
        reflectivity_range: Tuple[float, float],
        roughness_range: Tuple[float, float],
        tolerance_steps: int,
        min_separation_steps: Optional[int] = None,
    ) -> TrialResult:
        pulse_spikes, echo_spikes, true_distances_m, true_delays = self._simulate_scene(
            num_objects=num_objects,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
            reflectivity_range=reflectivity_range,
            roughness_range=roughness_range,
            min_separation_steps=(
                self.scene_cfg["min_separation_steps"]
                if min_separation_steps is None
                else int(min_separation_steps)
            ),
        )

        bank_spikes = self.net.forward(pulse_spikes, echo_spikes)
        spike_counts = bank_spikes.sum(dim=0)
        if spike_counts.ndim != 2 or spike_counts.shape[0] != 1:
            raise ValueError("Evaluation currently expects batch size 1.")

        spike_counts_np = spike_counts[0].detach().cpu().numpy()
        active_indices = np.flatnonzero(spike_counts_np >= float(self.detect_cfg["min_spikes"]))
        detected_delays = [int(self.candidate_delays[idx]) for idx in active_indices.tolist()]

        decoded_objects = self.net.decode_population_response(
            spike_counts_np,
            min_peak_spikes=self.detect_cfg["min_peak_spikes"],
            min_object_spikes=self.detect_cfg["min_object_spikes"],
        )
        decoded_delays = [float(obj.delay_steps) for obj in decoded_objects]
        decoded_distances_m = [self._delay_steps_to_distance_m(delay) for delay in decoded_delays]

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

        return TrialResult(
            num_objects=num_objects,
            true_distances_m=true_distances_m,
            true_delays=true_delays,
            detected_delays=sorted(detected_delays),
            decoded_delays=sorted(decoded_delays),
            decoded_distances_m=sorted(decoded_distances_m),
            detection_misses=detection_metrics.misses,
            detection_false_positives=detection_metrics.false_positives,
            population_misses=population_metrics.misses,
            population_false_positives=population_metrics.false_positives,
            population_abs_delay_errors=population_metrics.abs_delay_errors,
            population_abs_distance_errors_m=population_metrics.abs_distance_errors_m,
            has_detection_error=int(detection_metrics.misses > 0 or detection_metrics.false_positives > 0),
            has_population_error=int(population_metrics.misses > 0 or population_metrics.false_positives > 0),
        )

    def evaluate(
        self,
        object_counts: Sequence[int],
        trials_per_count: int,
        min_distance_m: float,
        max_distance_m: float,
        reflectivity_range: Tuple[float, float],
        roughness_range: Tuple[float, float],
        tolerance_steps: int,
        min_separation_steps: Optional[int] = None,
    ) -> Tuple[List[TrialResult], Dict[int, Dict[str, float]]]:
        results = []

        for n_objects in object_counts:
            for _ in range(trials_per_count):
                results.append(
                    self.run_trial(
                        num_objects=int(n_objects),
                        min_distance_m=min_distance_m,
                        max_distance_m=max_distance_m,
                        reflectivity_range=reflectivity_range,
                        roughness_range=roughness_range,
                        tolerance_steps=tolerance_steps,
                        min_separation_steps=min_separation_steps,
                    )
                )

        summary: Dict[int, Dict[str, float]] = {}
        for n_objects in object_counts:
            group = [result for result in results if result.num_objects == int(n_objects)]
            if not group:
                continue

            trials = len(group)
            total_true = sum(len(result.true_delays) for result in group)
            total_detected = sum(len(result.detected_delays) for result in group)
            total_decoded = sum(len(result.decoded_delays) for result in group)

            detection_misses = sum(result.detection_misses for result in group)
            detection_fp = sum(result.detection_false_positives for result in group)
            population_misses = sum(result.population_misses for result in group)
            population_fp = sum(result.population_false_positives for result in group)

            population_delay_errors = [
                err for result in group for err in result.population_abs_delay_errors
            ]
            population_distance_errors_m = [
                err for result in group for err in result.population_abs_distance_errors_m
            ]

            summary[int(n_objects)] = {
                "trials": float(trials),
                "detection_error_probability": sum(result.has_detection_error for result in group) / float(trials),
                "detection_miss_rate": detection_misses / float(total_true) if total_true > 0 else 0.0,
                "detection_false_positive_rate": (
                    detection_fp / float(total_detected) if total_detected > 0 else 0.0
                ),
                "population_error_probability": sum(result.has_population_error for result in group) / float(trials),
                "population_miss_rate": population_misses / float(total_true) if total_true > 0 else 0.0,
                "population_false_positive_rate": (
                    population_fp / float(total_decoded) if total_decoded > 0 else 0.0
                ),
                "mean_population_abs_delay_error": (
                    float(np.mean(population_delay_errors)) if population_delay_errors else 0.0
                ),
                "mean_population_abs_distance_error_m": (
                    float(np.mean(population_distance_errors_m)) if population_distance_errors_m else 0.0
                ),
            }

        return results, summary


def _parse_int_list(csv_values: str) -> List[int]:
    return [int(value.strip()) for value in csv_values.split(",") if value.strip()]


def main() -> None:
    eval_cfg = {
        "object_counts": "1,2,3",
        "trials_per_count": 100,
        "min_distance_m": 0.15,
        "max_distance_m": 3.4,
        "reflectivity_min": 0.1,
        "reflectivity_max": 0.8,
        "roughness_min": 0.1,
        "roughness_max": 0.35,
        "tolerance_steps": 10 ,
        "min_separation_steps": 2,
        "seed": 42,
    }

    torch.manual_seed(eval_cfg["seed"])
    np.random.seed(eval_cfg["seed"])

    framework = FMSweepEvaluationFramework(seed=eval_cfg["seed"])
    object_counts = _parse_int_list(eval_cfg["object_counts"])

    results, summary = framework.evaluate(
        object_counts=object_counts,
        trials_per_count=eval_cfg["trials_per_count"],
        min_distance_m=eval_cfg["min_distance_m"],
        max_distance_m=eval_cfg["max_distance_m"],
        reflectivity_range=(eval_cfg["reflectivity_min"], eval_cfg["reflectivity_max"]),
        roughness_range=(eval_cfg["roughness_min"], eval_cfg["roughness_max"]),
        tolerance_steps=eval_cfg["tolerance_steps"],
        min_separation_steps=eval_cfg["min_separation_steps"],
    )

    print("=" * 124)
    print("FM-FM MULTI-OBJECT EVALUATION")
    print("=" * 124)
    print(
        "objects | trials | det_err | det_miss | det_fp | pop_err | pop_miss | pop_fp | mean_|dist|"
    )
    for n_objects in object_counts:
        if n_objects not in summary:
            continue
        stats = summary[n_objects]
        print(
            f"{n_objects:7d} | {int(stats['trials']):6d} | "
            f"{stats['detection_error_probability']:.3f}   | {stats['detection_miss_rate']:.3f}    | "
            f"{stats['detection_false_positive_rate']:.3f} | {stats['population_error_probability']:.3f}   | "
            f"{stats['population_miss_rate']:.3f}     | {stats['population_false_positive_rate']:.3f}  | "
            f"{stats['mean_population_abs_distance_error_m']:.4f}"
        )

    if results:
        example = next((result for result in results if result.num_objects == object_counts[-1]), results[0])
        print("\nExample trial")
        print(f"True delays:      {example.true_delays}")
        print(f"Detected Threshold hits:   {example.detected_delays}")
        print(f"Decoded delays:   {[round(delay, 2) for delay in example.decoded_delays]}")
        print(f"True distances m: {[round(distance, 3) for distance in example.true_distances_m]}")
        print(f"Decoded dist. m:  {[round(distance, 3) for distance in example.decoded_distances_m]}")


if __name__ == "__main__":
    main()
