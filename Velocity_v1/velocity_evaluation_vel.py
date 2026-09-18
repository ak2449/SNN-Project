from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import os
import sys
from typing import List, Optional, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "snn_velocity_mplcache"),
)

import matplotlib.pyplot as plt
import numpy as np
import torch

PLOT_BLUE = "#3B6CA8"

try:
    from sim_modules import (
        DecodedObject,
        DopplerDecodedObject,
        DopplerEncoder,
        FMSweepBankNetwork,
        FovealDopplerBankNetwork,
        TonotopicEncoder,
        generate_fm_echo_from_waveform,
    )
except ModuleNotFoundError:
    from Velocity_v1.sim_modules import (
        DecodedObject,
        DopplerDecodedObject,
        DopplerEncoder,
        FMSweepBankNetwork,
        FovealDopplerBankNetwork,
        TonotopicEncoder,
        generate_fm_echo_from_waveform,
    )


@dataclass
class VelocityTrialResult:
    trial_index: int
    true_distance_m: float
    true_velocity_m_s: float
    snr_db: Optional[float]
    true_delay_steps: int
    decoded_delay_steps: Optional[float]
    decoded_distance_m: Optional[float]
    decoded_shift_channels: Optional[float]
    decoded_velocity_m_s: Optional[float]
    range_abs_error_m: Optional[float]
    velocity_error_m_s: Optional[float]
    velocity_abs_error_m_s: Optional[float]
    joint_error_cells: Optional[float]
    range_detected: bool
    velocity_detected: bool
    range_peak_spikes: float
    velocity_peak_spikes: float


@dataclass
class SweepSummaryRow:
    sweep_name: str
    x_value: float
    trials: int
    valid_velocity_trials: int
    valid_joint_trials: int
    mean_decoded_velocity_m_s: float
    std_decoded_velocity_m_s: float
    mean_velocity_error_m_s: float
    mean_abs_velocity_error_m_s: float
    std_abs_velocity_error_m_s: float
    rmse_velocity_m_s: float
    mean_range_abs_error_m: float
    mean_joint_error_cells: float
    velocity_miss_rate: float
    joint_miss_rate: float
    y_value: Optional[float] = None


class VelocityEvaluationFramework:
    """Evaluation harness for the CF-FM range and velocity demo.

    The framework keeps the same parallel-pathway structure as
    ``full_sim_vel.py``: a broadband FM delay bank decodes range, and the
    decoded delay gates a narrowband foveal Doppler bank for velocity.
    """

    def __init__(self, seed: int = 3):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

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
        self.doppler_encoder_cfg = {
            "cf_freq_khz": 60.0,
            "half_range_khz": 3.0,
            "freq_channels": 60,
        }
        self.network_cfg = {
            "delays_start": 1,
            "delays_stop": 200,
            "weight": 0.1,
            "beta": 0.1,
            "threshold": 0.9,
            "refractory_steps": 2,
            "lateral_inhibition_strength": 3,
            "lateral_inhibition_radius": 8,
            "inhibition_decay": 0.1,
            "decode_radius_channels": 3,
        }
        self.doppler_net_cfg = {
            "shift_range": 25,
            "input_batch_size": 64,
            "spectral_sharpen_power": 12.0,
            "decode_peak_fraction": 0.75,
            "tuning_width_channels": 2.0,
            "gate_threshold": 1.0,
            "decode_radius_channels": 3,
            "weight": 0.2,
            "beta": 0.15,
            "threshold": 0.7,
            "refractory_steps": 2,
            "lateral_inhibition_strength": 2.0,
            "lateral_inhibition_radius": 6,
            "inhibition_decay": 0.1,
        }
        self.cf_pulse_cfg = {
            "pulse_start_time": 5,
            "duration": 8,
            "tone_freq_khz": self.doppler_encoder_cfg["cf_freq_khz"],
            "amplitude": 1.0,
        }
        self.pulse_cfg = {
            "pulse_start_time": 13,
            "duration": 4,
            "f_start": 90,
            "f_end": 30,
            "amplitude": 0.9,
        }
        self.velocity_decode_cfg = {
            "filter_taps": 223,
            "smoothing_frac": 0.8,
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
            "air_absorption_db_per_m_100khz": 0.05,
            "surface_roughness": 0.25,
            "surface_texture_scale_m": 0.002,
            "filter_taps": 63,
            "bandwidth_khz": 8.0,
            "smoothing_frac": 0.8,
            "snr_db": None,
            "clutter_fraction": 0.0,
            "num_clutter_paths": 30,
            "clutter_min_distance_m": 0.05,
            "clutter_max_distance_m": 3.4,
            "clutter_reflectivity_range": (0.2, 0.9),
            "clutter_roughness_range": (0.1, 0.3),
            "clutter_texture_scale_range_m": (0.001, 0.004),
        }
        self.object_default_cfg = {
            "distance_m": 1.15,
            "velocity_m_s": 3.0,
            "reflectivity": 0.9,
            "surface_roughness": 0.15,
        }
        self.detect_cfg = {
            "min_spikes": 2,
            "min_spikes_pop": 1,
        }

        self.device = self.sim_cfg["device"]
        self.encoder = TonotopicEncoder(**self.encoder_cfg)
        self.doppler_encoder = DopplerEncoder(**self.doppler_encoder_cfg)
        self.candidate_delays = np.arange(
            self.network_cfg["delays_start"],
            self.network_cfg["delays_stop"],
        )
        shift_range = int(self.doppler_net_cfg["shift_range"])
        self.candidate_shifts = np.arange(-shift_range, shift_range + 1)
        self.net = self._build_range_net()

        self._pulse_cache_key = None
        self._cached_pulse_waveform: Optional[torch.Tensor] = None
        self._cached_pulse_broadband_probs: Optional[torch.Tensor] = None
        self._cached_pulse_doppler_probs: Optional[torch.Tensor] = None

    @property
    def range_resolution_m(self) -> float:
        return self._delay_steps_to_distance_m(1.0)

    @property
    def velocity_resolution_m_s(self) -> float:
        return abs(
            self.doppler_encoder.shift_channels_to_velocity(
                1.0,
                speed_of_sound_m_s=self.echo_default_cfg["speed_of_sound_m_s"],
            )
        )

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
        if completed >= total:
            sys.stdout.write("\n")

    @staticmethod
    def _as_float_array(values: Sequence[float]) -> np.ndarray:
        return np.asarray([float(value) for value in values], dtype=float)

    @staticmethod
    def _nan_mean(values: Sequence[float]) -> float:
        values_np = np.asarray(values, dtype=float)
        return float(np.nanmean(values_np)) if np.isfinite(values_np).any() else float("nan")

    @staticmethod
    def _nan_std(values: Sequence[float]) -> float:
        values_np = np.asarray(values, dtype=float)
        finite = values_np[np.isfinite(values_np)]
        return float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0 if finite.size == 1 else float("nan")

    @staticmethod
    def _choose_value(
        rng: np.random.Generator,
        value: Optional[float],
        value_range: Optional[Tuple[float, float]],
        default_value: float,
    ) -> float:
        if value is not None:
            return float(value)
        if value_range is None:
            return float(default_value)
        lo, hi = value_range
        return float(rng.uniform(float(lo), float(hi)))

    @staticmethod
    def _save_or_show(fig, save_path: Optional[Path], show: bool) -> None:
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    @staticmethod
    def _axis_edges(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.asarray([0.0, 1.0], dtype=float)
        if values.size == 1:
            return np.asarray([values[0] - 0.5, values[0] + 0.5], dtype=float)

        mids = 0.5 * (values[:-1] + values[1:])
        first = values[0] - (mids[0] - values[0])
        last = values[-1] + (values[-1] - mids[-1])
        return np.concatenate(([first], mids, [last]))

    def _seed_trial(self) -> int:
        trial_seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)
        return trial_seed

    def _delay_steps_to_distance_m(self, delay_steps: float) -> float:
        step_duration_s = self.sim_cfg["step_duration_ms"] * 1e-3
        speed = self.echo_default_cfg["speed_of_sound_m_s"]
        return 0.5 * float(delay_steps) * step_duration_s * speed

    def _distance_to_delay_steps(self, distance_m: float) -> int:
        step_duration_s = self.sim_cfg["step_duration_ms"] * 1e-3
        speed = self.echo_default_cfg["speed_of_sound_m_s"]
        return int(round((2.0 * float(distance_m) / speed) / step_duration_s))

    def _build_range_net(self) -> FMSweepBankNetwork:
        return FMSweepBankNetwork(
            delays=self.candidate_delays,
            freq_channels=self.encoder_cfg["freq_channels"],
            weight=self.network_cfg["weight"],
            beta=self.network_cfg["beta"],
            threshold=self.network_cfg["threshold"],
            refractory_steps=self.network_cfg["refractory_steps"],
            lateral_inhibition_strength=self.network_cfg["lateral_inhibition_strength"],
            lateral_inhibition_radius=self.network_cfg["lateral_inhibition_radius"],
            inhibition_decay=self.network_cfg["inhibition_decay"],
            decode_radius_channels=self.network_cfg["decode_radius_channels"],
        ).to(self.device)

    def _build_doppler_net(self, delay_steps: int) -> FovealDopplerBankNetwork:
        return FovealDopplerBankNetwork(
            freq_shifts=self.candidate_shifts,
            delay=int(delay_steps),
            freq_channels=self.doppler_encoder_cfg["freq_channels"],
            weight=self.doppler_net_cfg["weight"],
            beta=self.doppler_net_cfg["beta"],
            threshold=self.doppler_net_cfg["threshold"],
            refractory_steps=self.doppler_net_cfg["refractory_steps"],
            lateral_inhibition_strength=self.doppler_net_cfg["lateral_inhibition_strength"],
            lateral_inhibition_radius=self.doppler_net_cfg["lateral_inhibition_radius"],
            inhibition_decay=self.doppler_net_cfg["inhibition_decay"],
            tuning_width_channels=self.doppler_net_cfg["tuning_width_channels"],
            gate_threshold=self.doppler_net_cfg["gate_threshold"],
            decode_radius_channels=self.doppler_net_cfg["decode_radius_channels"],
        ).to(self.device)

    def _pulse_cache_signature(self) -> Tuple[object, ...]:
        return (
            self.device,
            tuple(sorted(self.sim_cfg.items())),
            tuple(sorted(self.encoder_cfg.items())),
            tuple(sorted(self.doppler_encoder_cfg.items())),
            tuple(sorted(self.cf_pulse_cfg.items())),
            tuple(sorted(self.pulse_cfg.items())),
            tuple(sorted(self.pulse_encoding_cfg.items())),
            tuple(sorted(self.velocity_decode_cfg.items())),
            float(self.doppler_net_cfg["spectral_sharpen_power"]),
        )

    def _combine_call_components(self, *waveforms: torch.Tensor) -> torch.Tensor:
        combined = torch.stack(waveforms, dim=0).sum(dim=0)
        peak = combined.abs().max().clamp_min(1e-8)
        return combined / peak if peak.item() > 1.0 else combined

    def _sharpen_doppler_probabilities(self, spike_probs: torch.Tensor) -> torch.Tensor:
        power = float(self.doppler_net_cfg["spectral_sharpen_power"])
        if power <= 1.0:
            return spike_probs

        max_spike_prob = max(float(self.pulse_encoding_cfg["max_spike_prob"]), 1e-8)
        normalized = (spike_probs / max_spike_prob).clamp(0.0, 1.0)
        return (normalized.pow(power) * max_spike_prob).clamp(0.0, 1.0)

    def _get_pulse_template(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_key = self._pulse_cache_signature()
        if (
            self._pulse_cache_key == cache_key
            and self._cached_pulse_waveform is not None
            and self._cached_pulse_broadband_probs is not None
            and self._cached_pulse_doppler_probs is not None
        ):
            return (
                self._cached_pulse_waveform,
                self._cached_pulse_broadband_probs,
                self._cached_pulse_doppler_probs,
            )

        pulse_waveform_cf = self.doppler_encoder.generate_tone_waveform(
            num_steps=self.sim_cfg["num_steps"],
            start_time=self.cf_pulse_cfg["pulse_start_time"],
            duration=self.cf_pulse_cfg["duration"],
            freq_khz=self.cf_pulse_cfg["tone_freq_khz"],
            sample_rate_hz=self.sim_cfg["sample_rate_hz"],
            step_duration_ms=self.sim_cfg["step_duration_ms"],
            amplitude=self.cf_pulse_cfg["amplitude"],
            device=self.device,
        )
        pulse_waveform_fm = self.encoder.generate_fm_sweep_waveform(
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
        pulse_waveform = self._combine_call_components(pulse_waveform_cf, pulse_waveform_fm)

        pulse_broadband_probs = self.encoder.compute_spike_probabilities_from_waveform(
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
        pulse_doppler_probs = self.doppler_encoder.compute_spike_probabilities_from_waveform(
            waveform=pulse_waveform,
            num_steps=self.sim_cfg["num_steps"],
            max_spike_prob=self.pulse_encoding_cfg["max_spike_prob"],
            sample_rate_hz=self.sim_cfg["sample_rate_hz"],
            step_duration_ms=self.sim_cfg["step_duration_ms"],
            filter_taps=self.velocity_decode_cfg["filter_taps"],
            smoothing_frac=self.velocity_decode_cfg["smoothing_frac"],
            device=self.device,
        )
        pulse_doppler_probs = self._sharpen_doppler_probabilities(pulse_doppler_probs)

        self._pulse_cache_key = cache_key
        self._cached_pulse_waveform = pulse_waveform
        self._cached_pulse_broadband_probs = pulse_broadband_probs
        self._cached_pulse_doppler_probs = pulse_doppler_probs
        return pulse_waveform, pulse_broadband_probs, pulse_doppler_probs

    def _build_pulse(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pulse_waveform, pulse_broadband_probs, pulse_doppler_probs = self._get_pulse_template()
        pulse_broadband = self.encoder.sample_spikes_from_probabilities(
            spike_probs=pulse_broadband_probs,
            batch_size=1,
            device=self.device,
        )
        pulse_doppler = self.doppler_encoder.sample_spikes_from_probabilities(
            spike_probs=pulse_doppler_probs,
            batch_size=self.doppler_net_cfg["input_batch_size"],
            device=self.device,
        )
        return pulse_waveform, pulse_broadband, pulse_doppler

    def _simulate_echo(
        self,
        pulse_waveform: torch.Tensor,
        distance_m: float,
        velocity_m_s: float,
        reflectivity: float,
        surface_roughness: float,
        snr_db: Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        obj_echo_cfg = dict(self.echo_default_cfg)
        obj_echo_cfg.update(
            {
                "distance_m": float(distance_m),
                "velocity_m_s": float(velocity_m_s),
                "reflectivity": float(reflectivity),
                "surface_roughness": float(surface_roughness),
                "snr_db": snr_db,
            }
        )

        echo_broadband, echo_waveform, true_delay = generate_fm_echo_from_waveform(
            pulse_waveform=pulse_waveform,
            encoder=self.encoder,
            num_steps=self.sim_cfg["num_steps"],
            distance_m=obj_echo_cfg["distance_m"],
            reflectivity=obj_echo_cfg["reflectivity"],
            velocity_m_s=obj_echo_cfg["velocity_m_s"],
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

        echo_doppler_probs = self.doppler_encoder.compute_spike_probabilities_from_waveform(
            waveform=echo_waveform,
            num_steps=self.sim_cfg["num_steps"],
            max_spike_prob=obj_echo_cfg["max_spike_prob"],
            sample_rate_hz=self.sim_cfg["sample_rate_hz"],
            step_duration_ms=self.sim_cfg["step_duration_ms"],
            filter_taps=self.velocity_decode_cfg["filter_taps"],
            smoothing_frac=self.velocity_decode_cfg["smoothing_frac"],
            device=self.device,
        )
        echo_doppler = self.doppler_encoder.sample_spikes_from_probabilities(
            spike_probs=self._sharpen_doppler_probabilities(echo_doppler_probs),
            batch_size=self.doppler_net_cfg["input_batch_size"],
            device=self.device,
        )

        return echo_broadband, echo_doppler, int(true_delay)

    def _decode_range(
        self,
        pulse_broadband: torch.Tensor,
        echo_broadband: torch.Tensor,
    ) -> Tuple[Optional[DecodedObject], torch.Tensor, np.ndarray]:
        bank_spikes = self.net.forward(pulse_broadband, echo_broadband)
        spike_counts_np = bank_spikes.sum(dim=0)[0].detach().cpu().numpy()

        decoded_objects = self.net.decode_population_response(
            spike_counts_np,
            min_peak_spikes=self.detect_cfg["min_spikes"],
            min_object_spikes=self.detect_cfg["min_spikes_pop"],
        )
        decoded_objects.sort(
            key=lambda obj: (obj.peak_spikes, obj.total_spikes),
            reverse=True,
        )
        decoded_objects = decoded_objects[:1]

        if not decoded_objects and spike_counts_np.max() > 0:
            best_idx = int(np.argmax(spike_counts_np))
            best_delay = float(self.candidate_delays[best_idx])
            best_spikes = float(spike_counts_np[best_idx])
            decoded_objects = [
                DecodedObject(
                    delay_steps=best_delay,
                    peak_delay_steps=int(best_delay),
                    start_delay_steps=int(best_delay),
                    end_delay_steps=int(best_delay),
                    total_spikes=best_spikes,
                    peak_spikes=best_spikes,
                )
            ]

        decoded_object = decoded_objects[0] if decoded_objects else None
        return decoded_object, bank_spikes, spike_counts_np

    def _decode_velocity(
        self,
        pulse_doppler: torch.Tensor,
        echo_doppler: torch.Tensor,
        decoded_delay: Optional[DecodedObject],
    ) -> Tuple[Optional[DopplerDecodedObject], torch.Tensor, np.ndarray]:
        if decoded_delay is None:
            empty_spikes = torch.zeros(
                self.sim_cfg["num_steps"],
                1,
                len(self.candidate_shifts),
                device=self.device,
            )
            empty_counts = np.zeros(len(self.candidate_shifts), dtype=float)
            return None, empty_spikes, empty_counts

        doppler_net = self._build_doppler_net(int(round(decoded_delay.delay_steps)))
        doppler_bank_spikes = doppler_net.forward(pulse_doppler, echo_doppler)
        doppler_population_spikes = doppler_bank_spikes.sum(dim=1, keepdim=True)
        spike_counts_np = doppler_population_spikes.sum(dim=0)[0].detach().cpu().numpy()

        peak_threshold = max(
            float(self.detect_cfg["min_spikes"]),
            float(spike_counts_np.max()) * float(self.doppler_net_cfg["decode_peak_fraction"]),
        )
        decoded_objects = doppler_net.decode_population_response(
            spike_counts_np,
            min_peak_spikes=peak_threshold,
            min_object_spikes=max(float(self.detect_cfg["min_spikes_pop"]), peak_threshold),
            doppler_encoder=self.doppler_encoder,
            speed_of_sound_m_s=self.echo_default_cfg["speed_of_sound_m_s"],
        )
        decoded_objects.sort(key=lambda obj: obj.peak_spikes, reverse=True)
        decoded_objects = decoded_objects[:1]

        if not decoded_objects and spike_counts_np.max() > 0:
            best_idx = int(np.argmax(spike_counts_np))
            best_shift = float(self.candidate_shifts[best_idx])
            best_spikes = float(spike_counts_np[best_idx])
            decoded_objects = [
                DopplerDecodedObject(
                    shift_channels=best_shift,
                    peak_shift_channels=int(best_shift),
                    start_shift_channels=int(best_shift),
                    end_shift_channels=int(best_shift),
                    total_spikes=best_spikes,
                    peak_spikes=best_spikes,
                    velocity_m_s=self.doppler_encoder.shift_channels_to_velocity(
                        best_shift,
                        speed_of_sound_m_s=self.echo_default_cfg["speed_of_sound_m_s"],
                    ),
                )
            ]

        decoded_object = decoded_objects[0] if decoded_objects else None
        return decoded_object, doppler_population_spikes, spike_counts_np

    def _joint_error_cells(
        self,
        range_abs_error_m: Optional[float],
        velocity_abs_error_m_s: Optional[float],
    ) -> Optional[float]:
        if range_abs_error_m is None or velocity_abs_error_m_s is None:
            return None

        range_cells = float(range_abs_error_m) / max(self.range_resolution_m, 1e-12)
        velocity_cells = float(velocity_abs_error_m_s) / max(self.velocity_resolution_m_s, 1e-12)
        return float(np.sqrt(range_cells ** 2 + velocity_cells ** 2))

    def run_trial(
        self,
        trial_index: int,
        distance_m: Optional[float] = None,
        velocity_m_s: Optional[float] = None,
        snr_db: Optional[float] = None,
        reflectivity: Optional[float] = None,
        reflectivity_range: Optional[Tuple[float, float]] = None,
        surface_roughness: Optional[float] = None,
        roughness_range: Optional[Tuple[float, float]] = None,
    ) -> VelocityTrialResult:
        self._seed_trial()

        true_distance_m = (
            self.object_default_cfg["distance_m"] if distance_m is None else float(distance_m)
        )
        true_velocity_m_s = (
            self.object_default_cfg["velocity_m_s"] if velocity_m_s is None else float(velocity_m_s)
        )
        reflectivity_value = self._choose_value(
            self.rng,
            reflectivity,
            reflectivity_range,
            self.object_default_cfg["reflectivity"],
        )
        roughness_value = self._choose_value(
            self.rng,
            surface_roughness,
            roughness_range,
            self.object_default_cfg["surface_roughness"],
        )

        pulse_waveform, pulse_broadband, pulse_doppler = self._build_pulse()
        echo_broadband, echo_doppler, true_delay_steps = self._simulate_echo(
            pulse_waveform=pulse_waveform,
            distance_m=true_distance_m,
            velocity_m_s=true_velocity_m_s,
            reflectivity=reflectivity_value,
            surface_roughness=roughness_value,
            snr_db=snr_db,
        )

        decoded_delay, _, _ = self._decode_range(pulse_broadband, echo_broadband)
        decoded_velocity, _, _ = self._decode_velocity(
            pulse_doppler=pulse_doppler,
            echo_doppler=echo_doppler,
            decoded_delay=decoded_delay,
        )

        decoded_delay_steps = None
        decoded_distance_m = None
        range_abs_error_m = None
        range_peak_spikes = 0.0
        if decoded_delay is not None:
            decoded_delay_steps = float(decoded_delay.delay_steps)
            decoded_distance_m = self._delay_steps_to_distance_m(decoded_delay_steps)
            range_abs_error_m = abs(decoded_distance_m - true_distance_m)
            range_peak_spikes = float(decoded_delay.peak_spikes)

        decoded_shift_channels = None
        decoded_velocity_m_s = None
        velocity_error_m_s = None
        velocity_abs_error_m_s = None
        velocity_peak_spikes = 0.0
        if decoded_velocity is not None:
            decoded_shift_channels = float(decoded_velocity.shift_channels)
            decoded_velocity_m_s = float(decoded_velocity.velocity_m_s)
            velocity_error_m_s = decoded_velocity_m_s - true_velocity_m_s
            velocity_abs_error_m_s = abs(velocity_error_m_s)
            velocity_peak_spikes = float(decoded_velocity.peak_spikes)

        joint_error_cells = self._joint_error_cells(range_abs_error_m, velocity_abs_error_m_s)

        return VelocityTrialResult(
            trial_index=int(trial_index),
            true_distance_m=float(true_distance_m),
            true_velocity_m_s=float(true_velocity_m_s),
            snr_db=snr_db,
            true_delay_steps=int(true_delay_steps),
            decoded_delay_steps=decoded_delay_steps,
            decoded_distance_m=decoded_distance_m,
            decoded_shift_channels=decoded_shift_channels,
            decoded_velocity_m_s=decoded_velocity_m_s,
            range_abs_error_m=range_abs_error_m,
            velocity_error_m_s=velocity_error_m_s,
            velocity_abs_error_m_s=velocity_abs_error_m_s,
            joint_error_cells=joint_error_cells,
            range_detected=decoded_delay is not None,
            velocity_detected=decoded_velocity is not None,
            range_peak_spikes=range_peak_spikes,
            velocity_peak_spikes=velocity_peak_spikes,
        )

    def summarize_results(
        self,
        sweep_name: str,
        x_value: float,
        results: Sequence[VelocityTrialResult],
        y_value: Optional[float] = None,
    ) -> SweepSummaryRow:
        decoded_velocities = [
            result.decoded_velocity_m_s
            for result in results
            if result.decoded_velocity_m_s is not None
        ]
        velocity_errors = [
            result.velocity_error_m_s
            for result in results
            if result.velocity_error_m_s is not None
        ]
        velocity_abs_errors = [
            result.velocity_abs_error_m_s
            for result in results
            if result.velocity_abs_error_m_s is not None
        ]
        range_abs_errors = [
            result.range_abs_error_m
            for result in results
            if result.range_abs_error_m is not None
        ]
        joint_errors = [
            result.joint_error_cells
            for result in results
            if result.joint_error_cells is not None
        ]

        trials = len(results)
        valid_velocity_trials = len(decoded_velocities)
        valid_joint_trials = len(joint_errors)
        rmse_velocity = (
            float(np.sqrt(np.mean(np.asarray(velocity_errors, dtype=float) ** 2)))
            if velocity_errors
            else float("nan")
        )

        return SweepSummaryRow(
            sweep_name=sweep_name,
            x_value=float(x_value),
            trials=trials,
            valid_velocity_trials=valid_velocity_trials,
            valid_joint_trials=valid_joint_trials,
            mean_decoded_velocity_m_s=self._nan_mean(decoded_velocities),
            std_decoded_velocity_m_s=self._nan_std(decoded_velocities),
            mean_velocity_error_m_s=self._nan_mean(velocity_errors),
            mean_abs_velocity_error_m_s=self._nan_mean(velocity_abs_errors),
            std_abs_velocity_error_m_s=self._nan_std(velocity_abs_errors),
            rmse_velocity_m_s=rmse_velocity,
            mean_range_abs_error_m=self._nan_mean(range_abs_errors),
            mean_joint_error_cells=self._nan_mean(joint_errors),
            velocity_miss_rate=(
                1.0 - valid_velocity_trials / float(trials) if trials > 0 else float("nan")
            ),
            joint_miss_rate=(
                1.0 - valid_joint_trials / float(trials) if trials > 0 else float("nan")
            ),
            y_value=y_value,
        )

    def evaluate_velocity_sweep(
        self,
        velocity_values: Optional[Sequence[float]] = None,
        distance_m: float = 1.15,
        snr_db: Optional[float] = None,
        trials_per_velocity: int = 32,
        reflectivity_range: Optional[Tuple[float, float]] = (0.85, 0.95),
        roughness_range: Optional[Tuple[float, float]] = (0.1, 0.2),
        progress: bool = True,
        plot: bool = True,
        save_path: Optional[Path] = None,
        show_plot: bool = True,
    ) -> Tuple[List[VelocityTrialResult], List[SweepSummaryRow]]:
        if velocity_values is None:
            velocity_values = np.linspace(-8.0, 8.0, 17)

        velocity_values_np = self._as_float_array(velocity_values)
        results: List[VelocityTrialResult] = []
        summary_rows: List[SweepSummaryRow] = []
        total = int(len(velocity_values_np) * int(trials_per_velocity))
        completed = 0
        if progress:
            self._print_progress(completed, total)

        for velocity in velocity_values_np:
            group: List[VelocityTrialResult] = []
            for trial_idx in range(1, int(trials_per_velocity) + 1):
                result = self.run_trial(
                    trial_index=trial_idx,
                    distance_m=distance_m,
                    velocity_m_s=float(velocity),
                    snr_db=snr_db,
                    reflectivity_range=reflectivity_range,
                    roughness_range=roughness_range,
                )
                results.append(result)
                group.append(result)
                completed += 1
                if progress:
                    self._print_progress(completed, total)

            summary_rows.append(
                self.summarize_results(
                    sweep_name="true_velocity_m_s",
                    x_value=float(velocity),
                    results=group,
                )
            )

        if plot:
            self.plot_velocity_vs_true_velocity(
                summary_rows=summary_rows,
                save_path=save_path,
                show=show_plot,
            )

        return results, summary_rows

    def evaluate_snr_sweep(
        self,
        snr_values: Optional[Sequence[float]] = None,
        distance_m: float = 1.15,
        velocity_m_s: float = 3.0,
        trials_per_snr: int = 32,
        reflectivity_range: Optional[Tuple[float, float]] = (0.85, 0.95),
        roughness_range: Optional[Tuple[float, float]] = (0.1, 0.2),
        progress: bool = True,
        plot: bool = True,
        save_path: Optional[Path] = None,
        show_plot: bool = True,
    ) -> Tuple[List[VelocityTrialResult], List[SweepSummaryRow]]:
        if snr_values is None:
            snr_values = [-20, -15, -10, -5, 0, 5, 10, 15, 20]

        snr_values_np = self._as_float_array(snr_values)
        results: List[VelocityTrialResult] = []
        summary_rows: List[SweepSummaryRow] = []
        total = int(len(snr_values_np) * int(trials_per_snr))
        completed = 0
        if progress:
            self._print_progress(completed, total)

        for snr_db in snr_values_np:
            group: List[VelocityTrialResult] = []
            for trial_idx in range(1, int(trials_per_snr) + 1):
                result = self.run_trial(
                    trial_index=trial_idx,
                    distance_m=distance_m,
                    velocity_m_s=velocity_m_s,
                    snr_db=float(snr_db),
                    reflectivity_range=reflectivity_range,
                    roughness_range=roughness_range,
                )
                results.append(result)
                group.append(result)
                completed += 1
                if progress:
                    self._print_progress(completed, total)

            summary_rows.append(
                self.summarize_results(
                    sweep_name="snr_db",
                    x_value=float(snr_db),
                    results=group,
                )
            )

        if plot:
            self.plot_velocity_accuracy_vs_snr(
                summary_rows=summary_rows,
                save_path=save_path,
                show=show_plot,
            )

        return results, summary_rows

    def evaluate_joint_grid(
        self,
        range_values_m: Optional[Sequence[float]] = None,
        velocity_values_m_s: Optional[Sequence[float]] = None,
        snr_db: Optional[float] = None,
        trials_per_cell: int = 12,
        reflectivity_range: Optional[Tuple[float, float]] = (0.85, 0.95),
        roughness_range: Optional[Tuple[float, float]] = (0.1, 0.2),
        progress: bool = True,
        plot: bool = True,
        save_path: Optional[Path] = None,
        show_plot: bool = True,
    ) -> Tuple[List[VelocityTrialResult], np.ndarray, np.ndarray, List[SweepSummaryRow]]:
        if range_values_m is None:
            range_values_m = np.linspace(0.35, 2.75, 9)
        if velocity_values_m_s is None:
            velocity_values_m_s = np.linspace(-8.0, 8.0, 9)

        range_values_np = self._as_float_array(range_values_m)
        velocity_values_np = self._as_float_array(velocity_values_m_s)
        joint_error_grid = np.full((len(velocity_values_np), len(range_values_np)), np.nan)
        miss_rate_grid = np.full_like(joint_error_grid, np.nan, dtype=float)
        results: List[VelocityTrialResult] = []
        summary_rows: List[SweepSummaryRow] = []

        total = int(len(range_values_np) * len(velocity_values_np) * int(trials_per_cell))
        completed = 0
        if progress:
            self._print_progress(completed, total)

        for vel_idx, velocity in enumerate(velocity_values_np):
            for range_idx, distance in enumerate(range_values_np):
                group: List[VelocityTrialResult] = []
                for trial_idx in range(1, int(trials_per_cell) + 1):
                    result = self.run_trial(
                        trial_index=trial_idx,
                        distance_m=float(distance),
                        velocity_m_s=float(velocity),
                        snr_db=snr_db,
                        reflectivity_range=reflectivity_range,
                        roughness_range=roughness_range,
                    )
                    results.append(result)
                    group.append(result)
                    completed += 1
                    if progress:
                        self._print_progress(completed, total)

                summary = self.summarize_results(
                    sweep_name="joint_range_velocity",
                    x_value=float(distance),
                    results=group,
                    y_value=float(velocity),
                )
                summary_rows.append(summary)
                joint_error_grid[vel_idx, range_idx] = summary.mean_joint_error_cells
                miss_rate_grid[vel_idx, range_idx] = summary.joint_miss_rate

        if plot:
            self.plot_joint_error_heatmap(
                range_values_m=range_values_np,
                velocity_values_m_s=velocity_values_np,
                joint_error_grid=joint_error_grid,
                save_path=save_path,
                show=show_plot,
            )

        return results, joint_error_grid, miss_rate_grid, summary_rows

    def plot_velocity_vs_true_velocity(
        self,
        summary_rows: Sequence[SweepSummaryRow],
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        rows = sorted(summary_rows, key=lambda row: row.x_value)
        true_velocity = np.asarray([row.x_value for row in rows], dtype=float)
        mean_decoded = np.asarray([row.mean_decoded_velocity_m_s for row in rows], dtype=float)
        yerr = np.asarray([row.std_decoded_velocity_m_s for row in rows], dtype=float)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.errorbar(
            true_velocity,
            mean_decoded,
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=4,
            color=PLOT_BLUE,
            ecolor="0.35",
            label="Decoded velocity (mean +/- SD)",
        )
        finite_axis_values = np.concatenate(
            (
                true_velocity[np.isfinite(true_velocity)],
                mean_decoded[np.isfinite(mean_decoded)],
            )
        )
        if finite_axis_values.size == 0:
            finite_axis_values = np.asarray([-1.0, 1.0], dtype=float)
        min_v = float(np.min(finite_axis_values))
        max_v = float(np.max(finite_axis_values))
        pad = max(0.5, 0.06 * (max_v - min_v))
        ax.plot([min_v - pad, max_v + pad], [min_v - pad, max_v + pad], "k--", lw=1.2, label="Ideal")
        ax.set_title("Velocity Accuracy vs True Velocity")
        ax.set_xlabel("True velocity (m/s)")
        ax.set_ylabel("Decoded velocity (m/s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        self._save_or_show(fig, save_path, show)

    def plot_velocity_accuracy_vs_snr(
        self,
        summary_rows: Sequence[SweepSummaryRow],
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        rows = sorted(summary_rows, key=lambda row: row.x_value)
        snr_db = np.asarray([row.x_value for row in rows], dtype=float)
        mae = np.asarray([row.mean_abs_velocity_error_m_s for row in rows], dtype=float)
        yerr = np.asarray([row.std_abs_velocity_error_m_s for row in rows], dtype=float)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.errorbar(
            snr_db,
            mae,
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=4,
            color=PLOT_BLUE,
            ecolor="0.35",
            label="Velocity MAE (mean +/- SD)",
        )
        ax.set_title("Velocity Accuracy vs SNR")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Mean absolute velocity error (m/s)")
        ax.set_xticks(snr_db)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        self._save_or_show(fig, save_path, show)

    def plot_joint_error_heatmap(
        self,
        range_values_m: Sequence[float],
        velocity_values_m_s: Sequence[float],
        joint_error_grid: np.ndarray,
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        ranges = self._as_float_array(range_values_m)
        velocities = self._as_float_array(velocity_values_m_s)
        range_edges = self._axis_edges(ranges)
        velocity_edges = self._axis_edges(velocities)

        fig, ax = plt.subplots(figsize=(9, 6))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="lightgrey")
        mesh = ax.pcolormesh(
            range_edges,
            velocity_edges,
            np.ma.masked_invalid(joint_error_grid),
            shading="auto",
            cmap=cmap,
        )
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Mean joint error (range/velocity resolution cells)")
        ax.set_title("Joint Range and Velocity Accuracy")
        ax.set_xlabel("True range (m)")
        ax.set_ylabel("True velocity (m/s)")
        ax.set_xticks(ranges)
        ax.set_yticks(velocities)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        self._save_or_show(fig, save_path, show)

    @staticmethod
    def write_trial_results_csv(
        results: Sequence[VelocityTrialResult],
        output_path: Path,
    ) -> None:
        if not results:
            return
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))

    @staticmethod
    def write_summary_csv(
        rows: Sequence[SweepSummaryRow],
        output_path: Path,
    ) -> None:
        if not rows:
            return
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))


def _print_summary_table(title: str, rows: Sequence[SweepSummaryRow]) -> None:
    print("\n" + "=" * 118)
    print(title)
    print("=" * 118)
    print(
        "x_value | trials | valid | mean_dec_v | vel_mae | vel_rmse | range_mae | joint_err | miss"
    )
    for row in rows:
        print(
            f"{row.x_value:7.2f} | {row.trials:6d} | {row.valid_velocity_trials:5d} | "
            f"{row.mean_decoded_velocity_m_s:10.3f} | {row.mean_abs_velocity_error_m_s:7.3f} | "
            f"{row.rmse_velocity_m_s:8.3f} | {row.mean_range_abs_error_m:9.4f} | "
            f"{row.mean_joint_error_cells:9.3f} | {row.velocity_miss_rate:5.2f}"
        )


def main() -> None:
    eval_cfg = {
        "seed": 42,
        "distance_m": 1.15,
        "velocity_m_s": 3.0,
        "velocity_values": np.linspace(-8.0, 8.0, 17),
        "snr_values": [-20, -15, -10, -5, 0, 5, 10, 15, 20],
        "joint_range_values_m": np.linspace(0.35, 2.75, 9),
        "joint_velocity_values_m_s": np.linspace(-8.0, 8.0, 9),
        "trials_per_velocity": 1,
        "trials_per_snr": 100,
        "trials_per_joint_cell": 8,
        "save_dir": Path(__file__).resolve().parent / "evaluation_outputs",
        "show_plots": True,
    }

    framework = VelocityEvaluationFramework(seed=eval_cfg["seed"])
    save_dir = Path(eval_cfg["save_dir"])

    velocity_results, velocity_rows = framework.evaluate_velocity_sweep(
        velocity_values=eval_cfg["velocity_values"],
        distance_m=eval_cfg["distance_m"],
        trials_per_velocity=eval_cfg["trials_per_velocity"],
        save_path=save_dir / "velocity_vs_true_velocity.png",
        show_plot=eval_cfg["show_plots"],
    )
    _print_summary_table("VELOCITY SWEEP", velocity_rows)
    framework.write_summary_csv(velocity_rows, save_dir / "velocity_sweep_summary.csv")
    framework.write_trial_results_csv(velocity_results, save_dir / "velocity_sweep_trials.csv")

    snr_results, snr_rows = framework.evaluate_snr_sweep(
        snr_values=eval_cfg["snr_values"],
        distance_m=eval_cfg["distance_m"],
        velocity_m_s=eval_cfg["velocity_m_s"],
        trials_per_snr=eval_cfg["trials_per_snr"],
        save_path=save_dir / "velocity_vs_snr.png",
        show_plot=eval_cfg["show_plots"],
    )
    _print_summary_table("SNR SWEEP", snr_rows)
    framework.write_summary_csv(snr_rows, save_dir / "snr_sweep_summary.csv")
    framework.write_trial_results_csv(snr_results, save_dir / "snr_sweep_trials.csv")

    joint_results, joint_grid, miss_grid, joint_rows = framework.evaluate_joint_grid(
        range_values_m=eval_cfg["joint_range_values_m"],
        velocity_values_m_s=eval_cfg["joint_velocity_values_m_s"],
        trials_per_cell=eval_cfg["trials_per_joint_cell"],
        save_path=save_dir / "joint_range_velocity_error.png",
        show_plot=eval_cfg["show_plots"],
    )
    framework.write_trial_results_csv(joint_results, save_dir / "joint_grid_trials.csv")
    framework.write_summary_csv(joint_rows, save_dir / "joint_grid_summary.csv")
    np.save(save_dir / "joint_error_grid.npy", joint_grid)
    np.save(save_dir / "joint_miss_rate_grid.npy", miss_grid)

    print(f"\nSaved figures and CSVs to: {save_dir}")
    print(
        "Joint error is Euclidean error after normalizing range by one delay step "
        f"({framework.range_resolution_m:.4f} m) and velocity by one Doppler channel "
        f"({framework.velocity_resolution_m_s:.4f} m/s)."
    )


if __name__ == "__main__":
    main()
