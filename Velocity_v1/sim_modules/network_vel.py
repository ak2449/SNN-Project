import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List


# =====================================================================
# Data-classes for decoded results
# =====================================================================

@dataclass
class DecodedObject:
    """Population-decoded object hypothesis from the delay bank."""

    delay_steps: float
    peak_delay_steps: int
    start_delay_steps: int
    end_delay_steps: int
    total_spikes: float
    peak_spikes: float


@dataclass
class DopplerDecodedObject:
    """Population-decoded velocity hypothesis from the Doppler bank."""

    shift_channels: float       # Centre-of-mass decoded shift (fractional channels)
    peak_shift_channels: int    # Channel index of the peak neuron
    start_shift_channels: int
    end_shift_channels: int
    total_spikes: float
    peak_spikes: float
    velocity_m_s: float         # Converted velocity (filled in by caller)


# =====================================================================
# Shared population helpers
# =====================================================================

def _build_lateral_kernel(n_neurons, radius):
    n_neurons = int(n_neurons)
    radius = max(0, min(int(radius), max(0, n_neurons - 1)))
    if n_neurons <= 0 or radius <= 0:
        return torch.zeros((n_neurons, n_neurons), dtype=torch.float32)

    idx = torch.arange(n_neurons, dtype=torch.float32)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    return torch.where(
        (dist > 0) & (dist <= float(radius)),
        1.0 / dist.clamp_min(1.0),
        torch.zeros_like(dist),
    )


def _find_peak_indices(spike_counts, min_peak_spikes):
    counts = np.asarray(spike_counts, dtype=float)
    peaks = []

    for idx, center in enumerate(counts):
        if center < min_peak_spikes:
            continue

        left = counts[idx - 1] if idx > 0 else -np.inf
        right = counts[idx + 1] if idx < counts.size - 1 else -np.inf
        is_local_max = center >= left and center >= right and (center > left or center > right)
        if is_local_max:
            peaks.append(idx)

    return peaks


def _decode_local_peak_centroids(
    spike_counts,
    value_axis,
    min_peak_spikes,
    min_object_spikes,
    radius_channels,
):
    """
    Decode each local maximum using a fixed-radius centre of mass around the peak.
    """
    counts = np.asarray(spike_counts, dtype=float)
    values = np.asarray(value_axis, dtype=float)
    if counts.ndim != 1 or values.ndim != 1 or counts.size != values.size:
        raise ValueError("Spike counts and value axis must be matching 1-D vectors.")

    min_peak_spikes = float(min_peak_spikes)
    min_object_spikes = float(min_peak_spikes if min_object_spikes is None else min_object_spikes)
    radius = max(0, int(radius_channels))
    peak_indices = _find_peak_indices(counts, min_peak_spikes=min_peak_spikes)
    decoded = []

    for peak_idx in peak_indices:
        start_idx = max(0, peak_idx - radius)
        end_idx = min(counts.size - 1, peak_idx + radius)
        segment_counts = counts[start_idx: end_idx + 1]
        total_spikes = float(segment_counts.sum())
        if total_spikes < min_object_spikes:
            continue

        segment_values = values[start_idx: end_idx + 1]
        centroid = float(np.sum(segment_values * segment_counts) / max(total_spikes, 1e-8))
        decoded.append(
            (
                centroid,
                peak_idx,
                start_idx,
                end_idx,
                total_spikes,
                float(counts[peak_idx]),
            )
        )

    return decoded


class FMSweepBankNetwork(nn.Module):
    """Bank of delay-tuned coincidence detectors."""

    def __init__(
        self,
        delays,
        freq_channels=30,
        weight=0.2,
        beta=0.6,
        threshold=0.6,
        refractory_steps=0,
        lateral_inhibition_strength=0.0,
        lateral_inhibition_radius=1,
        inhibition_decay=0.0,
        decode_radius_channels=3,
    ):
        super().__init__()
        self.delays = np.array(delays, dtype=int)
        self.freq_channels = freq_channels
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.refractory_steps = int(refractory_steps)
        self.lateral_inhibition_strength = float(lateral_inhibition_strength)
        self.lateral_inhibition_radius = int(lateral_inhibition_radius)
        self.inhibition_decay = float(inhibition_decay)
        self.decode_radius_channels = int(decode_radius_channels)
        self.register_buffer("delay_tensor", torch.as_tensor(self.delays, dtype=torch.long))
        self.register_buffer(
            "lateral_kernel",
            _build_lateral_kernel(
                n_neurons=len(self.delays),
                radius=self.lateral_inhibition_radius,
            ),
        )
        self.freq_w = nn.Parameter(
            torch.full((len(self.delays), self.freq_channels), float(weight), dtype=torch.float32)
        )
        self.lif = snn.Leaky(beta=self.beta, threshold=self.threshold)

    def decode_population_response(
        self,
        spike_counts,
        min_peak_spikes=1,
        min_object_spikes=None,
    ) -> List[DecodedObject]:
        """
        Decode the full delay-bank population response into object hypotheses.

        Each local maximum is treated as an object peak. A fixed-radius local
        centre of mass around that peak gives the decoded delay estimate.
        """
        counts = np.asarray(spike_counts, dtype=float)
        if counts.ndim != 1:
            raise ValueError("Population decoding expects a 1-D spike count vector.")
        if counts.size != len(self.delays):
            raise ValueError("Spike count vector must match the number of delay neurons.")

        decoded_objects: List[DecodedObject] = []
        for decoded_delay, peak_idx, start_idx, end_idx, total_spikes, peak_spikes in _decode_local_peak_centroids(
            spike_counts=counts,
            value_axis=self.delays,
            min_peak_spikes=min_peak_spikes,
            min_object_spikes=min_object_spikes,
            radius_channels=self.decode_radius_channels,
        ):
            decoded_objects.append(
                DecodedObject(
                    delay_steps=decoded_delay,
                    peak_delay_steps=int(self.delays[peak_idx]),
                    start_delay_steps=int(self.delays[start_idx]),
                    end_delay_steps=int(self.delays[end_idx]),
                    total_spikes=total_spikes,
                    peak_spikes=peak_spikes,
                )
            )

        return decoded_objects

    def _lateral_neighbor_drive(self, spikes):
        """Spread a neuron's spike to neighboring delay neurons."""
        lateral_kernel = self.lateral_kernel
        if lateral_kernel.device != spikes.device or lateral_kernel.dtype != spikes.dtype:
            lateral_kernel = lateral_kernel.to(device=spikes.device, dtype=spikes.dtype)
        return spikes @ lateral_kernel

    @torch.no_grad()
    def forward(self, pulse_spikes, echo_spikes):
        t_steps, batch_size, _ = pulse_spikes.shape
        device = pulse_spikes.device
        n_neurons = len(self.delays)
        use_lateral = self.lateral_inhibition_strength > 0 and self.lateral_inhibition_radius > 0
        dtype = pulse_spikes.dtype

        if n_neurons == 0:
            return torch.zeros(t_steps, batch_size, 0, device=device, dtype=dtype)

        delay_tensor = self.delay_tensor.to(device=device)
        max_delay = int(delay_tensor.max().item()) if n_neurons > 0 else 0
        if max_delay > 0:
            pulse_pad = torch.zeros(max_delay, batch_size, self.freq_channels, device=device, dtype=dtype)
            pulse_history = torch.cat((pulse_pad, pulse_spikes), dim=0)
        else:
            pulse_history = pulse_spikes

        mem = torch.zeros(batch_size, n_neurons, device=device, dtype=dtype)
        refractory_counter = torch.zeros(batch_size, n_neurons, device=device, dtype=dtype)
        bank_spk_rec = torch.empty(t_steps, batch_size, n_neurons, device=device, dtype=dtype)
        inh_trace = torch.zeros(batch_size, n_neurons, device=device, dtype=dtype)
        freq_w = self.freq_w.to(device=device, dtype=dtype).unsqueeze(0)

        for t in range(t_steps):
            delayed_indices = t + max_delay - delay_tensor
            delayed_pulse = pulse_history[delayed_indices].permute(1, 0, 2)
            echo_t = echo_spikes[t].unsqueeze(1)
            coinc = delayed_pulse * echo_t
            cur = (coinc * freq_w).sum(dim=2)

            if self.refractory_steps > 0:
                active_mask = refractory_counter <= 0
                spk_active, mem_active = self.lif(cur, mem)
                current_spikes = torch.where(active_mask, spk_active, torch.zeros_like(spk_active))
                mem = torch.where(active_mask, mem_active, self.beta * mem)
                refractory_counter = torch.where(
                    current_spikes > 0,
                    torch.full_like(refractory_counter, float(self.refractory_steps)),
                    torch.clamp(refractory_counter - 1, min=0),
                )
            else:
                current_spikes, mem = self.lif(cur, mem)

            if use_lateral:
                neighbor_drive = self._lateral_neighbor_drive(current_spikes)
                inh_trace = self.inhibition_decay * inh_trace + neighbor_drive
                mem = mem - self.lateral_inhibition_strength * inh_trace

            bank_spk_rec[t] = current_spikes

        return bank_spk_rec

    @torch.no_grad()
    def detect_objects(self, pulse_spikes, echo_spikes, min_spikes=1):
        bank_spk = self.forward(pulse_spikes, echo_spikes)
        spike_counts = bank_spk.sum(dim=0).squeeze()

        active_indices = torch.nonzero(spike_counts >= min_spikes).flatten()
        detected_delays = self.delays[active_indices.cpu().numpy()]
        detected_counts = spike_counts[active_indices].cpu().numpy()

        return detected_delays, bank_spk, detected_counts

    @torch.no_grad()
    def decode_objects_population(
        self,
        pulse_spikes,
        echo_spikes,
        min_peak_spikes=1,
        min_object_spikes=None,
    ) -> List[DecodedObject]:
        """
        Run the bank and decode its full population response into object hypotheses.
        """
        bank_spk = self.forward(pulse_spikes, echo_spikes)
        spike_counts = bank_spk.sum(dim=0)
        if spike_counts.ndim != 2 or spike_counts.shape[0] != 1:
            raise ValueError("Population decoding currently expects batch size 1.")

        return self.decode_population_response(
            spike_counts[0].detach().cpu().numpy(),
            min_peak_spikes=min_peak_spikes,
            min_object_spikes=min_object_spikes,
        )


# =====================================================================
# Auditory-fovea Doppler bank for velocity estimation
# =====================================================================

class FovealDopplerBankNetwork(nn.Module):
    """Frequency-offset-tuned Doppler bank based on an auditory fovea map.

    This network does not make shifted copies of the echo spectrum.  Each
    neuron has a fixed Gaussian receptive field over the high-resolution
    CF/foveal frequency channels.  A positive ``freq_shift`` means an
    approaching target: the echo lands at a higher acoustic frequency, which is
    a lower channel index because the encoder's frequency axis runs high-to-low.

    The decoded range is used as an echo-time gate: Doppler neurons only see
    echo activity when the corresponding delayed pulse activity is present.
    This keeps the velocity stage closer to a tonotopic/foveal place-code
    interpretation instead of an engineered spectral-shift matched filter.
    """

    @staticmethod
    def _build_foveal_tuning(
        freq_shifts,
        freq_channels,
        reference_channel,
        tuning_width_channels,
    ):
        freq_shifts = torch.as_tensor(freq_shifts, dtype=torch.float32)
        channel_idx = torch.arange(int(freq_channels), dtype=torch.float32).unsqueeze(0)
        centers = float(reference_channel) - freq_shifts.unsqueeze(1)
        sigma = max(float(tuning_width_channels), 1e-6)

        # tuning = torch.exp(-0.5 * ((channel_idx - centers) / sigma) ** 2)
        # tuning = tuning / tuning.max(dim=1, keepdim=True).values.clamp_min(1e-8)
        tuning = torch.exp(-0.5 * ((channel_idx - centers) / sigma) ** 2)
        tuning = tuning / tuning.max(dim=1, keepdim=True).values.clamp_min(1e-8)
        # Suppress neurons whose preferred centre falls outside the encoder.
        valid = ((centers >= 0) & (centers <= freq_channels - 1)).float()
        tuning = tuning * valid
        return tuning

    def __init__(
        self,
        freq_shifts,
        delay,
        freq_channels=60,
        weight=0.2,
        beta=0.6,
        threshold=0.6,
        refractory_steps=0,
        lateral_inhibition_strength=0.0,
        lateral_inhibition_radius=1,
        inhibition_decay=0.0,
        reference_channel=None,
        tuning_width_channels=2.0,
        gate_threshold=1.0,
        decode_radius_channels=3,
    ):
        super().__init__()
        self.freq_shifts = np.array(freq_shifts, dtype=int)
        self.delay = int(delay)
        self.freq_channels = int(freq_channels)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.refractory_steps = int(refractory_steps)
        self.lateral_inhibition_strength = float(lateral_inhibition_strength)
        self.lateral_inhibition_radius = int(lateral_inhibition_radius)
        self.inhibition_decay = float(inhibition_decay)
        self.reference_channel = (
            0.5 * (self.freq_channels - 1)
            if reference_channel is None
            else float(reference_channel)
        )
        self.tuning_width_channels = float(tuning_width_channels)
        self.gate_threshold = float(gate_threshold)
        self.decode_radius_channels = int(decode_radius_channels)

        n_neurons = len(self.freq_shifts)
        self.register_buffer(
            "shift_tensor",
            torch.as_tensor(self.freq_shifts, dtype=torch.long),
        )
        self.register_buffer(
            "lateral_kernel",
            _build_lateral_kernel(n_neurons, self.lateral_inhibition_radius),
        )
        self.register_buffer(
            "foveal_tuning",
            self._build_foveal_tuning(
                freq_shifts=self.freq_shifts,
                freq_channels=self.freq_channels,
                reference_channel=self.reference_channel,
                tuning_width_channels=self.tuning_width_channels,
            ),
        )
        self.response_gain = nn.Parameter(
            torch.full((n_neurons,), float(weight), dtype=torch.float32)
        )
        self.lif = snn.Leaky(beta=self.beta, threshold=self.threshold)

    @torch.no_grad()
    def forward(self, pulse_spikes, echo_spikes):
        """Run the foveal Doppler bank over CF pulse/echo spike trains.

        ``pulse_spikes`` is used only to build a delayed echo-time gate.  The
        velocity evidence itself comes from where the echo lands on the foveal
        frequency map.
        """
        if pulse_spikes.ndim != 3 or echo_spikes.ndim != 3:
            raise ValueError("pulse_spikes and echo_spikes must have shape [time, batch, freq].")

        t_steps, batch_size, freq_ch = pulse_spikes.shape
        if echo_spikes.shape != pulse_spikes.shape:
            raise ValueError("pulse_spikes and echo_spikes must have the same shape.")
        if freq_ch != self.freq_channels:
            raise ValueError(
                f"FovealDopplerBankNetwork expected {self.freq_channels} channels "
                f"but received {freq_ch}."
            )

        device = pulse_spikes.device
        dtype = pulse_spikes.dtype
        n_neurons = len(self.freq_shifts)
        use_lateral = (
            self.lateral_inhibition_strength > 0 and self.lateral_inhibition_radius > 0
        )

        if n_neurons == 0:
            return torch.zeros(t_steps, batch_size, 0, device=device, dtype=dtype)

        if self.delay > 0:
            pad = torch.zeros(self.delay, batch_size, freq_ch, device=device, dtype=dtype)
            pulse_history = torch.cat((pad, pulse_spikes), dim=0)
        else:
            pulse_history = pulse_spikes

        mem = torch.zeros(batch_size, n_neurons, device=device, dtype=dtype)
        refractory_counter = torch.zeros(batch_size, n_neurons, device=device, dtype=dtype)
        inh_trace = torch.zeros(batch_size, n_neurons, device=device, dtype=dtype)
        bank_spk_rec = torch.empty(t_steps, batch_size, n_neurons, device=device, dtype=dtype)

        tuning = self.foveal_tuning.to(device=device, dtype=dtype)
        response_gain = self.response_gain.to(device=device, dtype=dtype).unsqueeze(0)

        for t in range(t_steps):
            delayed_pulse = pulse_history[t]
            echo_gate = (delayed_pulse.sum(dim=1, keepdim=True) >= self.gate_threshold).to(dtype)

            echo_t = echo_spikes[t]
            foveal_drive = echo_t @ tuning.T
            cur = echo_gate * foveal_drive * response_gain

            if self.refractory_steps > 0:
                active_mask = refractory_counter <= 0
                spk_active, mem_active = self.lif(cur, mem)
                current_spikes = torch.where(active_mask, spk_active, torch.zeros_like(spk_active))
                mem = torch.where(active_mask, mem_active, self.beta * mem)
                refractory_counter = torch.where(
                    current_spikes > 0,
                    torch.full_like(refractory_counter, float(self.refractory_steps)),
                    torch.clamp(refractory_counter - 1, min=0),
                )
            else:
                current_spikes, mem = self.lif(cur, mem)

            if use_lateral:
                lateral_kernel = self.lateral_kernel.to(device=device, dtype=dtype)
                neighbor_drive = current_spikes @ lateral_kernel.T
                inh_trace = self.inhibition_decay * inh_trace + neighbor_drive
                mem = mem - self.lateral_inhibition_strength * inh_trace

            bank_spk_rec[t] = current_spikes

        return bank_spk_rec

    def decode_population_response(
        self,
        spike_counts,
        min_peak_spikes=1,
        min_object_spikes=None,
        doppler_encoder=None,
        speed_of_sound_m_s=343.0,
    ) -> List[DopplerDecodedObject]:
        """Decode a foveal place-code response using local peak centroids."""
        counts = np.asarray(spike_counts, dtype=float)
        if counts.ndim != 1 or counts.size != len(self.freq_shifts):
            raise ValueError("Spike count vector must match the number of Doppler neurons.")

        decoded: List[DopplerDecodedObject] = []
        for com_shift, peak_idx, start_idx, end_idx, total, peak_spikes in _decode_local_peak_centroids(
            spike_counts=counts,
            value_axis=self.freq_shifts,
            min_peak_spikes=min_peak_spikes,
            min_object_spikes=min_object_spikes,
            radius_channels=self.decode_radius_channels,
        ):
            vel = 0.0
            if doppler_encoder is not None:
                vel = doppler_encoder.shift_channels_to_velocity(
                    com_shift, speed_of_sound_m_s=speed_of_sound_m_s
                )

            decoded.append(
                DopplerDecodedObject(
                    shift_channels=com_shift,
                    peak_shift_channels=int(self.freq_shifts[peak_idx]),
                    start_shift_channels=int(self.freq_shifts[start_idx]),
                    end_shift_channels=int(self.freq_shifts[end_idx]),
                    total_spikes=total,
                    peak_spikes=peak_spikes,
                    velocity_m_s=vel,
                )
            )

        return decoded
