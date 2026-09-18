import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List


@dataclass
class DecodedObject:
    """Population-decoded object hypothesis from the delay bank."""

    delay_steps: float
    peak_delay_steps: int
    start_delay_steps: int
    end_delay_steps: int
    total_spikes: float
    peak_spikes: float


class FMSweepCoincidenceNeuronBio(nn.Module):
    """Delay-line FM-FM coincidence detector neuron."""

    def __init__(
        self,
        delay,
        freq_channels=30,
        weight=1.0,
        beta=0.6,
        threshold=1.0,
        refractory_steps=0,
        coincidence_power=1.0,
        eps=1e-6,
    ):
        super().__init__()
        self.delay = int(delay)
        self.freq_channels = int(freq_channels)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.weight = float(weight)
        self.refractory_steps = int(refractory_steps)
        self.coincidence_power = float(coincidence_power)
        self.eps = float(eps)

        self.freq_w = nn.Parameter(torch.ones(self.freq_channels) * self.weight)
        self.lif = snn.Leaky(beta=self.beta, threshold=self.threshold)

    def init_state(self, batch_size, device="cpu"):
        mem = self.lif.init_leaky()
        if isinstance(mem, torch.Tensor):
            mem = mem.to(device)

        buffer = torch.zeros(max(1, self.delay), batch_size, self.freq_channels, device=device)
        refractory_counter = torch.zeros(batch_size, 1, device=device)
        return mem, buffer, refractory_counter

    def step(self, x_t, state):
        mem, buffer, refractory_counter = state

        pulse = x_t[:, : self.freq_channels]
        echo = x_t[:, self.freq_channels :]

        if self.delay == 0:
            delayed_pulse = pulse
            new_buffer = buffer
        else:
            delayed_pulse = buffer[0].clone()
            new_buffer = torch.cat((buffer[1:], pulse.unsqueeze(0)), dim=0)

        coinc = delayed_pulse * echo
        cur = (coinc * self.freq_w.unsqueeze(0)).sum(dim=1, keepdim=True)

        if self.refractory_steps > 0:
            active_mask = refractory_counter <= 0
            spk_active, mem_active = self.lif(cur, mem)
            spk = torch.where(active_mask, spk_active, torch.zeros_like(spk_active))
            mem = torch.where(active_mask, mem_active, self.beta * mem)
            refractory_counter = torch.where(
                spk > 0,
                torch.full_like(refractory_counter, float(self.refractory_steps)),
                torch.clamp(refractory_counter - 1, min=0),
            )
        else:
            spk, mem = self.lif(cur, mem)

        return spk, (mem, new_buffer, refractory_counter)


class WTAReadoutLayer(nn.Module):
    """
    Soft winner-take-all readout over delay-bank spikes.

    This is a separate readout stage: it consumes a bank spike tensor shaped
    [time, batch, neurons] and applies distance-dependent lateral inhibition
    before producing a sparse readout spike train of the same shape.
    """

    def __init__(
        self,
        n_neurons,
        beta=0.7,
        threshold=0.5,
        inhibition_strength=3.0,
        inhibition_radius=8,
        inhibition_falloff="gaussian",
        sigma=2.0,
    ):
        super().__init__()
        self.n = int(n_neurons)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.inhibition_strength = float(inhibition_strength)
        self.inhibition_radius = int(inhibition_radius)
        self.inhibition_falloff = str(inhibition_falloff)
        self.sigma = float(sigma)
        self.lif = snn.Leaky(beta=self.beta, threshold=self.threshold)
        self.register_buffer(
            "inh_kernel",
            self._build_kernel(
                n=self.n,
                radius=self.inhibition_radius,
                strength=self.inhibition_strength,
                falloff=self.inhibition_falloff,
                sigma=self.sigma,
            ),
        )

    @staticmethod
    def _build_kernel(n, radius, strength, falloff, sigma):
        """Return W[i, j]: spike from neuron j inhibits neuron i."""
        n = int(n)
        radius = max(0, int(radius))
        strength = float(strength)
        sigma = max(float(sigma), 1e-6)

        idx = torch.arange(n, dtype=torch.float32)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()

        if falloff == "gaussian":
            kernel = strength * torch.exp(-0.5 * (dist / sigma) ** 2)
        elif falloff == "linear":
            if radius <= 0:
                kernel = torch.zeros_like(dist)
            else:
                kernel = strength * (1.0 - dist / float(radius))
                kernel = kernel.clamp(min=0.0)
        else:
            raise ValueError("inhibition_falloff must be 'gaussian' or 'linear'.")

        kernel.fill_diagonal_(0.0)
        if radius <= 0:
            return torch.zeros_like(kernel)
        return kernel * (dist <= float(radius))

    @torch.no_grad()
    def forward(self, bank_spikes):
        if bank_spikes.ndim != 3:
            raise ValueError("bank_spikes must have shape [time, batch, neurons].")

        t_steps, batch_size, n_neurons = bank_spikes.shape
        if n_neurons != self.n:
            raise ValueError(
                f"WTAReadoutLayer expected {self.n} neurons but received {n_neurons}."
            )

        device = bank_spikes.device
        dtype = bank_spikes.dtype
        mem = torch.zeros(batch_size, n_neurons, device=device, dtype=dtype)
        out = torch.zeros_like(bank_spikes)
        inh_kernel = self.inh_kernel.to(device=device, dtype=dtype)

        for t in range(t_steps):
            drive = bank_spikes[t]
            inh = drive @ inh_kernel.T
            spk, mem = self.lif(drive - inh, mem)
            out[t] = spk

        return out


class FMSweepBankNetwork(nn.Module):
    """Bank of delay-tuned coincidence detectors."""

    @staticmethod
    def _build_lateral_kernel(n_neurons, radius):
        n_neurons = int(n_neurons)
        radius = max(0, min(int(radius), max(0, n_neurons - 1)))
        if n_neurons <= 0 or radius <= 0:
            return torch.zeros((n_neurons, n_neurons), dtype=torch.float32)

        idx = torch.arange(n_neurons, dtype=torch.float32)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        kernel = torch.where(
            (dist > 0) & (dist <= float(radius)),
            1.0 / dist.clamp_min(1.0),
            torch.zeros_like(dist),
        )
        return kernel

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
        self.register_buffer("delay_tensor", torch.as_tensor(self.delays, dtype=torch.long))
        self.register_buffer(
            "lateral_kernel",
            self._build_lateral_kernel(
                n_neurons=len(self.delays),
                radius=self.lateral_inhibition_radius,
            ),
        )
        self.freq_w = nn.Parameter(
            torch.full((len(self.delays), self.freq_channels), float(weight), dtype=torch.float32)
        )
        self.lif = snn.Leaky(beta=self.beta, threshold=self.threshold)

    def _find_peak_indices(self, spike_counts, min_peak_spikes):
        """Identify local maxima in the delay-bank response."""
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

    def decode_population_response(
        self,
        spike_counts,
        min_peak_spikes=1,
        min_object_spikes=None,
    ) -> List[DecodedObject]:
        """
        Decode the full delay-bank population response into object hypotheses.

        Each local maximum is treated as an object peak. Activity between adjacent
        peaks is partitioned at the valley, then a center-of-mass over that segment
        gives the decoded delay estimate.
        """
        counts = np.asarray(spike_counts, dtype=float)
        if counts.ndim != 1:
            raise ValueError("Population decoding expects a 1-D spike count vector.")
        if counts.size != len(self.delays):
            raise ValueError("Spike count vector must match the number of delay neurons.")

        min_peak_spikes = float(min_peak_spikes)
        min_object_spikes = float(min_peak_spikes if min_object_spikes is None else min_object_spikes)
        peak_indices = self._find_peak_indices(counts, min_peak_spikes=min_peak_spikes)
        if not peak_indices:
            return []

        left_edges = [0]
        right_edges = []
        for left_peak, right_peak in zip(peak_indices[:-1], peak_indices[1:]):
            valley_window = counts[left_peak : right_peak + 1]
            valley_offset = int(np.argmin(valley_window))
            boundary = left_peak + valley_offset
            right_edges.append(boundary)
            left_edges.append(boundary + 1)
        right_edges.append(len(counts) - 1)

        decoded_objects: List[DecodedObject] = []
        for peak_idx, start_idx, end_idx in zip(peak_indices, left_edges, right_edges):
            segment_counts = counts[start_idx : end_idx + 1]
            total_spikes = float(segment_counts.sum())
            if total_spikes < min_object_spikes:
                continue

            segment_delays = self.delays[start_idx : end_idx + 1].astype(float)
            decoded_delay = float(np.sum(segment_delays * segment_counts) / max(total_spikes, 1e-8))
            decoded_objects.append(
                DecodedObject(
                    delay_steps=decoded_delay,
                    peak_delay_steps=int(self.delays[peak_idx]),
                    start_delay_steps=int(self.delays[start_idx]),
                    end_delay_steps=int(self.delays[end_idx]),
                    total_spikes=total_spikes,
                    peak_spikes=float(counts[peak_idx]),
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
