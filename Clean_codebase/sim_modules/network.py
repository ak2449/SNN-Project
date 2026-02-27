import numpy as np
import snntorch as snn
import torch
import torch.nn as nn


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
    ):
        super().__init__()
        self.delays = np.array(delays, dtype=int)
        self.freq_channels = freq_channels
        self.lateral_inhibition_strength = float(lateral_inhibition_strength)
        self.lateral_inhibition_radius = int(lateral_inhibition_radius)
        self.inhibition_decay = float(inhibition_decay)

        self.bank = nn.ModuleList(
            [
                FMSweepCoincidenceNeuronBio(
                    delay=d,
                    freq_channels=freq_channels,
                    weight=weight,
                    beta=beta,
                    threshold=threshold,
                    refractory_steps=refractory_steps,
                    coincidence_power=1.0,
                )
                for d in self.delays
            ]
        )

    def _lateral_neighbor_drive(self, spikes):
        """Spread a neuron's spike to neighboring delay neurons."""
        n_neurons = spikes.shape[1]
        drive = torch.zeros_like(spikes)
        radius = min(self.lateral_inhibition_radius, max(0, n_neurons - 1))
        for offset in range(1, radius + 1):
            weight = 1.0 / float(offset)
            drive[:, offset:] += weight * spikes[:, :-offset]
            drive[:, :-offset] += weight * spikes[:, offset:]
        return drive

    @torch.no_grad()
    def forward(self, pulse_spikes, echo_spikes):
        t_steps, batch_size, _ = pulse_spikes.shape
        device = pulse_spikes.device
        n_neurons = len(self.bank)
        use_lateral = self.lateral_inhibition_strength > 0 and self.lateral_inhibition_radius > 0

        bank_states = [neuron.init_state(batch_size, device=device) for neuron in self.bank]
        bank_spk_rec = []
        inh_trace = torch.zeros(batch_size, n_neurons, device=device)

        for t in range(t_steps):
            x_t = torch.cat([pulse_spikes[t], echo_spikes[t]], dim=1)
            spk_t_list = []
            for i, neuron in enumerate(self.bank):
                spk_i, bank_states[i] = neuron.step(x_t, bank_states[i])
                spk_t_list.append(spk_i)
            current_spikes = torch.cat(spk_t_list, dim=1)
            bank_spk_rec.append(current_spikes)

            if use_lateral:
                neighbor_drive = self._lateral_neighbor_drive(current_spikes)
                inh_trace = self.inhibition_decay * inh_trace + neighbor_drive
                for i in range(n_neurons):
                    mem, buffer, refractory_counter = bank_states[i]
                    mem = mem - self.lateral_inhibition_strength * inh_trace[:, i : i + 1]
                    bank_states[i] = (mem, buffer, refractory_counter)

        return torch.stack(bank_spk_rec, dim=0)

    @torch.no_grad()
    def detect_objects(self, pulse_spikes, echo_spikes, min_spikes=1):
        bank_spk = self.forward(pulse_spikes, echo_spikes)
        spike_counts = bank_spk.sum(dim=0).squeeze()

        active_indices = torch.nonzero(spike_counts >= min_spikes).flatten()
        detected_delays = self.delays[active_indices.cpu().numpy()]
        detected_counts = spike_counts[active_indices].cpu().numpy()

        return detected_delays, bank_spk, detected_counts
