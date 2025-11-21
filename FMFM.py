import torch
import torch.nn as nn
import snntorch as snn
import numpy as np

# -----------------------
# 1. Define an FM–FM-like spiking neuron
# -----------------------
class FMFMNeuron(nn.Module):
    def __init__(self, beta=0.95):
        super().__init__()
        # 2 input channels: FM1, FM3
        self.fc = nn.Linear(2, 1, bias=False)

        # initialise weights so FM1 is a "priming" current,
        # FM3 is a strong trigger
        with torch.no_grad():
            self.fc.weight[:] = torch.tensor([[0.7, 0.6]])  # [w_FM1, w_FM3]y

        # Leaky integrate-and-fire neuron
        self.lif = snn.Leaky(beta=beta)

    def forward(self, spike_seq):
        """
        spike_seq: [num_steps, batch_size, 2]
        Returns:
            spk_rec: [num_steps, batch_size, 1]
        """
        num_steps, batch_size, _ = spike_seq.shape
        mem = self.lif.init_leaky()

        spk_rec = []
        for t in range(num_steps):
            cur = self.fc(spike_seq[t])       # current from FM1 + FM3
            spk, mem = self.lif(cur, mem)     # LIF dynamics
            spk_rec.append(spk)

        spk_rec = torch.stack(spk_rec, dim=0)
        return spk_rec


def generate_pulse_echo(num_steps=50, delay=10, batch_size=1):
    """
    Create spike trains:
      - FM1: single spike at t=5
      - FM3: single spike at t=5+delay
    """
    spikes = torch.zeros(num_steps, batch_size, 2)

    t_pulse = 5
    t_echo = t_pulse + delay
    if t_echo < num_steps:
        spikes[t_pulse, :, 0] = 1.0  # FM1 channel
        spikes[t_echo, :, 1] = 1.0   # FM3 channel

    return spikes

# Example: test response for different delays
device = 'cpu'
neuron = FMFMNeuron(beta=0.9).to(device)

for delay in np.linspace(1, 20,20, dtype=int):
    spike_seq = generate_pulse_echo(delay=delay).to(device)
    spk_out = neuron(spike_seq)              # [T, 1, 1]
    total_spikes = spk_out.sum().item()
    print(f"Delay {delay} -> total spikes: {total_spikes}")

