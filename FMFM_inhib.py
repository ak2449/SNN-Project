import torch
import torch.nn as nn
import snntorch as snn
import numpy as np

# -----------------------
# FM–FM-like spiking neuron with early inhibition
# -----------------------
class FMFMNeuronInhib(nn.Module):
    def __init__(self, beta_mem=0.9, beta_inh=0.6):
        super().__init__()

        # 2 input channels: FM1, FM3 (excitatory)
        self.fc_exc = nn.Linear(2, 1, bias=False)

        # excitatory weights: FM1 is "priming", FM3 is strong trigger
        with torch.no_grad():
            self.fc_exc.weight[:] = torch.tensor([[0.7,1]])  # [w_FM1, w_FM3]

        # inhibitory parameters
        self.beta_inh = beta_inh
        # negative weight from inhibitory state into the neuron
        self.w_inh = nn.Parameter(torch.tensor(-1.0))  # start around -1.0

        # Leaky integrate-and-fire neuron (you can lower threshold if needed)
        self.lif = snn.Leaky(beta=beta_mem)  # threshold defaults to 1.0

    def forward(self, spike_seq):
        """
        spike_seq: [num_steps, batch_size, 2]   (FM1, FM3)
        Returns:
            spk_rec: [num_steps, batch_size, 1]
        """
        num_steps, batch_size, _ = spike_seq.shape

        # init membrane and inhibitory state
        mem = self.lif.init_leaky()
        inh_state = torch.zeros(batch_size, 1, device=spike_seq.device)

        spk_rec = []
        mem_rec = []

        for t in range(num_steps):
            x_t = spike_seq[t]                 # [batch, 2]

            # excitatory current from FM1 / FM3
            cur_exc = self.fc_exc(x_t)         # [batch, 1]

            # update inhibitory "synapse" state driven by FM1 (channel 0)
            fm1_t = x_t[:, 0:1]               # [batch, 1]
            inh_state = self.beta_inh * inh_state + fm1_t

            # inhibitory current is negative
            cur_inh = self.w_inh * inh_state  # [batch, 1]

            # total current
            cur = cur_exc + cur_inh

            # LIF dynamics
            spk, mem = self.lif(cur, mem)
            spk_rec.append(spk)
            mem_rec.append(mem)

        spk_rec = torch.stack(spk_rec, dim=0)
        mem_rec = torch.stack(mem_rec, dim=0)

        return spk_rec,mem_rec


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


# -----------------------
# Test response for different delays
# -----------------------
device = "cpu"
neuron = FMFMNeuronInhib(beta_mem=0.7, beta_inh=0.4).to(device)

for delay in range(1, 21):
    spike_seq = generate_pulse_echo(delay=delay).to(device)
    spk_out,mem_rec = neuron(spike_seq)              # [T, 1, 1]
    total_spikes = spk_out.sum().item()
    max_mem = mem_rec.max().item()
    print(f"Delay {delay:2d} -> spikes: {total_spikes:.1f}, max V_mem: {max_mem:.3f}")

