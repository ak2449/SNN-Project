import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
import matplotlib.pyplot as plt


# -----------------------
# FM–FM-like spiking neuron using Synaptic model
# -----------------------
class FMFMSynapticNeuron(nn.Module):
    def __init__(self, alpha=0.9, beta=0.9,w_fm1=0.5, w_fm3=0.5, threshold=1.0):
        super().__init__()

        # 2 input channels: FM1 (pulse), FM3 (echo)
        self.fc = nn.Linear(2, 1, bias=False)

        with torch.no_grad():
            self.fc.weight[:] = torch.tensor([[w_fm1, w_fm3]])  # [w_FM1, w_FM3]

        # 2nd-order LIF neuron with synaptic current
        self.syn = snn.Synaptic(alpha=alpha, beta=beta, threshold=threshold)

    def forward(self, spike_seq, return_states=False):
        """
        spike_seq: [num_steps, batch_size, 2]   (FM1, FM3)
        Returns:
            spk_rec: [num_steps, batch_size, 1]
            (optionally) syn_rec, mem_rec
        """
        num_steps, batch_size, _ = spike_seq.shape

        # init synaptic current & membrane.
        # NOTE: in your snntorch version, init_synaptic() -> (syn, mem)
        syn, mem = self.syn.init_synaptic()

        spk_rec, syn_rec, mem_rec = [], [], []

        for t in range(num_steps):
            x_t = spike_seq[t]          # [batch, 2]
            cur_in = self.fc(x_t)       # [batch, 1]

            # Synaptic neuron update: returns (spk, syn, mem)
            spk, syn, mem = self.syn(cur_in, syn, mem)

            spk_rec.append(spk)
            syn_rec.append(syn)
            mem_rec.append(mem)

        spk_rec = torch.stack(spk_rec, dim=0)   # [T, B, 1]
        syn_rec = torch.stack(syn_rec, dim=0)   # [T, B, 1]
        mem_rec = torch.stack(mem_rec, dim=0)   # [T, B, 1]

        if return_states:
            return spk_rec, syn_rec, mem_rec
        return spk_rec


def plot_neuron_states(spk_out, syn_rec, mem_rec, delay=None, batch_idx=0):
    """
    Plot spike output, synaptic current, and membrane voltage over time
    for a single sample in the batch, in one figure with 3 subplots.

    spk_out, syn_rec, mem_rec: [T, B, 1] tensors from the neuron
    delay: (optional) integer delay, just for the plot title
    batch_idx: which batch element to visualize (default: 0)
    """
    # Move to CPU, detach from graph, squeeze to 1D: [T]
    fm1 = spike_seq[:, batch_idx, 0].detach().cpu().numpy()
    fm3 = spike_seq[:, batch_idx, 1].detach().cpu().numpy()
    spk = spk_out[:, batch_idx, 0].detach().cpu().numpy()
    syn = syn_rec[:, batch_idx, 0].detach().cpu().numpy()
    mem = mem_rec[:, batch_idx, 0].detach().cpu().numpy()

    T = len(spk)
    t = np.arange(T)

    title_suffix = f" (delay = {delay})" if delay is not None else ""

    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    fig.suptitle("FM–FM Synaptic Neuron Dynamics" + title_suffix)

    # --- Input spikes ---
    ax = axes[0]
    ax.stem(t, fm1, linefmt="C0-", markerfmt="C0o", basefmt=" ", label="FM1")
    ax.stem(t, fm3, linefmt="C1-", markerfmt="C1s", basefmt=" ", label="FM3")
    ax.set_ylabel("Input")
    ax.set_title("Input spikes (FM1 & FM3)")
    ax.legend()
    ax.grid(True)

    # --- Output Spikes ---
    ax = axes[3]
    ax.stem(t, spk)
    ax.set_ylabel("Spike")
    ax.set_title("Spike output")
    ax.grid(True)

    # --- Synaptic current ---
    ax = axes[1]
    ax.plot(t, syn)
    ax.set_ylabel("Syn current")
    ax.set_title("Synaptic current")
    ax.grid(True)

    # --- Membrane potential ---
    ax = axes[2]
    ax.plot(t, mem)
    ax.hlines(y=1.0, color='r', linestyle='--', label='Threshold',xmin=0, xmax=T-1)
    ax.set_xlabel("Time step")
    ax.set_ylabel("V_mem")
    ax.set_title("Membrane potential")
    ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

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
        spikes[t_echo,  :, 1] = 1.0  # FM3 channel

    return spikes


# -----------------------
# Test response for different delays
# -----------------------
device = "cpu"

neuron = FMFMSynapticNeuron(
    alpha=0.6,
    beta=0.4,
    w_fm1=0.3,
    w_fm3=0.55,
    threshold=1.0,
).to(device)

for delay in range(1, 21):
    spike_seq = generate_pulse_echo(delay=delay).to(device)
    spk_out, syn_rec, mem_rec = neuron(spike_seq, return_states=True)

    total_spikes = spk_out.sum().item()
    max_syn = syn_rec.max().item()
    max_mem = mem_rec.max().item()
    print(
        f"Delay {delay:2d} -> spikes: {total_spikes:.1f}, "
        f"max syn: {max_syn:.3f}, max V_mem: {max_mem:.3f}"
    )


delay = 5
spike_seq = generate_pulse_echo(delay=delay).to(device)
spk_out, syn_rec, mem_rec = neuron(spike_seq, return_states=True)

plot_neuron_states(spk_out, syn_rec, mem_rec, delay=delay, batch_idx=0)
