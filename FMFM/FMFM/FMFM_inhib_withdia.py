import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
import matplotlib.pyplot as plt

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
            self.fc_exc.weight[:] = torch.tensor([[0, 1]])  # [w_FM1, w_FM3]

        # inhibitory parameters
        self.beta_inh = beta_inh
        # negative weight from inhibitory state into the neuron
        self.w_inh = nn.Parameter(torch.tensor(0.0))  # start around -1.0

        # Leaky integrate-and-fire neuron (threshold defaults to 1.0)
        self.lif = snn.Leaky(beta=beta_mem)  # threshold defaults to 1.0

    def forward(self, spike_seq):
        """
        spike_seq: [num_steps, batch_size, 2]   (FM1, FM3)
        Returns:
            spk_rec : [T, B, 1]
            exc_rec : [T, B, 1]  (excitatory current)
            inh_rec : [T, B, 1]  (inhibitory current)
            mem_rec : [T, B, 1]
        """
        num_steps, batch_size, _ = spike_seq.shape

        # init membrane and inhibitory state
        mem = self.lif.init_leaky()  # if your version needs batch_size, adjust here
        inh_state = torch.zeros(batch_size, 1, device=spike_seq.device)

        spk_rec = []
        mem_rec = []
        exc_rec = []
        inh_rec = []

        for t in range(num_steps):
            x_t = spike_seq[t]                 # [B, 2]

            # excitatory current from FM1 / FM3
            cur_exc = self.fc_exc(x_t)         # [B, 1]

            # update inhibitory "synapse" state driven by FM1 (channel 0)
            fm1_t = x_t[:, 0:1]                # [B, 1]
            inh_state = self.beta_inh * inh_state + fm1_t

            # inhibitory current is negative
            cur_inh = self.w_inh * inh_state   # [B, 1]

            # total current
            cur = cur_exc + cur_inh            # [B, 1]

            # LIF dynamics
            spk, mem = self.lif(cur, mem)

            # record all
            spk_rec.append(spk)
            mem_rec.append(mem)
            exc_rec.append(cur_exc)
            inh_rec.append(cur_inh)
        
        spk_rec = torch.stack(spk_rec, dim=0)  # [T, B, 1]
        mem_rec = torch.stack(mem_rec, dim=0)  # [T, B, 1]
        exc_rec = torch.stack(exc_rec, dim=0)  # [T, B, 1]
        inh_rec = torch.stack(inh_rec, dim=0)  # [T, B, 1]

        return spk_rec, exc_rec, inh_rec, mem_rec


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


def plot_neuron_states(spike_seq, spk_out, exc_rec, inh_rec, mem_rec,
                       delay=None, batch_idx=0):
    """
    Plot, for a single batch element:
      - Input spikes (FM1 & FM3)
      - Excitatory, inhibitory, and total currents
      - Membrane potential (with threshold)
      - Output spikes

    spike_seq: [T, B, 2]
    spk_out, exc_rec, inh_rec, mem_rec: [T, B, 1]
    """

    # Move to CPU, detach, squeeze to 1D over time
    fm1 = spike_seq[:, batch_idx, 0].detach().cpu().numpy()
    fm3 = spike_seq[:, batch_idx, 1].detach().cpu().numpy()
    spk = spk_out[:, batch_idx, 0].detach().cpu().numpy()
    exc = exc_rec[:, batch_idx, 0].detach().cpu().numpy()
    inh = inh_rec[:, batch_idx, 0].detach().cpu().numpy()
    mem = mem_rec[:, batch_idx, 0].detach().cpu().numpy()

    # total current reaching the neuron
    tot = exc + inh

    T = len(spk)
    t = np.arange(T)

    title_suffix = f" (delay = {delay})" if delay is not None else ""

    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    fig.suptitle("FM–FM Neuron with Early Inhibition" + title_suffix)

    # --- 1) Input spikes (FM1 & FM3) ---
    ax = axes[0]
    ax.stem(t, fm1, linefmt="C0-", markerfmt="C0o", basefmt=" ", label="FM1")
    ax.stem(t, fm3, linefmt="C1-", markerfmt="C1s", basefmt=" ", label="FM3")
    ax.set_ylabel("Input")
    ax.set_title("Input spikes (FM1 & FM3)")
    ax.legend()
    ax.grid(True)

    # --- 2) Excitatory, inhibitory, and total currents ---
    ax = axes[1]
    ax.plot(t, exc, label="Excitatory")
    ax.plot(t, inh, label="Inhibitory")
    ax.plot(t, tot, linestyle="--", label="Total (exc + inh)")
    ax.set_ylabel("Current")
    ax.set_title("Excitatory / Inhibitory / Total currents")
    ax.legend(loc='upper right')
    ax.grid(True)

    # --- 3) Membrane potential ---
    ax = axes[2]
    ax.plot(t, mem)
    ax.hlines(y=1.0, xmin=0, xmax=T - 1, color="r", linestyle="--", label="Threshold")
    ax.set_ylabel("V_mem")
    ax.set_title("Membrane potential")
    ax.legend()
    ax.grid(True)

    # --- 4) Output spikes ---
    ax = axes[3]
    ax.stem(t, spk)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Spike")
    ax.set_title("Output spikes")
    ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# -----------------------
# Test response for different delays
# -----------------------
device = "cpu"
neuron = FMFMNeuronInhib(beta_mem=0.5, beta_inh=0.5).to(device)

print("----------------------------------------------------------------")
print("Initial Wights")
print("Excitatory weights (fc_exc.weight):")
print(neuron.fc_exc.weight)

print("Inhibitory weight (w_inh):")
print(neuron.w_inh)

print("----------------------------------------------------------------")

for delay in range(1, 21):
    spike_seq = generate_pulse_echo(delay=delay).to(device)
    spk_out, exc_rec, inh_rec, mem_rec = neuron(spike_seq)  # [T, 1, 1] each

    total_spikes = spk_out.sum().item()
    max_mem = mem_rec.max().item()
    print(f"Delay {delay:2d} -> spikes: {total_spikes:.1f}, max V_mem: {max_mem:.3f}")

# Example: plot dynamics for a specific delay
# delay = 7
# spike_seq = generate_pulse_echo(delay=delay).to(device)
# spk_out, exc_rec, inh_rec, mem_rec = neuron(spike_seq)

# plot_neuron_states(spike_seq, spk_out, exc_rec, inh_rec, mem_rec,
#                    delay=delay, batch_idx=0)

losses = []  # To store the loss for each step

# Track the progress of training
print("Training starts...")

neuron.fc_exc.weight.requires_grad_(True)
neuron.w_inh.requires_grad_(True)

optimizer = torch.optim.Adam([neuron.fc_exc.weight, neuron.w_inh], lr=2e-2)
steps = 700
target_d = 5
for step in range(steps):
    optimizer.zero_grad()
    loss = 0.0
    for d in range(1, 10):
        spike_seq = generate_pulse_echo(delay=d)
        spk, exc, inh, mem = neuron(spike_seq)
        spikes = spk.sum()

        if d == target_d:
            loss += (spikes - 1.0)**2    # want spike
        else:
            loss += 0.2 * (spikes - 0.0)**2  # penalise spikes

    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if step % 20 == 0:
        losses.append(loss.item())
    if step % 100 == 0:  # Print every 100 steps
        print(f"Step {step}/{steps}, Loss: {loss.item():.4f}")



print("----------------------------------------------------------------")

print("Excitatory weights (fc_exc.weight):")
print(neuron.fc_exc.weight)

print("Inhibitory weight (w_inh):")
print(neuron.w_inh)

print("----------------------------------------------------------------")


for delay in range(1, 21):
    spike_seq = generate_pulse_echo(delay=delay).to(device)
    spk_out, exc_rec, inh_rec, mem_rec = neuron(spike_seq)  # [T, 1, 1] each

    total_spikes = spk_out.sum().item()
    max_mem = mem_rec.max().item()
    print(f"Delay {delay:2d} -> spikes: {total_spikes:.1f}, max V_mem: {max_mem:.3f}")

# Plot the loss over time
plt.plot(losses)
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Loss Function over Time')
plt.show()