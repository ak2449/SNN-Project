import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Delay-tuned FMFM neuron
# -------------------------------

# -------------------------------
# Delay-tuned FMFM neuron
# -------------------------------
class FMFMNeuron(nn.Module):
    def __init__(self, delay, chosen_w1, ute=1.0001, beta=0.95):
        super().__init__()
        self.beta = beta
        self.chosen_w1 = chosen_w1
        self.ute = ute

        self.w1, self.w2 = self.calculate_weights(delay)

        self.fc = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            # weight must be shape [1,2]
            self.fc.weight[:] = torch.tensor([[self.w1, self.w2]], dtype=torch.float32)

        self.lif = snn.Leaky(beta=beta)

    def calculate_weights(self, delay):
        # Using your derived condition:
        # U(te) = beta^delay * w1 + w2  ==> w2 = ute - beta^delay * w1
        w1 = float(self.chosen_w1)
        w2 = float(self.ute) - (float(self.beta) ** int(delay)) * w1

        return w1, w2

    def forward(self, spike_seq):
        """
        spike_seq: [T, B, 2]
        returns: spk_rec [T, B, 1]
        """
        T, B, _ = spike_seq.shape
        mem = self.lif.init_leaky()

        spk_rec = []
        for t in range(T):
            cur = self.fc(spike_seq[t])
            spk, mem = self.lif(cur, mem)
            spk_rec.append(spk)

        return torch.stack(spk_rec, dim=0)
    

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

class FMFMNeuronBank(nn.Module):
    def __init__(self, delays, chosen_w1=0.7, ute=1.01, beta=0.9):
        super().__init__()
        self.delays = np.array(delays, dtype=int)

        self.neurons = nn.ModuleList([
            FMFMNeuron(delay=d, chosen_w1=chosen_w1, ute=ute, beta=beta)
            for d in self.delays
        ])

    @torch.no_grad()
    def run_bank(self, spike_seq):
        """
        Returns:
          spk_all: [T, B, N] where N=len(delays)
        """
        spk_list = []
        for neuron in self.neurons:
            spk = neuron(spike_seq).squeeze(-1)  # [T,B]
            spk_list.append(spk)
        spk_all = torch.stack(spk_list, dim=-1)  # [T,B,N]
        return spk_all

    @torch.no_grad()
    def decode_delay_fm3_agnostic(self, spike_seq):
        """
        FM3-agnostic decoding: pick delay whose tuned neuron spikes most OVERALL.
        Returns:
          pred_delay: int
          scores: [N]
          spk_all: [T,B,N]
        """
        spk_all = self.run_bank(spike_seq)  # [T,B,N]

        # Score each neuron by total spikes across time and batch
        scores = spk_all.sum(dim=(0, 1))    # [N]

        best_idx = int(torch.argmax(scores).item())
        pred_delay = int(self.delays[best_idx])
        return pred_delay, scores, spk_all


candidate_delays = np.arange(1, 11)
bank = FMFMNeuronBank(candidate_delays, chosen_w1=0.7, ute=1.001, beta=0.9)

unknown_delay = 7
spike_seq = generate_pulse_echo(num_steps=50, delay=unknown_delay, batch_size=1)

pred_delay, scores, spk_all = bank.decode_delay_fm3_agnostic(spike_seq)

print("True:", unknown_delay, " Pred:", pred_delay)
print("Scores:", scores.numpy())

plt.figure(figsize=(7,4))
plt.plot(candidate_delays, scores.cpu().numpy(), marker="o")
plt.xlabel("Candidate delay")
plt.ylabel("Score (spikes near FM3)")
plt.title("Delay decoding via neuron bank")
plt.grid(True)
plt.show()

# Optional: visualize spike raster across neurons (T x num_delays)
plt.figure(figsize=(8,4))
plt.imshow(spk_all[:, 0, :].cpu().numpy(), aspect="auto", origin="lower", interpolation="nearest")
plt.xlabel("Neuron tuned delay index")
plt.ylabel("Time step")
plt.title("Spikes from all delay-tuned neurons (columns=delays)")
plt.xticks(np.arange(len(candidate_delays)), candidate_delays)
plt.colorbar(label="Spike")
plt.show()