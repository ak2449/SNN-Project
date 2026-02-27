import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
import matplotlib.pyplot as plt

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


# -------------------------------
# Generate input with UNKNOWN delay (for testing)
# -------------------------------
def generate_pulse_echo(num_steps=50, delay=10, batch_size=1, t_pulse=5):
    spikes = torch.zeros(num_steps, batch_size, 2)
    t_echo = t_pulse + delay
    if t_echo < num_steps:
        spikes[t_pulse, :, 0] = 1.0  # FM1
        spikes[t_echo, :, 1] = 1.0   # FM3
    return spikes


# -------------------------------
# Bank + decoder
# -------------------------------
class FMFMNeuronBank(nn.Module):
    def __init__(self, delays, chosen_w1=0.7, ute=1.01, beta=0.9):
        super().__init__()
        self.delays = np.array(delays, dtype=int)

        # ModuleList so parameters are registered properly
        self.neurons = nn.ModuleList([
            FMFMNeuron(delay=d, chosen_w1=chosen_w1, ute=ute, beta=beta)
            for d in self.delays
        ])

    @torch.no_grad()
    def decode_delay(self, spike_seq, score_window=3):
        """
        Runs the unknown spike_seq through all delay-tuned neurons.
        Returns:
          pred_delay: int
          scores: tensor [num_delays]
          spk_all: tensor [T, B, num_delays]
        """
        T, B, _ = spike_seq.shape

        # Find FM3 time (assumes single FM3 spike per sample; if none, fallback to whole-trace score)
        fm3_times = torch.argmax(spike_seq[:, :, 1], dim=0)  # [B]
        print(spike_seq)
        print(fm3_times)
        has_fm3 = spike_seq[:, :, 1].sum(dim=0) > 0          # [B]

        spk_list = []
        scores = []

        for neuron in self.neurons:
            spk = neuron(spike_seq)          # [T,B,1]
            spk = spk.squeeze(-1)            # [T,B]
            spk_list.append(spk)

            if bool(has_fm3.item()):  # batch_size=1 typical here
                te = int(fm3_times.item())
                t0 = max(0, te)
                t1 = min(T, te + score_window)
                s = spk[t0:t1].sum()          # spikes shortly after FM3
            else:
                s = spk.sum()                 # fallback
            scores.append(s)

        scores = torch.stack(scores)          # [num_delays]
        spk_all = torch.stack(spk_list, dim=-1)  # [T,B,num_delays]

        best_idx = int(torch.argmax(scores).item())
        pred_delay = int(self.delays[best_idx])
        return pred_delay, scores, spk_all


# -------------------------------
# Demo: decode an unknown delay
# -------------------------------
device = "cpu"

candidate_delays = np.arange(1, 11)  # bank delays 1..10
bank = FMFMNeuronBank(candidate_delays, chosen_w1=0.7, ute=1.01, beta=0.9).to(device)

unknown_delay = 6  # pretend we don't know this
spike_seq = generate_pulse_echo(num_steps=50, delay=unknown_delay, batch_size=1).to(device)

pred_delay, scores, spk_all = bank.decode_delay(spike_seq, score_window=3)

print(f"True delay: {unknown_delay}")
print(f"Predicted delay: {pred_delay}")
print("Scores per candidate delay:", scores.cpu().numpy())

# Plot scores vs delay
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
