import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
import matplotlib.pyplot as plt

class FMFMNeuron(nn.Module):
    def __init__(self, delay, chosen_w1, ute=1.01, beta=0.95):
        super().__init__()
        self.beta = beta
        self.chosen_w1 = chosen_w1
        self.ute = ute

        self.w1, self.w2 = self.calculate_weights(delay)

        self.fc = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.fc.weight[:] = torch.tensor([[self.w1, self.w2]], dtype=torch.float32)

        self.lif = snn.Leaky(beta=beta)

    def calculate_weights(self, delay):
        w1 = float(self.chosen_w1)
        w2 = float(self.ute) - (float(self.beta) ** int(delay)) * w1
        return w1, w2

    def forward(self, spike_seq, return_mem=False):
        T, B, _ = spike_seq.shape
        mem = self.lif.init_leaky()

        spk_rec = []
        mem_rec = []

        for t in range(T):
            cur = self.fc(spike_seq[t])
            spk, mem = self.lif(cur, mem)
            spk_rec.append(spk)
            mem_rec.append(mem)

        spk_rec = torch.stack(spk_rec, dim=0)   # [T,B,1]
        mem_rec = torch.stack(mem_rec, dim=0)   # [T,B,1]

        if return_mem:
            return spk_rec, mem_rec
        return spk_rec


class FMFMNeuronBank(nn.Module):
    def __init__(self, delays, chosen_w1=0.7, ute=1.01, beta=0.9):
        super().__init__()
        self.delays = np.array(delays, dtype=int)
        self.neurons = nn.ModuleList([
            FMFMNeuron(delay=d, chosen_w1=chosen_w1, ute=ute, beta=beta)
            for d in self.delays
        ])

    @torch.no_grad()
    def decode_delay_just_above_threshold(self, spike_seq, uth=1.0, miss_penalty=5.0):
        """
        Score each neuron by how close its max membrane potential is to uth,
        preferring slightly ABOVE uth.

        Returns:
          pred_delay: int
          scores: tensor [N]
          umax: tensor [N]  (max membrane per neuron)
        """
        scores = []
        umax_list = []

        for neuron in self.neurons:
            _, mem = neuron(spike_seq, return_mem=True)    # mem: [T,B,1]
            u_max = mem.max()                              # scalar
            umax_list.append(u_max)

            delta = u_max - uth  # overshoot

            if delta >= 0:
                # closer to 0+ is better (slightly above threshold)
                score = -delta
            else:
                # strongly penalize missing threshold
                score = -miss_penalty * (-delta)

            scores.append(score)

        scores = torch.stack(scores)        # [N]
        umax = torch.stack(umax_list)       # [N]

        best_idx = int(torch.argmax(scores).item())
        pred_delay = int(self.delays[best_idx])
        return pred_delay, scores, umax

def generate_pulse_echo(num_steps=50, delay=10, batch_size=1, t_pulse=5):
    spikes = torch.zeros(num_steps, batch_size, 2)
    t_echo = t_pulse + delay
    if t_echo < num_steps:
        spikes[t_pulse, :, 0] = 1.0
        spikes[t_echo, :, 1] = 1.0
    return spikes

candidate_delays = np.arange(1, 11)
bank = FMFMNeuronBank(candidate_delays, chosen_w1=0.7, ute=1.01, beta=0.9)

unknown_delay = 5
spike_seq = generate_pulse_echo(num_steps=50, delay=unknown_delay, batch_size=1)

pred_delay, scores, umax = bank.decode_delay_just_above_threshold(
    spike_seq,
    uth=1.0,           # set this to your actual snntorch threshold if different
    miss_penalty=5.0   # increase if you want to heavily prefer “at least crosses”
)

print("True:", unknown_delay, "Pred:", pred_delay)
print("umax:", umax.numpy())
print("scores:", scores.numpy())

plt.figure(figsize=(7,4))
plt.plot(candidate_delays, scores.cpu().numpy(), marker="o")
plt.xlabel("Candidate delay")
plt.ylabel("Score (spikes near FM3)")
plt.title("Delay decoding via neuron bank")
plt.grid(True)
plt.show()
