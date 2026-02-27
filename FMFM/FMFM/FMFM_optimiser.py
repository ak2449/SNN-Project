import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF

class FMFMNeuron(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()
        self.fc = nn.Linear(2, 1, bias=False)
        self.lif = snn.Leaky(beta=beta)

    def forward(self, spike_seq):
        num_steps = spike_seq.size(0)
        mem = self.lif.init_leaky()
        spk_rec = []

        for t in range(num_steps):
            cur = self.fc(spike_seq[t])
            spk, mem = self.lif(cur, mem)
            spk_rec.append(spk)

        return torch.stack(spk_rec, dim=0)  # [T, B, 1]

def generate_pulse_echo(num_steps=50, delay=10, batch_size=1):
    spikes = torch.zeros(num_steps, batch_size, 2)
    t_pulse = 5
    t_echo = t_pulse + delay
    if t_echo < num_steps:
        spikes[t_pulse, :, 0] = 1.0  # FM1
        spikes[t_echo, :, 1] = 1.0   # FM3
    return spikes

device = "cpu"
model = FMFMNeuron(beta=0.9).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

best_delay = 5
num_steps = 50

for epoch in range(1000):
    loss_epoch = 0.0

    for delay in range(0, 15):
        spike_seq = generate_pulse_echo(num_steps, delay).to(device)
        spk_rec = model(spike_seq)        # [T, 1, 1]

        # total spikes over time and batch
        spk_count = spk_rec.sum()         # scalar

        # target: spike at best_delay, silent otherwise
        target = torch.tensor(1.0 if delay == best_delay else 0.0)

        # simple MSE on spike count
        loss = (spk_count - target)**2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, loss: {loss_epoch:.3f}")


for delay in range(0, 15):
    spk_rec = model(generate_pulse_echo(num_steps, delay).to(device))
    print(delay, spk_rec.sum().item())
