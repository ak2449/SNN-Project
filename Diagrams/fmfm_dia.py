import torch
import torch.nn as nn
import snntorch as snn
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# 1. Define the Neuron (Modified to return membrane potential)
# -----------------------
class FMFMNeuron(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()
        self.fc = nn.Linear(2, 1, bias=False)
        
        # Weights: FM1 (0.7) is priming, FM3 (0.5) is trigger
        with torch.no_grad():
            self.fc.weight[:] = torch.tensor([[0.7, 0.597]]) 
        
        # LIF Neuron
        self.lif = snn.Leaky(beta=beta)

    def forward(self, spike_seq):
        num_steps, batch_size, _ = spike_seq.shape
        mem = self.lif.init_leaky()
        
        spk_rec = []
        mem_rec = [] # Recording membrane potential for plotting

        for t in range(num_steps):
            cur = self.fc(spike_seq[t])
            spk, mem = self.lif(cur, mem)
            
            spk_rec.append(spk)
            mem_rec.append(mem)

        return torch.stack(spk_rec), torch.stack(mem_rec)

# -----------------------
# 2. Helper to generate inputs
# -----------------------
def generate_pulse_echo(num_steps=20, delay=5):
    spikes = torch.zeros(num_steps, 1, 2)
    t_pulse = 5
    t_echo = t_pulse + delay
    
    if t_echo < num_steps:
        spikes[t_pulse, 0, 0] = 1.0  # FM1 (Pulse)
        spikes[t_echo, 0, 1] = 1.0   # FM3 (Echo)
    
    return spikes

# -----------------------
# 3. Run Simulation
# -----------------------
device = 'cpu'
neuron = FMFMNeuron(beta=0.9).to(device) # Slightly lower beta to visually exaggerate decay

# Simulation 1: Short Delay (Hit)
delay_hit = 5
input_hit = generate_pulse_echo(delay=delay_hit)
spk_hit, mem_hit = neuron(input_hit)

# Simulation 2: Long Delay (Miss)
delay_miss = 10
input_miss = generate_pulse_echo(delay=delay_miss)
spk_miss, mem_miss = neuron(input_miss)

# Simulation 3: Long Delay (Miss)
delay_e = 2
input_early = generate_pulse_echo(delay=delay_e)
spk_e, mem_e = neuron(input_early)



# -----------------------
# 4. Plotting
# -----------------------
# Convert to numpy for plotting
mem_hit_np = mem_hit.detach().squeeze().numpy()
mem_miss_np = mem_miss.detach().squeeze().numpy()
mem_e_np = mem_e.detach().squeeze().numpy()

spk_hit_np = spk_hit.detach().squeeze().numpy()

fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True)


# --- Plot 1: Short Delay (Coincidence) ---
axs[0].plot(mem_e_np, color='tab:red', linewidth=2, label='Membrane Potential')
axs[0].axhline(y=1.0, color='black', linestyle='--', label='Threshold (1.0)')
axs[0].set_title(f"Scenario A: Short Delay", fontsize=12, fontweight='bold')
axs[0].set_xlabel("Time Step")
axs[0].set_ylabel("Membrane Potential (U)")
axs[0].grid(True, alpha=0.3)

# Annotate inputs
axs[0].annotate('Pulse (FM1)', xy=(5, 0.7), xytext=(1, 0.8), arrowprops=dict(facecolor='black', arrowstyle='->'))
axs[0].annotate('Echo (FM3)', xy=(5+2, 1.175), xytext=(9, 0.8), arrowprops=dict(facecolor='black', arrowstyle='->'))
axs[0].text(5+delay_hit-1, 1.1, "Early Spike", color='red', fontweight='bold')

# --- Plot 2: Perfect Delay (Coincidence) ---
axs[1].plot(mem_hit_np, color='tab:blue', linewidth=2, label='Membrane Potential')
axs[1].axhline(y=1.0, color='black', linestyle='--', label='Threshold (1.0)')
axs[1].set_title(f"Scenario B: Perfect Delay", fontsize=12, fontweight='bold')
axs[1].set_xlabel("Time Step")
axs[1].set_ylabel("Membrane Potential (U)")
axs[1].grid(True, alpha=0.3)

# Annotate inputs
axs[1].annotate('Pulse (FM1)', xy=(5, 0.7), xytext=(2, 0.8), arrowprops=dict(facecolor='black', arrowstyle='->'))
axs[1].annotate('Echo (FM3)', xy=(5+delay_hit, 1.02), xytext=(12, 0.8), arrowprops=dict(facecolor='black', arrowstyle='->'))
axs[1].text(5+delay_hit, 1.05, "SPIKE!", color='red', fontweight='bold')


# --- Plot 3: Long Delay (Decay) ---
axs[2].plot(mem_miss_np, color='tab:orange', linewidth=2, label='Membrane Potential')
axs[2].axhline(y=1.0, color='black', linestyle='--', label='Threshold')
axs[2].set_title(f"Scenario C: Long Delay", fontsize=12, fontweight='bold')
axs[2].set_xlabel("Time Step")
axs[2].grid(True, alpha=0.3)

# Annotate inputs
axs[2].annotate('Pulse (FM1)', xy=(5, 0.7), xytext=(8, 0.8), arrowprops=dict(facecolor='black', arrowstyle='->'))
axs[2].annotate('Echo (FM3)', xy=(5+delay_miss, 0.84), xytext=(15, 0.9), arrowprops=dict(facecolor='black', arrowstyle='->'))



plt.tight_layout()
plt.show()