import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Custom Neuron (Manual Physics)
# -------------------------------
class FMFMNeuron(nn.Module):
    def __init__(self, delay, chosen_w1, ute=1.01, beta=0.9, threshold=1.0):
        super().__init__()
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.w1 = float(chosen_w1)
        self.w2 = float(ute - (self.beta ** int(delay)) * self.w1)

    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, 1, device=device)

    def calculate_currents(self, x_t):
        i_pulse = x_t[:, 0:1] * self.w1
        i_echo  = x_t[:, 1:2] * self.w2
        return i_pulse, i_echo

# -------------------------------
# 2. Readout (Fixed Sensitivity)
# -------------------------------
class SpikingWTAReadout(nn.Module):
    def __init__(self, num_inputs, beta=0.7, threshold=1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        
        # FIX: Weight > Threshold ensures immediate firing
        self.weight = 2.0  

    def init_state(self, device):
        return torch.zeros(1, 1, device=device)

    def step(self, input_current, mem):
        mem = self.beta * mem + (input_current * self.weight)
        spk = (mem > self.threshold).float()
        if spk > 0:
            mem = torch.zeros_like(mem) # Hard reset after fire
        return spk, mem

# -------------------------------
# 3. Network with Echo Consumption
# -------------------------------
class FMFM_Final_Network(nn.Module):
    def __init__(self, delays, **kwargs):
        super().__init__()
        self.delays = np.array(delays, dtype=int)
        self.bank = nn.ModuleList([FMFMNeuron(delay=d, **kwargs) for d in self.delays])
        
        self.readout = SpikingWTAReadout(
            num_inputs=len(self.delays),
            beta=0.5, # Fast reset
            threshold=1.0
        )

    @torch.no_grad()
    def forward(self, spike_seq):
        T, B, _ = spike_seq.shape
        device = spike_seq.device
        N = len(self.bank)

        bank_mem = [n.init_state(B, device) for n in self.bank]
        ro_mem = self.readout.init_state(device)
        suppressed_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        bank_spk_rec = []
        ro_spk_rec = []
        detections = []

        for t in range(T):
            x_t = spike_seq[t] 

            # --- A. PREDICTION ---
            pulse_currents = []
            echo_currents = []
            potential_mems = []
            
            for i, neuron in enumerate(self.bank):
                i_p, i_e = neuron.calculate_currents(x_t)
                pulse_currents.append(i_p)
                echo_currents.append(i_e)
                
                # Check potential voltage
                v_pred = neuron.beta * bank_mem[i] + i_p + i_e
                potential_mems.append(v_pred)

            # --- B. ARBITRATION ---
            candidates = []
            for i in range(N):
                if not suppressed_mask[:, i]: 
                    if potential_mems[i] > self.bank[i].threshold:
                        candidates.append(i)
            
            winner_idx = None
            if len(candidates) > 0:
                # Pick smallest delay
                cand_delays = [self.delays[i] for i in candidates]
                min_d = min(cand_delays)
                
                for i in candidates:
                    if self.delays[i] == min_d:
                        winner_idx = i
                        break
                
                detections.append((t, self.delays[winner_idx]))
                suppressed_mask[:, winner_idx] = True

            # --- C. UPDATE (Echo Consumption) ---
            current_spikes = []
            for i in range(N):
                spk = 0.0
                if i == winner_idx:
                    spk = 1.0
                    new_mem = torch.zeros_like(bank_mem[i])
                else:
                    if winner_idx is not None:
                        # Echo Consumed by winner -> Losers only see pulse
                        new_mem = self.bank[i].beta * bank_mem[i] + pulse_currents[i]
                    else:
                        new_mem = potential_mems[i]

                bank_mem[i] = new_mem
                current_spikes.append(torch.tensor([[spk]], device=device))

            bank_spk_t = torch.cat(current_spikes, dim=1)
            bank_spk_rec.append(bank_spk_t)

            # --- D. READOUT ---
            # Sum all spikes (should be only 1 due to WTA, but safe to sum)
            ro_sum = bank_spk_t.sum(dim=1, keepdim=True)
            ro_spk, ro_mem = self.readout.step(ro_sum, ro_mem)
            ro_spk_rec.append(ro_spk.squeeze(-1))

        return torch.stack(bank_spk_rec), torch.stack(ro_spk_rec), detections

# -------------------------------
# 4. Test
# -------------------------------
def generate_multi_echo(num_steps, delays, t_pulse=5):
    spikes = torch.zeros(num_steps, 1, 2)
    spikes[t_pulse, :, 0] = 1.0 
    for d in delays:
        spikes[t_pulse + d, :, 1] = 1.0
    return spikes

if __name__ == "__main__":
    delays = np.arange(1, 15)
    net = FMFM_Final_Network(delays, chosen_w1=0.7, ute=1.01, beta=0.9)
    
    t_pulse = 5
    true_delays = [4, 9,12] 
    spikes = generate_multi_echo(num_steps=40, delays=true_delays, t_pulse=t_pulse)
    
    bank_spk, ro_spk, detections = net(spikes)
    
    print("Pulse at t =", t_pulse)
    print("True delays:", true_delays)
    print("Detections found:(time index, delay)", detections)

    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(spikes[:, 0, 0].cpu().numpy(), color='blue', label="Pulse")
    plt.plot(spikes[:, 0, 1].cpu().numpy(), color='green', label="Echo")
    plt.title("Input Spike Train")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(ro_spk[:, 0], 'r-o', label="Readout Spikes")
    plt.title("Readout Neuron (Corrected Sensitivity)")
    plt.ylim(-0.1, 1.2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.imshow(bank_spk[:, 0, :].T, aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
    plt.title("Bank Activity")
    plt.xlabel("Time Step")
    plt.ylabel("Tuned Delay")
    plt.yticks(range(len(delays)), delays)
    
    plt.tight_layout()
    plt.show()