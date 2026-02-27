import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Network Components (Same as before)
# -------------------------------
class FMFMNeuron(nn.Module):
    def __init__(self, delay, chosen_w1, ute=1.01, beta=0.9, threshold=1.0):
        super().__init__()
        self.beta = float(beta)
        self.chosen_w1 = float(chosen_w1)
        self.ute = float(ute)
        self.w1, self.w2 = self.calculate_weights(delay)
        
        self.fc = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.fc.weight[:] = torch.tensor([[self.w1, self.w2]], dtype=torch.float32)
        self.lif = snn.Leaky(beta=beta, threshold=threshold)

    def calculate_weights(self, delay: int):
        w1 = self.chosen_w1
        w2 = self.ute - (self.beta ** int(delay)) * w1
        return float(w1), float(w2)

    def init_state(self, batch_size: int, device: str):
        mem = self.lif.init_leaky()
        if isinstance(mem, torch.Tensor):
            mem = mem.to(device)
        return mem

    def step(self, x_t, mem):
        cur = self.fc(x_t)
        spk, mem = self.lif(cur, mem)
        return spk, mem

class SpikingWTAReadout(nn.Module):
    def __init__(self, num_inputs, beta=0.9, threshold=1.0):
        super().__init__()
        self.readout = snn.Leaky(beta=beta, threshold=threshold)
        self.fc_exc = nn.Linear(num_inputs, 1, bias=False)
        with torch.no_grad():
            self.fc_exc.weight[:] = torch.ones(1, num_inputs)

    def init_state(self, device="cpu"):
        mem = self.readout.init_leaky()
        if isinstance(mem, torch.Tensor):
            mem = mem.to(device)
        return mem

    def step(self, bank_spk_t, mem):
        cur = self.fc_exc(bank_spk_t)
        spk, mem = self.readout(cur, mem)
        return spk, mem

class FMFM_WTA_Network(nn.Module):
    def __init__(self, delays, chosen_w1=0.7, ute=1.01, beta=0.9, threshold_bank=1.0):
        super().__init__()
        self.delays = np.array(delays, dtype=int)
        self.bank = nn.ModuleList([
            FMFMNeuron(delay=d, chosen_w1=chosen_w1, ute=ute, beta=beta, threshold=threshold_bank)
            for d in self.delays
        ])
        self.readout = SpikingWTAReadout(num_inputs=len(self.delays), beta=0.9)

    @torch.no_grad()
    def forward(self, spike_seq):
        T, B, _ = spike_seq.shape
        device = spike_seq.device
        bank_mem = [neuron.init_state(B, device=device) for neuron in self.bank]
        ro_mem = self.readout.init_state(device=device)

        bank_spk_rec = []
        inhibit = False
        winner_idx = None # Just tracks first winner for batch 0 logic (simplified)
        
        # Note: This simplified forward pass assumes we just want the bank spikes 
        # for visualization. The accurate decoding loop is handled in evaluate()
        for t in range(T):
            x_t = spike_seq[t]
            spk_t_list = []
            
            for i, neuron in enumerate(self.bank):
                spk_i, bank_mem[i] = neuron.step(x_t, bank_mem[i])
                if inhibit: spk_i = torch.zeros_like(spk_i)
                spk_t_list.append(spk_i)

            bank_spk_t = torch.cat(spk_t_list, dim=1)
            bank_spk_rec.append(bank_spk_t)

            ro_spk_t, ro_mem = self.readout.step(bank_spk_t, ro_mem)
            if (not inhibit) and (ro_spk_t.sum() > 0):
                inhibit = True

        return torch.stack(bank_spk_rec, dim=0)

    @torch.no_grad()
    def decode_single(self, spike_seq_single):
        # Dedicated decoder for a single sequence [T, 1, 2] to get exact prediction
        bank_spk = self.forward(spike_seq_single) # [T, 1, N]
        # Find first spike in bank
        # Flatten to [T, N]
        raster = bank_spk[:, 0, :]
        rows, cols = torch.where(raster > 0)
        if len(rows) > 0:
            # First spike time
            first_t = rows.min()
            # Which neuron spiked at that time? (Take the one with smallest delay index)
            neurons_at_t = cols[rows == first_t]
            winner_idx = neurons_at_t.min().item()
            return int(self.delays[winner_idx])
        return None

# -------------------------------
# 2. Noisy Generator
# -------------------------------
def generate_noisy_pulse_echo(num_steps=50, delay=10, batch_size=1, t_pulse=5, 
                             jitter_sigma=1.0, noise_rate=0.05, dropout_prob=0.1, device="cpu"):
    spikes = torch.zeros(num_steps, batch_size, 2, device=device)
    
    # Pulse (Ch 0)
    if 0 <= t_pulse < num_steps:
        spikes[t_pulse, :, 0] = 1.0

    # Echo (Ch 1)
    mask_echo = torch.rand(batch_size, device=device) > dropout_prob
    noise = torch.randn(batch_size, device=device) * jitter_sigma
    t_echo_float = t_pulse + delay + noise
    t_echo = torch.round(t_echo_float).long()

    for b in range(batch_size):
        if mask_echo[b]:
            t = t_echo[b].item()
            if 0 <= t < num_steps:
                spikes[t, b, 1] = 1.0
    
    # Background Noise
    noise_mask = torch.rand_like(spikes) < noise_rate
    spikes = torch.clamp(spikes + noise_mask.float(), 0.0, 1.0)
    return spikes

# -------------------------------
# 3. New Visualization Logic
# -------------------------------
def visualize_impact(net, delay, noise_params, device="cpu"):
    """
    Runs a batch of tests, calculates accuracy, and plots ONE sample 
    input vs output.
    """
    num_trials = 100
    
    # 1. Generate Batch
    spike_seq = generate_noisy_pulse_echo(
        num_steps=40,
        delay=delay,
        batch_size=num_trials,
        t_pulse=5,
        **noise_params,
        device=device
    )
    
    # 2. Calculate Accuracy
    correct = 0
    valid = 0
    for i in range(num_trials):
        single_seq = spike_seq[:, i:i+1, :]
        pred = net.decode_single(single_seq)
        if pred is not None:
            valid += 1
            # Tolerance +/- 1
            if abs(pred - delay) <= 1:
                correct += 1
    
    acc = (correct / num_trials) * 100
    
    # 3. Get Data for Plotting (Use the first trial in the batch)
    sample_input = spike_seq[:, 0, :].cpu() # [T, 2]
    
    # Run network on just this sample to get response for plotting
    # We re-run simply to ensure we capture the specific spike train for the plot
    sample_output = net.forward(spike_seq[:, 0:1, :]) # [T, 1, N]
    sample_output = sample_output[:, 0, :].cpu() # [T, N]

    # 4. Plotting
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    # Subplot 1: Input Raster
    # Transpose to [Channels, Time] for imshow
    # Channel 0 (Pulse) is bottom, Channel 1 (Echo) is top
    input_img = sample_input.T.numpy()
    
    # Custom colormap or just Greys
    ax[0].imshow(input_img, aspect='auto', interpolation='nearest', cmap='Greys', origin='lower')
    ax[0].set_yticks([0, 1])
    ax[0].set_yticklabels(['Pulse', 'Echo'])
    ax[0].set_title(f"Input Spike Train (Sample)\nAccuracy over {num_trials} trials: {acc:.1f}%")
    ax[0].set_ylabel("Input Channels")
    
    # Subplot 2: Bank Response
    # Transpose to [Neurons, Time]
    out_img = sample_output.T.numpy()
    
    im = ax[1].imshow(out_img, aspect='auto', interpolation='nearest', cmap='hot', origin='lower')
    ax[1].set_ylabel("Tuned Delay (Index)")
    ax[1].set_xlabel("Time Step")
    ax[1].set_title("Network Response (Active Neurons)")
    
    # Create descriptive title based on noise params
    param_str = ", ".join([f"{k}={v}" for k,v in noise_params.items() if v > 0])
    if not param_str: param_str = "Clean Signal"
    plt.suptitle(f"Testing Scenario: {param_str}", fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.show()

# -------------------------------
# 4. Main Execution
# -------------------------------
if __name__ == "__main__":
    device = "cpu"
    target_delay = 6
    delays = np.arange(1, 11)
    
    net = FMFM_WTA_Network(
        delays=delays,
        chosen_w1=0.7,
        ute=1.05,  # Increased slightly to be more robust to jitter
        beta=0.9
    ).to(device)
    
    print(f"Target Delay: {target_delay}")
    
    # 
    # Scenario A: Clean
    print("Visualizing Clean...")
    visualize_impact(net, target_delay, 
                     {"jitter_sigma": 0.0, "noise_rate": 0.0}, device)

    # Scenario B: Low Jitter (Realistic)
    print("Visualizing Low Jitter...")
    visualize_impact(net, target_delay, 
                     {"jitter_sigma": 0.5, "noise_rate": 0.0}, device)

    # Scenario C: High Jitter (Stress Test)
    # You will likely see accuracy drop significantly here
    print("Visualizing High Jitter...")
    visualize_impact(net, target_delay, 
                     {"jitter_sigma": 1.5, "noise_rate": 0.0}, device)
    
    # Scenario D: Background Noise (False Positives)
    # You will see random dots in the input raster
    print("Visualizing Background Noise...")
    visualize_impact(net, target_delay, 
                     {"jitter_sigma": 0.0, "noise_rate": 0.1}, device)