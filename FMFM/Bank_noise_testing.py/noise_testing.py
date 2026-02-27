import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
import matplotlib.pyplot as plt

import torch

def generate_noisy_pulse_echo(
    num_steps=50, 
    delay=10, 
    batch_size=1, 
    t_pulse=5, 
    jitter_sigma=1.0,    # Std dev of timing error (Temporal Jitter)
    noise_rate=0.05,     # Prob of random background spike (False Positives)
    dropout_prob=0.1,    # Prob of missing the echo entirely (False Negatives)
    device="cpu"
):
    """
    Generates a pulse-echo sequence with:
    - Timing Jitter: Echo time varies normally around the true delay.
    - Background Noise: Random spikes added to both channels.
    - Dropout: The echo might fail to appear.
    """
    spikes = torch.zeros(num_steps, batch_size, 2, device=device)
    
    # 1. Pulse Generation (Channel 0) - usually reliable
    if 0 <= t_pulse < num_steps:
        spikes[t_pulse, :, 0] = 1.0

    # 2. Echo Generation (Channel 1) - prone to physics
    # Determine if echo survives (Dropout)
    mask_echo = torch.rand(batch_size, device=device) > dropout_prob
    
    # Calculate Jittered Timing
    # t_echo = t_pulse + delay + noise
    noise = torch.randn(batch_size, device=device) * jitter_sigma
    t_echo_float = t_pulse + delay + noise
    t_echo = torch.round(t_echo_float).long()

    # Apply echoes
    for b in range(batch_size):
        if mask_echo[b]:
            t = t_echo[b].item()
            if 0 <= t < num_steps:
                spikes[t, b, 1] = 1.0
    
    # 3. Background Noise (Poisson-like)
    # Add random spikes to the whole tensor with probability `noise_rate`
    noise_mask = torch.rand_like(spikes) < noise_rate
    
    # Combine true signal with noise (clamp to 1.0)
    spikes = torch.clamp(spikes + noise_mask.float(), 0.0, 1.0)
    return spikes

def evaluate_accuracy(net, delay, noise_params, num_trials=100, device="cpu"):
    """
    Runs the network multiple times and calculates accuracy.
    """
    # Create a batch of trials
    spike_seq = generate_noisy_pulse_echo(
        num_steps=50,
        delay=delay,
        batch_size=num_trials,
        t_pulse=5,
        **noise_params,
        device=device
    )
    
    # Forward pass
    pred_delays_list = []
    
    # We need to process the batch. Your current decode() is set up for single items
    # or specific batch logic. Let's use the forward pass directly.
    bank_spk, readout_spk, inhibit_time, winner_idx = net.forward(spike_seq)
    
    # Logic to extract predictions from the batch results
    # We need to find the winner for EACH item in the batch
    # This requires a slight modification to your network logic to handle 
    # independent batch inhibition, OR we can just loop for this test.
    
    correct_count = 0
    valid_count = 0 # times the network produced ANY answer
    
    # Since the network code provided has a global 'inhibit' flag that might 
    # not be batch-independent in the Python loop, 
    # let's loop through trials for safety in this test function.
    
    for i in range(num_trials):
        single_seq = spike_seq[:, i:i+1, :] # [T, 1, 2]
        pred_delay, _, _, _, _ = net.decode(single_seq)
        
        if pred_delay is not None:
            valid_count += 1
            # We allow a tolerance of +/- 1 step due to jitter
            if abs(pred_delay - delay) <= 1: 
                correct_count += 1
                
    accuracy = correct_count / num_trials if num_trials > 0 else 0
    return accuracy

# -------------------------------
# Single delay-tuned FMFM neuron (step-by-step capable)
# -------------------------------
class FMFMNeuron(nn.Module):
    def __init__(self, delay, chosen_w1, ute=1.01, beta=0.9, threshold=1.0):
        super().__init__()
        self.beta = float(beta)
        self.chosen_w1 = float(chosen_w1)
        self.ute = float(ute)
        self.threshold = float(threshold)

        self.w1, self.w2 = self.calculate_weights(delay)

        self.fc = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.fc.weight[:] = torch.tensor([[self.w1, self.w2]], dtype=torch.float32)

        # explicit threshold for clarity
        self.lif = snn.Leaky(beta=beta, threshold=threshold)

    def calculate_weights(self, delay: int):
        # Design condition at echo time te:
        # U(te) = beta^delay * w1 + w2 = ute  => w2 = ute - beta^delay * w1
        w1 = self.chosen_w1
        w2 = self.ute - (self.beta ** int(delay)) * w1
        return float(w1), float(w2)

    def init_state(self, batch_size: int, device: str):
        # snntorch returns a tensor state; we keep it and pass it back each step
        mem = self.lif.init_leaky()
        if isinstance(mem, torch.Tensor):
            mem = mem.to(device)
        return mem

    def step(self, x_t, mem):
        """
        x_t: [B,2] spikes at time t
        mem: neuron state
        returns: spk_t [B,1], mem_next
        """
        cur = self.fc(x_t)          # [B,1]
        spk, mem = self.lif(cur, mem)
        return spk, mem


# -------------------------------
# Pulse-echo generator
# -------------------------------
def generate_pulse_echo(num_steps=50, delay=10, batch_size=1, t_pulse=5, device="cpu"):
    spikes = torch.zeros(num_steps, batch_size, 2, device=device)
    t_echo = t_pulse + delay
    if 0 <= t_pulse < num_steps:
        spikes[t_pulse, :, 0] = 1.0  # FM1
    if 0 <= t_echo < num_steps:
        spikes[t_echo, :, 1] = 1.0   # FM3
    return spikes


# -------------------------------
# Spiking readout that triggers global inhibition (WTA)
# -------------------------------
class SpikingWTAReadout(nn.Module):
    """
    Readout neuron integrates spikes from the bank.
    When it spikes, we activate a global inhibition flag to suppress
    *future* spikes from the bank, yielding a clean winner.
    """
    def __init__(self, num_inputs, beta=0.9, threshold=1.0):
        super().__init__()
        self.readout = snn.Leaky(beta=beta, threshold=threshold)

        # Excitatory weights from bank spikes -> readout current
        self.fc_exc = nn.Linear(num_inputs, 1, bias=False)
        with torch.no_grad():
            self.fc_exc.weight[:] = torch.ones(1, num_inputs)

    def init_state(self, device="cpu"):
        mem = self.readout.init_leaky()
        if isinstance(mem, torch.Tensor):
            mem = mem.to(device)
        return mem

    def step(self, bank_spk_t, mem):
        """
        bank_spk_t: [B,N] spikes at time t from bank
        returns: ro_spk [B,1], ro_mem_next
        """
        cur = self.fc_exc(bank_spk_t)     # [B,1]
        spk, mem = self.readout(cur, mem)
        return spk, mem


# -------------------------------
# Full network: bank + spiking WTA readout + inhibition
# -------------------------------
class FMFM_WTA_Network(nn.Module):
    def __init__(
        self,
        delays,
        chosen_w1=0.7,
        ute=1.01,
        beta=0.9,
        threshold_bank=1.0,
        beta_readout=0.9,
        threshold_readout=1.0,
    ):
        super().__init__()
        self.delays = np.array(delays, dtype=int)

        self.bank = nn.ModuleList([
            FMFMNeuron(delay=d, chosen_w1=chosen_w1, ute=ute, beta=beta, threshold=threshold_bank)
            for d in self.delays
        ])

        self.readout = SpikingWTAReadout(
            num_inputs=len(self.delays),
            beta=beta_readout,
            threshold=threshold_readout,
        )


    @torch.no_grad()
    def forward(self, spike_seq):
        T, B, _ = spike_seq.shape
        device = spike_seq.device
        N = len(self.bank)

        bank_mem = [neuron.init_state(B, device=device) for neuron in self.bank]
        ro_mem = self.readout.init_state(device=device)

        bank_spk_rec = []
        ro_spk_rec = []

        inhibit = False
        inhibit_time = None
        winner_idx = None

        for t in range(T):
            x_t = spike_seq[t]
            spk_t_list = []
            
            # 1. Run all neurons first
            active_indices_at_t = [] # Track everyone who spiked this step
            
            for i, neuron in enumerate(self.bank):
                spk_i, bank_mem[i] = neuron.step(x_t, bank_mem[i])
                
                if inhibit:
                    spk_i = torch.zeros_like(spk_i)
                
                if spk_i.sum() > 0:
                    active_indices_at_t.append(i)
                    
                spk_t_list.append(spk_i)

            bank_spk_t = torch.cat(spk_t_list, dim=1)
            bank_spk_rec.append(bank_spk_t)

            # 2. Check for winner among the active indices
            if (not inhibit) and (winner_idx is None) and (len(active_indices_at_t) > 0):
                # We have spikes! But which one is the true winner?
                # We need the one associated with the SMALLEST delay value.
                
                # Get the delay values for all neurons that spiked
                spiking_delays = [self.delays[i] for i in active_indices_at_t]
                
                # Find the minimum delay (this avoids the "early arrival" problem)
                min_delay = min(spiking_delays)
                
                # Find which index that delay belongs to
                # Note: np.where returns a tuple, we take [0][0]
                true_winner_local_idx = np.where(self.delays == min_delay)[0][0]
                
                winner_idx = true_winner_local_idx

            # 3. Readout / Inhibition logic
            ro_spk_t, ro_mem = self.readout.step(bank_spk_t, ro_mem)
            ro_spk_rec.append(ro_spk_t.squeeze(-1))

            if (not inhibit) and (ro_spk_t.sum() > 0):
                inhibit = True
                inhibit_time = t

        bank_spk = torch.stack(bank_spk_rec, dim=0)
        readout_spk = torch.stack(ro_spk_rec, dim=0)

        return bank_spk, readout_spk, inhibit_time, winner_idx


    @torch.no_grad()
    def decode(self, spike_seq):
        bank_spk, readout_spk, inhibit_time, winner_idx = self.forward(spike_seq)

        if winner_idx is None:
            return None, bank_spk, readout_spk, inhibit_time, winner_idx

        pred_delay = int(self.delays[winner_idx])
        return pred_delay, bank_spk, readout_spk, inhibit_time, winner_idx


# -------------------------------
# Demo / test
# -------------------------------
if __name__ == "__main__":
    device = "cpu"

    candidate_delays = np.arange(1, 11)  # bank delays 1..10
    net = FMFM_WTA_Network(
        delays=candidate_delays,
        chosen_w1=0.7,
        ute=1.01,
        beta=0.9,
        threshold_bank=1.0,
        beta_readout=0.9,
        threshold_readout=1.0,
    ).to(device)

    unknown_delay = 5
    spike_seq = generate_pulse_echo(num_steps=50, delay=unknown_delay, batch_size=1, device=device)

    pred_delay, bank_spk, readout_spk, inhibit_time, winner_idx = net.decode(spike_seq)

    print(f"True delay: {unknown_delay}")
    print(f"Predicted delay: {pred_delay}")
    print(f"Inhibition triggered at t = {inhibit_time}")
    if winner_idx is not None:
        print(f"Winner index: {winner_idx}, winner tuned delay: {candidate_delays[winner_idx]}")


# ... (Previous code) ...

print("\n--- STRESS TESTING ---")

# 1. Test Robustness to Jitter (Timing Errors)
jitter_levels = [0.0, 0.5, 1.0, 1.5, 2.0]
target_delay = 5

print(f"Testing Delay {target_delay} with increasing Jitter:")
for sigma in jitter_levels:
    params = {
        "jitter_sigma": sigma,
        "noise_rate": 0.0,
        "dropout_prob": 0.0
    }
    acc = evaluate_accuracy(net, target_delay, params, num_trials=50, device=device)
    print(f"  Jitter Sigma {sigma}: {acc*100:.1f}% Accuracy")

# 2. Test Robustness to Background Noise (Clutter)
noise_levels = [0.0, 0.01, 0.05, 0.10,0.2]
print(f"\nTesting Delay {target_delay} with increasing Background Noise:")
for rate in noise_levels:
    params = {
        "jitter_sigma": 0.0,
        "noise_rate": rate,
        "dropout_prob": 0.0
    }
    acc = evaluate_accuracy(net, target_delay, params, num_trials=50, device=device)
    print(f"  Noise Rate {rate}: {acc*100:.1f}% Accuracy")

# 3. Visualizing a Noisy Input
print("\nPlotting a sample noisy input...")
noisy_input = generate_noisy_pulse_echo(
    num_steps=50, 
    delay=5, 
    batch_size=1, 
    jitter_sigma=1.0, 
    noise_rate=0.05
)

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].imshow(noisy_input[:, 0, :].cpu().numpy().T, aspect="auto", interpolation="nearest", cmap="Greys")
ax[0].set_title("Noisy Input Raster (Jitter + Static)")
ax[0].set_ylabel("Channel (0=Pulse, 1=Echo)")
ax[0].set_yticks([0, 1])

# Run this specific noisy input through the net
_, bank_spk_noisy, _, _, _ = net.decode(noisy_input)
ax[1].imshow(bank_spk_noisy[:, 0, :].cpu().numpy().T, aspect="auto", interpolation="nearest", cmap="hot")
ax[1].set_title("Bank Response to Noisy Input")
ax[1].set_xlabel("Time Step")
ax[1].set_ylabel("Delay Tuned Neurons")
plt.tight_layout()
plt.show()