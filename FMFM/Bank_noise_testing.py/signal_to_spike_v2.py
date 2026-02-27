import torch
import matplotlib.pyplot as plt
import numpy as np

def signal_to_spike_transduction(true_delay, batch_size, t_pulse=5, signal_strength=0.8, num_steps=50):
    """
    Simulates the FM-FM sequence:
    - Channel 0: FM1 Transmitted Pulse (Precise, high amplitude)
    - Channel 1: FM3 Received Echo (Jittery, attenuated, delayed)
    
    Returns:
        spikes: [num_steps, batch_size, 2]
        prob_dists: [num_steps, batch_size, 2]
    """
    time_axis = torch.arange(0, num_steps).float()
    
    # Initialize outputs: (Time, Batch, Channels)
    # Channel 0 = Pulse, Channel 1 = Echo
    spikes = torch.zeros(num_steps, batch_size, 2)
    prob_dists = torch.zeros(num_steps, batch_size, 2)
    
    # ---------------------------------------------------------
    # Channel 0: FM1 Pulse Generation (High precision, Strong)
    # ---------------------------------------------------------
    # The pulse happens exactly at t_pulse (very little jitter)
    pulse_center = t_pulse
    # Narrow sigma implies high precision in timing
    sigma_pulse = 0.5 
    
    # Calculate Gaussian for Pulse
    # Note: We broadcast across batch (pulse is identical for all trials usually)
    dist_pulse = torch.exp(-0.5 * ((time_axis - pulse_center) / sigma_pulse)**2)
    
    # Fill Batch for Channel 0
    for b in range(batch_size):
        prob_dists[:, b, 0] = dist_pulse
        # Sample spikes (Bernoulli)
        spikes[:, b, 0] = torch.bernoulli(dist_pulse)

    # ---------------------------------------------------------
    # Channel 1: FM3 Echo Generation (Noisy, Delayed)
    # ---------------------------------------------------------
    # The echo happens at t_pulse + delay + noise
    # We add randomness to the center (Phase jitter)
    jitter = torch.randn(batch_size) * 0.5 
    echo_centers = t_pulse + true_delay + jitter
    sigma_echo = 1.3 # Wider sigma implies signal spread/uncertainty
    
    for b in range(batch_size):
        # Gaussian curve for Echo
        center = echo_centers[b]
        dist_echo = torch.exp(-0.5 * ((time_axis - center) / sigma_echo)**2)
        
        # Scale by signal strength (attenuation due to distance)
        dist_echo = dist_echo * signal_strength 
        
        prob_dists[:, b, 1] = dist_echo
        
        # Sample spikes
        spikes[:, b, 1] = torch.bernoulli(dist_echo)

    return spikes, prob_dists, echo_centers


def plot_pulse_echo_sequence(spikes, prob_dists, batch_index=0, title="FM-FM Pulse-Echo Sequence"):
    """
    Visualizes Channel 0 (Pulse) and Channel 1 (Echo) for a single specific trial.
    """
    # Extract data for the specific batch index
    # Shapes become [Time, 2]
    spikes_sample = spikes[:, batch_index, :]
    probs_sample = prob_dists[:, batch_index, :]
    
    time_steps = spikes.shape[0]
    t = np.arange(time_steps)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # -----------------------------
    # Top Plot: Analog Probabilities (The "Signal")
    # -----------------------------
    ax0 = axes[0]
    ax0.plot(t, probs_sample[:, 0], color='blue', linewidth=2, label='FM1 Pulse (Tx)')
    ax0.plot(t, probs_sample[:, 1], color='red', linewidth=2, label='FM3 Echo (Rx)')
    
    ax0.set_ylabel('Spike Probability\n(Analog Signal strength)')
    ax0.set_title(f"{title} - Analog Probability Distributions", fontweight='bold')
    ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.3)
    ax0.set_ylim(-0.1, 1.1)
    
    # -----------------------------
    # Bottom Plot: Spike Raster (The "Input to Neuron")
    # -----------------------------
    ax1 = axes[1]
    
    # Channel 0 Spikes (Pulse)
    pulse_spikes = np.where(spikes_sample[:, 0] == 1)[0]
    ax1.vlines(pulse_spikes, 0.6, 1.4, color='blue', linewidth=2, label='Pulse Spikes')
    
    # Channel 1 Spikes (Echo)
    echo_spikes = np.where(spikes_sample[:, 1] == 1)[0]
    ax1.vlines(echo_spikes, -0.4, 0.4, color='red', linewidth=2, label='Echo Spikes')
    
    # Formatting
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Ch1: Echo', 'Ch0: Pulse'])
    ax1.set_xlabel('Time Step (ms)')
    ax1.set_title("Resulting Spike Trains (Neural Input)", fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(0, time_steps)
    ax1.set_ylim(-1, 2)
    
    plt.tight_layout()
    return fig

# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    TRUE_DELAY = 15
    T_PULSE = 5
    BATCH_SIZE = 5
    SIGNAL_STRENGTH = 0.85
    
    # Run Simulation
    spikes, prob_dists, centers = signal_to_spike_transduction(
        true_delay=TRUE_DELAY, 
        batch_size=BATCH_SIZE, 
        t_pulse=T_PULSE, 
        signal_strength=SIGNAL_STRENGTH
    )
    
    # Print shapes for verification
    print(f"Spikes Shape: {spikes.shape} (Steps, Batch, Channels)")
    
    # Visualize the first trial in the batch
    plot_pulse_echo_sequence(
        spikes, 
        prob_dists, 
        batch_index=0, 
        title=f"Simulation: Delay={TRUE_DELAY}ms"
    )
    
    plt.show()