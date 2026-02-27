import torch
import matplotlib.pyplot as plt
import numpy as np

def signal_to_spike_transduction(true_delay, batch_size, t_pulse=5, signal_strength=0.8, num_steps=50):
    """
    Simulates the FM-FM sequence:
    - Channel 0: FM1 Transmitted Pulse (Precise, high amplitude)
    - Channel 1: FM3 Received Echo (Jittery, attenuated, delayed)
    """
    time_axis = torch.arange(0, num_steps).float()
    spikes = torch.zeros(num_steps, batch_size, 2)
    prob_dists = torch.zeros(num_steps, batch_size, 2)
    
    # ---------------------------------------------------------
    # Channel 0: FM1 Pulse Generation (High precision)
    # ---------------------------------------------------------
    pulse_center = t_pulse
    sigma_pulse = 0.4
    
    # Calculate Gaussian for Pulse (Broadcast across batch)
    dist_pulse = torch.exp(-0.5 * ((time_axis - pulse_center) / sigma_pulse)**2)
    
    for b in range(batch_size):
        prob_dists[:, b, 0] = dist_pulse
        spikes[:, b, 0] = torch.bernoulli(dist_pulse)

    # ---------------------------------------------------------
    # Channel 1: FM3 Echo Generation (Noisy, Delayed)
    # ---------------------------------------------------------
    # Jitter applied to the CENTER of the gaussian
    jitter = torch.randn(batch_size) * 0.5  # Increased jitter for visibility
    echo_centers = t_pulse + true_delay + jitter
    sigma_echo = 1.3
    
    for b in range(batch_size):
        center = echo_centers[b]
        dist_echo = torch.exp(-0.5 * ((time_axis - center) / sigma_echo)**2)
        dist_echo = dist_echo * signal_strength 
        
        prob_dists[:, b, 1] = dist_echo
        spikes[:, b, 1] = torch.bernoulli(dist_echo)

    return spikes, prob_dists, echo_centers


def plot_batch_overlay(spikes, prob_dists, title="Batch Overlay"):
    """
    Visualizes ALL batch samples overlaid on top of one another.
    """
    batch_size = spikes.shape[1]
    time_steps = spikes.shape[0]
    t = np.arange(time_steps)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # -----------------------------
    # Top Plot: Probability "Bundles" (Visualizing Jitter)
    # -----------------------------
    ax0 = axes[0]
    
    # Plot every single trial's probability curve with transparency
    # This creates a "cloud" showing the variance
    for b in range(batch_size):
        # Pulse (Blue) - High Alpha because they overlap perfectly
        ax0.plot(t, prob_dists[:, b, 0], color='blue', alpha=0.1, linewidth=1)
        # Echo (Red) - Low Alpha to show the spread/jitter
        ax0.plot(t, prob_dists[:, b, 1], color='red', alpha=0.15, linewidth=1)
        
    # Hack for custom legend (since we plotted 100 lines)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Pulse Probability (Fixed)'),
        Line2D([0], [0], color='red', lw=2, label='Echo Probability (Jittered)')
    ]
    
    ax0.set_ylabel('Spike Probability')
    ax0.set_title(f"{title} - Analog Probability Overlay", fontweight='bold')
    ax0.legend(handles=legend_elements, loc="upper right")
    ax0.grid(True, alpha=0.3)
    
    # -----------------------------
    # Bottom Plot: Raster of All Trials
    # -----------------------------
    ax1 = axes[1]
    
    # Iterate through batch to plot spikes
    # Y-axis = Trial Index
    # X-axis = Time
    for b in range(batch_size):
        # Channel 0 (Pulse) - Plot as Blue Dots
        pulse_times = np.where(spikes[:, b, 0] == 1)[0]
        if len(pulse_times) > 0:
            ax1.scatter(pulse_times, [b]*len(pulse_times), 
                       color='blue', s=10, marker='|', alpha=0.7)
            
        # Channel 1 (Echo) - Plot as Red Dots
        echo_times = np.where(spikes[:, b, 1] == 1)[0]
        if len(echo_times) > 0:
            ax1.scatter(echo_times, [b]*len(echo_times), 
                       color='red', s=15, marker='o', alpha=0.7)

    # Fake legend for scatter
    scatter_legend = [
        Line2D([0], [0], color='blue', marker='|', linestyle='None', label='Pulse Spike'),
        Line2D([0], [0], color='red', marker='o', linestyle='None', label='Echo Spike')
    ]

    ax1.set_ylabel('Trial Number (Batch Index)')
    ax1.set_xlabel('Time Step')
    ax1.set_title("Population Raster Plot (All Trials)", fontweight='bold')
    ax1.legend(handles=scatter_legend, loc="upper right")
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(0, time_steps)
    ax1.set_ylim(-1, batch_size)
    
    plt.tight_layout()
    return fig

# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    TRUE_DELAY = 15
    BATCH_SIZE = 25   # Large batch to see the statistical spread
    SIGNAL_STRENGTH = 0.7 # Lower strength to see some "Dropout" (missing red dots)
    
    # Run Simulation
    spikes, prob_dists, centers = signal_to_spike_transduction(
        true_delay=TRUE_DELAY, 
        batch_size=BATCH_SIZE, 
        signal_strength=SIGNAL_STRENGTH
    )
    
    print(f"Simulating {BATCH_SIZE} trials...")
    
    plot_batch_overlay(
        spikes, 
        prob_dists, 
        title=f"Jitter Analysis (Delay={TRUE_DELAY})"
    )
    
    plt.show()