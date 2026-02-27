import torch
import matplotlib.pyplot as plt
import numpy as np

def signal_to_spike_transduction(true_delay, batch_size, signal_strength=0.8):
    """
    Simulates the conversion from Analog Signal -> Noisy Spike Train
    
    Returns:
        spikes: torch.Tensor of shape (time_steps, batch_size)
        prob_dists: torch.Tensor of shape (time_steps, batch_size) containing the Gaussian probability distributions
        centers: torch.Tensor of shape (batch_size,) containing the jittered delay centers
    """
    # 1. Ideally, the spike happens exactly at 'true_delay'
    time_axis = torch.arange(0, 50)
    
    # 2. Create a Gaussian probability distribution (The Analog Signal shape)
    # The 'signal' is a bump centered at the delay.
    # We add randomness to the center (Phase jitter)
    center = true_delay + torch.randn(batch_size) * 0.5 
    
    spikes = torch.zeros(50, batch_size)
    prob_dists = torch.zeros(50, batch_size)
    
    for b in range(batch_size):
        # Gaussian curve representing the probability of firing at any time t
        # (This represents the analog waveform peak)
        prob_dist = torch.exp(-0.5 * (time_axis - center[b])**2)
        
        # Scale by signal strength (attenuation)
        prob_dist = prob_dist * signal_strength 
        
        # Store probability distribution
        prob_dists[:, b] = prob_dist
        
        # 3. Transduction: Sample spikes from this probability
        # This naturally creates Jitter (sampling from width) and Dropout (sampling low prob)
        generated_spikes = torch.bernoulli(prob_dist)
        spikes[:, b] = generated_spikes

    return spikes, prob_dists, center


def plot_spikes(spikes, title="Output Spike Raster Plot"):
    """
    Visualizes spike output as a raster plot.
    
    Args:
        spikes: torch.Tensor of shape (time_steps, batch_size) where 1 indicates a spike
        title: Title for the plot
    """
    spikes_np = spikes.numpy() if isinstance(spikes, torch.Tensor) else spikes
    time_steps, num_neurons = spikes_np.shape
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create raster plot
    for neuron_idx in range(num_neurons):
        spike_times = np.where(spikes_np[:, neuron_idx] == 1)[0]
        ax.vlines(spike_times, neuron_idx - 0.4, neuron_idx + 0.4, colors='black', linewidth=1)
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Neuron / Channel Index', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-1, time_steps)
    ax.set_ylim(-1, num_neurons)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def plot_spikes_with_gaussian(spikes, prob_dists, title="Gaussian Distribution and Spike Raster"):
    """
    Visualizes Gaussian curves above spike raster plot.
    
    Args:
        spikes: torch.Tensor of shape (time_steps, batch_size) where 1 indicates a spike
        prob_dists: torch.Tensor of shape (time_steps, batch_size) containing Gaussian probability distributions
        title: Title for the plot
    """
    spikes_np = spikes.numpy() if isinstance(spikes, torch.Tensor) else spikes
    prob_dists_np = prob_dists.numpy() if isinstance(prob_dists, torch.Tensor) else prob_dists
    time_steps, num_neurons = spikes_np.shape
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2]})
    
    # Top subplot: Gaussian curves
    ax_gauss = axes[0]
    colors = plt.cm.tab20(np.linspace(0, 1, num_neurons))
    
    for neuron_idx in range(num_neurons):
        ax_gauss.plot(prob_dists_np[:, neuron_idx], label=f'Neuron {neuron_idx}', 
                      color=colors[neuron_idx], linewidth=2, alpha=0.8)
    
    ax_gauss.set_ylabel('Probability', fontsize=11)
    ax_gauss.set_title('Gaussian Probability Distributions', fontsize=12, fontweight='bold')
    ax_gauss.grid(True, alpha=0.3)
    ax_gauss.legend(loc='upper right', fontsize=9, ncol=2)
    ax_gauss.set_xlim(-1, time_steps)
    
    # Bottom subplot: Raster plot
    ax_raster = axes[1]
    for neuron_idx in range(num_neurons):
        spike_times = np.where(spikes_np[:, neuron_idx] == 1)[0]
        ax_raster.vlines(spike_times, neuron_idx - 0.4, neuron_idx + 0.4, 
                        colors=colors[neuron_idx], linewidth=2)
    
    ax_raster.set_xlabel('Time (ms)', fontsize=11)
    ax_raster.set_ylabel('Neuron Index', fontsize=11)
    ax_raster.set_title('Output Spike Raster', fontsize=12, fontweight='bold')
    ax_raster.set_xlim(-1, time_steps)
    ax_raster.set_ylim(-1, num_neurons)
    ax_raster.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


def plot_spikes_heatmap(spikes, title="Output Spike Heatmap"):
    """
    Visualizes spikes as a heatmap.
    
    Args:
        spikes: torch.Tensor of shape (time_steps, batch_size)
        title: Title for the plot
    """
    spikes_np = spikes.numpy() if isinstance(spikes, torch.Tensor) else spikes
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(spikes_np.T, aspect='auto', cmap='binary', interpolation='nearest', origin='lower')
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Neuron / Channel Index', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Spike Activity')
    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    # Generate spikes
    true_delay = 20
    batch_size = 5
    signal_strength = 0.8
    
    spikes, prob_dists, centers = signal_to_spike_transduction(true_delay, batch_size, signal_strength)
    
    # Display visualization with Gaussian curves above raster plot
    fig = plot_spikes_with_gaussian(spikes, prob_dists, 
                                    f"Gaussian Distribution and Spike Raster (Delay={true_delay}, Signal Strength={signal_strength})")
    
    plt.show()