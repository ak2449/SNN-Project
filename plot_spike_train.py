import matplotlib.pyplot as plt
import torch
import numpy as np
from FMFM import generate_pulse_echo


def plot_spike_train(spike_seq, channel_names=None, save_path=None, title=None):
    """
    spike_seq: torch.Tensor of shape [T, batch, C]
    channel_names: list of str length C
    """
    # Move to CPU and numpy
    seq = spike_seq.detach().cpu().numpy()  # [T, batch, C]
    # If batch > 1, use first example
    seq = seq[:, 0, :]
    T, C = seq.shape

    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(C)]

    # Prepare raster-style plot: for each channel, show spike times as markers
    plt.figure(figsize=(8, 2 + 0.5 * C))
    for c in range(C):
        spike_times = np.where(seq[:, c] > 0.5)[0]
        plt.scatter(spike_times, np.ones_like(spike_times) * c, marker="|", s=200, color="k")

    plt.yticks(range(C), channel_names)
    plt.ylim(-0.5, C - 0.5)
    plt.xlim(0, T - 1)
    plt.xlabel("Time (steps)")
    plt.title(title or "Input spike train (raster)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    # Example usage: generate a pulse-echo with default settings
    delay = 10
    num_steps = 50
    spike_seq = generate_pulse_echo(num_steps=num_steps, delay=delay)

    channel_names = ["FM1 (pulse)", "FM3 (echo)"]
    plot_spike_train(spike_seq, channel_names=channel_names, save_path=None,
                     title=f"Input spike train (delay={delay})")
