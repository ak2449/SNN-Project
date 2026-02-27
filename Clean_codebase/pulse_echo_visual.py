import matplotlib.pyplot as plt
import torch

from sim_modules import TonotopicEncoder, generate_fm_echo_from_waveform


def plot_spikes(ax, spike_tensor, title, color, total_time):
    """Plot [T, B, F] spike tensor as time-frequency raster."""
    time_indices, _, freq_indices = torch.where(spike_tensor > 0)
    freq_channels = spike_tensor.shape[2]

    ax.scatter(
        time_indices.cpu().numpy(),
        freq_indices.cpu().numpy(),
        s=5,
        c=color,
        alpha=0.7,
    )
    ax.set_title(title)
    ax.set_ylabel(f"Frequency Channel (0=High, {freq_channels - 1}=Low)")
    ax.set_xlabel("Time (steps)")
    ax.set_xlim(0, total_time)
    ax.set_ylim(freq_channels - 1, 0)
    ax.grid(True, alpha=0.3)


def plot_waveform(ax, waveform, sample_rate_hz, title, color):
    """Plot 1-D waveform against time in milliseconds."""
    waveform = waveform.detach().cpu().flatten()
    time_ms = torch.arange(waveform.numel()) * (1000.0 / float(sample_rate_hz))
    ax.plot(time_ms.numpy(), waveform.numpy(), color=color, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)


def main():
    torch.manual_seed(7)

    device = "cpu"
    total_time = 200
    pulse_duration = 50
    freq_channels = 60
    pulse_start_time = 20
    sample_rate_hz = 250_000
    step_duration_ms = 0.1

    encoder = TonotopicEncoder(
        freq_channels=freq_channels,
        f_min=20,
        f_max=100,
        tuning_width=3.0,
    )

    pulse_waveform = encoder.generate_fm_sweep_waveform(
        num_steps=total_time,
        start_time=pulse_start_time,
        duration=pulse_duration,
        f_start=100,
        f_end=20,
        sample_rate_hz=sample_rate_hz,
        step_duration_ms=step_duration_ms,
        amplitude=1.0,
        device=device,
    )

    pulse = encoder.encode_waveform_to_spikes(
        waveform=pulse_waveform,
        num_steps=total_time,
        max_spike_prob=0.9,
        sample_rate_hz=sample_rate_hz,
        step_duration_ms=step_duration_ms,
        bandwidth_khz=8.0,
        device=device,
    )

    echo, echo_waveform, true_delay = generate_fm_echo_from_waveform(
        pulse_waveform=pulse_waveform,
        encoder=encoder,
        num_steps=total_time,
        distance_m=1.37,  # ~80-step round-trip delay at 0.1 ms/step.
        reflectivity=0.6,
        max_spike_prob=0.9,
        sample_rate_hz=sample_rate_hz,
        step_duration_ms=step_duration_ms,
        surface_roughness=0.55,
        clutter_fraction=0.8,
        num_clutter_paths=320,
        clutter_min_distance_m=0.4,
        clutter_max_distance_m=5.2,
        clutter_reflectivity_range=(0.1, 0.2),
        clutter_roughness_range=(0.2, 0.8),
        clutter_texture_scale_range_m=(0.0006, 0.006),
        device=device,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plot_spikes(axes[1, 0], pulse, "Outgoing Pulse Spikes (Chirp)", "blue", total_time)
    plot_spikes(
        axes[1, 1],
        echo,
        f"Returning Echo Spikes (Heavy Clutter, delay={true_delay} steps)",
        "red",
        total_time,
    )
    plot_waveform(
        axes[0, 0],
        pulse_waveform,
        sample_rate_hz,
        "Outgoing Pulse Waveform",
        "tab:blue",
    )
    plot_waveform(
        axes[0, 1],
        echo_waveform,
        sample_rate_hz,
        "Returning Echo Waveform (Heavy Clutter)",
        "tab:red",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
