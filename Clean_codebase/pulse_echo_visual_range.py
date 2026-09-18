import matplotlib.pyplot as plt
import torch

from sim_modules import TonotopicEncoder, generate_fm_echo_from_waveform


CONFIG = {
    "seed": 7,
    "simulation": {
        "device": "cpu",
        "total_time": 200,
        "sample_rate_hz": 1000_000,
        "step_duration_ms": 0.1,
    },
    "encoder": {
        "freq_channels": 30,
        "f_min": 30,
        "f_max": 90,
    },
    "pulse": {
        "start_time": 20,
        "duration": 50,
        "f_start": 90,
        "f_end": 30,
        "amplitude": 1.0,
        "max_spike_prob": 0.9,
        "bandwidth_khz": 8,
    },
    "echo": {
        "distance_m": 1.37,
        "reflectivity": 0.6,
        "max_spike_prob": 0.9,
        "bandwidth_khz": 8,
        "surface_roughness": 0.55,
        "snr_db": None,
        "clutter_fraction": 0,
        "num_clutter_paths": 320,
        "clutter_min_distance_m": 0.4,
        "clutter_max_distance_m": 5.2,
        "clutter_reflectivity_range": (0.1, 0.2),
        "clutter_roughness_range": (0.2, 0.8),
        "clutter_texture_scale_range_m": (0.0006, 0.006),
    },
    "plots": {
        "pulse_xlim_ms": (1.5, 7.5),
        "echo_xlim_ms": (9.5, 15.5),
        "pulse_figsize": (10, 7),
        "echo_figsize": (10, 7),
        "combined_figsize": (14, 8),
        "pulse_color": "tab:blue",
        "pulse_spike_color": "blue",
        "echo_color": "tab:red",
        "echo_spike_color": "red",
        "pulse_waveform_title": "Outgoing Pulse Waveform",
        "pulse_spike_title": "Outgoing Pulse Spikes (Chirp)",
        "echo_waveform_title": "Returning Echo Waveform (Heavy Clutter)",
        "echo_spike_title_template": "Returning Echo Spikes (Heavy Clutter, delay={true_delay} steps)",
    },
}


def plot_spikes(ax, spike_tensor, title, color, total_time, xlim_steps=None):
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
    if xlim_steps is not None:
        ax.set_xlim(*xlim_steps)
    else:
        ax.set_xlim(0, total_time)
    ax.set_ylim(freq_channels - 1, 0)
    ax.grid(True, alpha=0.3)


def plot_waveform(ax, waveform, sample_rate_hz, title, color, xlim_ms=None):
    """Plot 1-D waveform against time in milliseconds."""
    waveform = waveform.detach().cpu().flatten()
    time_ms = torch.arange(waveform.numel()) * (1000.0 / float(sample_rate_hz))
    ax.plot(time_ms.numpy(), waveform.numpy(), color=color, linewidth=1.0)
    if xlim_ms is not None:
        ax.set_xlim(*xlim_ms)
    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)


def main():
    torch.manual_seed(CONFIG["seed"])

    sim_cfg = CONFIG["simulation"]
    encoder_cfg = CONFIG["encoder"]
    pulse_cfg = CONFIG["pulse"]
    echo_cfg = CONFIG["echo"]
    plot_cfg = CONFIG["plots"]

    device = sim_cfg["device"]
    total_time = sim_cfg["total_time"]
    sample_rate_hz = sim_cfg["sample_rate_hz"]
    step_duration_ms = sim_cfg["step_duration_ms"]

    def ms_xlim_to_step_xlim(xlim_ms):
        if xlim_ms is None:
            return None
        start_step = xlim_ms[0] / step_duration_ms
        end_step = xlim_ms[1] / step_duration_ms
        return (max(0.0, start_step), min(float(total_time), end_step))

    pulse_xlim_steps = ms_xlim_to_step_xlim(plot_cfg["pulse_xlim_ms"])
    echo_xlim_steps = ms_xlim_to_step_xlim(plot_cfg["echo_xlim_ms"])

    encoder = TonotopicEncoder(**encoder_cfg)

    pulse_waveform = encoder.generate_fm_sweep_waveform(
        num_steps=total_time,
        start_time=pulse_cfg["start_time"],
        duration=pulse_cfg["duration"],
        f_start=pulse_cfg["f_start"],
        f_end=pulse_cfg["f_end"],
        sample_rate_hz=sample_rate_hz,
        step_duration_ms=step_duration_ms,
        amplitude=pulse_cfg["amplitude"],
        device=device,
    )

    pulse = encoder.encode_waveform_to_spikes(
        waveform=pulse_waveform,
        num_steps=total_time,
        max_spike_prob=pulse_cfg["max_spike_prob"],
        sample_rate_hz=sample_rate_hz,
        step_duration_ms=step_duration_ms,
        bandwidth_khz=pulse_cfg["bandwidth_khz"],
        device=device,
    )

    echo, echo_waveform, true_delay = generate_fm_echo_from_waveform(
        pulse_waveform=pulse_waveform,
        encoder=encoder,
        num_steps=total_time,
        distance_m=echo_cfg["distance_m"],
        reflectivity=echo_cfg["reflectivity"],
        max_spike_prob=echo_cfg["max_spike_prob"],
        bandwidth_khz=echo_cfg["bandwidth_khz"],
        sample_rate_hz=sample_rate_hz,
        step_duration_ms=step_duration_ms,
        surface_roughness=echo_cfg["surface_roughness"],
        snr_db=echo_cfg["snr_db"],
        clutter_fraction=echo_cfg["clutter_fraction"],
        num_clutter_paths=echo_cfg["num_clutter_paths"],
        clutter_min_distance_m=echo_cfg["clutter_min_distance_m"],
        clutter_max_distance_m=echo_cfg["clutter_max_distance_m"],
        clutter_reflectivity_range=echo_cfg["clutter_reflectivity_range"],
        clutter_roughness_range=echo_cfg["clutter_roughness_range"],
        clutter_texture_scale_range_m=echo_cfg["clutter_texture_scale_range_m"],
        device=device,
    )
    echo_spike_title = plot_cfg["echo_spike_title_template"].format(true_delay=true_delay)

    pulse_fig, pulse_axes = plt.subplots(2, 1, figsize=plot_cfg["pulse_figsize"], sharex=False)
    plot_waveform(
        pulse_axes[0],
        pulse_waveform,
        sample_rate_hz,
        plot_cfg["pulse_waveform_title"],
        plot_cfg["pulse_color"],
        xlim_ms=plot_cfg["pulse_xlim_ms"],
    )
    plot_spikes(
        pulse_axes[1],
        pulse,
        plot_cfg["pulse_spike_title"],
        plot_cfg["pulse_spike_color"],
        total_time,
        xlim_steps=pulse_xlim_steps,
    )
    pulse_fig.tight_layout()

    echo_fig, echo_axes = plt.subplots(2, 1, figsize=plot_cfg["echo_figsize"], sharex=False)
    plot_waveform(
        echo_axes[0],
        echo_waveform,
        sample_rate_hz,
        plot_cfg["echo_waveform_title"],
        plot_cfg["echo_color"],
        xlim_ms=plot_cfg["echo_xlim_ms"],
    )
    plot_spikes(
        echo_axes[1],
        echo,
        echo_spike_title,
        plot_cfg["echo_spike_color"],
        total_time,
        xlim_steps=echo_xlim_steps,
    )
    echo_fig.tight_layout()

    combined_fig, combined_axes = plt.subplots(2, 2, figsize=plot_cfg["combined_figsize"])
    plot_spikes(
        combined_axes[1, 0],
        pulse,
        plot_cfg["pulse_spike_title"],
        plot_cfg["pulse_spike_color"],
        total_time,
    )
    plot_spikes(
        combined_axes[1, 1],
        echo,
        echo_spike_title,
        plot_cfg["echo_spike_color"],
        total_time,
    )
    plot_waveform(
        combined_axes[0, 0],
        pulse_waveform,
        sample_rate_hz,
        plot_cfg["pulse_waveform_title"],
        plot_cfg["pulse_color"],
    )
    plot_waveform(
        combined_axes[0, 1],
        echo_waveform,
        sample_rate_hz,
        plot_cfg["echo_waveform_title"],
        plot_cfg["echo_color"],
    )

    combined_fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
