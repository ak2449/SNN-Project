import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch


def plot_pulse_echo_raster(
    ax,
    pulse_spikes,
    echo_spikes,
    title,
    pulse_color="#3B6CA8",
    echo_color="#D86B6B",
    total_time=None,
    pulse_alpha=0.6,
    echo_alpha=0.6,
):
    """Overlay pulse/echo raster on a shared axis."""
    assert pulse_spikes.shape == echo_spikes.shape, "Pulse and echo must have same shape [T,B,F]."

    t_steps, _, freq_channels = pulse_spikes.shape
    if total_time is None:
        total_time = t_steps

    t_p, _, f_p = torch.where(pulse_spikes > 0)
    ax.scatter(t_p.cpu().numpy(), f_p.cpu().numpy(), s=5, c=pulse_color, alpha=pulse_alpha, label="Pulse")

    t_e, _, f_e = torch.where(echo_spikes > 0)
    ax.scatter(t_e.cpu().numpy(), f_e.cpu().numpy(), s=5, c=echo_color, alpha=echo_alpha, label="Echo")

    ax.set_title(title)
    ax.set_ylabel(f"Frequency Channel (0=High, {freq_channels-1}=Low)")
    ax.set_xlabel("Time (steps)")
    ax.set_xlim(0, total_time)
    ax.set_ylim(freq_channels - 1, 0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
 


def plot_scenario_results(
    pulse_spikes,
    echo_spikes,
    bank_spikes,
    num_steps,
    pulse_start_time,
    pulse_color="#3B6CA8",
    echo_color="#D86B6B",
):
    """Render raster and delay bank response plots for a scenario."""
    fig = plt.figure(constrained_layout=True, figsize=(14, 7))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0])
    plot_pulse_echo_raster(
        ax=ax1,
        pulse_spikes=pulse_spikes,
        echo_spikes=echo_spikes,
        title="Pulse + Echo Raster (Time vs Frequency Channel)",
        pulse_color=pulse_color,
        echo_color=echo_color,
        total_time=num_steps,
    )

    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(
        bank_spikes[:, 0, :].T.detach().cpu().numpy(),
        aspect="auto",
        origin="lower",
        cmap="Greys",
        interpolation="nearest",
    )
    ax2.set_ylabel("Neuron (Delay index)")
    ax2.set_xlabel("Time step")
    ax2.set_title("FM-FM Bank Response (Delay-Tuned Neurons)")

    plt.suptitle("FM Sweep, Echo, and Delay-Tuned Neuron Response")
    plt.show()
