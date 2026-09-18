import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import torch


PULSE_COLOR = "#3B6CA8"
ECHO_COLOR = "#D86B6B"


def plot_pulse_echo_raster(
    ax,
    pulse_spikes,
    echo_spikes,
    title,
    pulse_color=PULSE_COLOR,
    echo_color=ECHO_COLOR,
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
    pulse_color=PULSE_COLOR,
    echo_color=ECHO_COLOR,
):
    """Render summary plots for pulse/echo and delay bank response."""
    fig = plt.figure(constrained_layout=True, figsize=(14, 9))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1.2, 2])

    ax0 = fig.add_subplot(gs[0])
    pulse_sum = pulse_spikes.sum(dim=2).squeeze().cpu().numpy()
    echo_sum = echo_spikes.sum(dim=2).squeeze().cpu().numpy()
    t_rel = np.arange(num_steps)

    valid_idx = t_rel >= 0
    ax0.plot(t_rel[valid_idx], pulse_sum[valid_idx], label="Pulse (sum freq)", color=pulse_color)
    ax0.plot(
        t_rel[valid_idx],
        echo_sum[valid_idx],
        label="Echo (sum freq)",
        color=echo_color,
        linestyle="--",
    )
    ax0.set_ylabel("Spikes (sum freq)")
    ax0.set_xlabel("Time since pulse onset (steps)")
    ax0.set_title("Pulse and Echo (Summed over Frequency Channels)")
    ax0.set_xlim(0, t_rel[-1])
    ax0.legend()

    ax1 = fig.add_subplot(gs[1])
    plot_pulse_echo_raster(
        ax=ax1,
        pulse_spikes=pulse_spikes,
        echo_spikes=echo_spikes,
        title="Pulse + Echo Raster (Time vs Frequency Channel)",
        pulse_color=pulse_color,
        echo_color=echo_color,
        total_time=num_steps,
    )

    ax2 = fig.add_subplot(gs[2])
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


# =====================================================================
# NEW: Doppler bank visualisation
# =====================================================================

def plot_doppler_results(
    pulse_spikes_doppler,
    echo_spikes_doppler,
    doppler_bank_spikes,
    freq_shifts,
    doppler_encoder=None,
    speed_of_sound_m_s=343.0,
    true_velocity_m_s=None,
    decoded_objects=None,
    num_steps=None,
    pulse_color=PULSE_COLOR,
    echo_color=ECHO_COLOR,
    population_color="black",
    population_alpha=0.6,
    true_color=ECHO_COLOR,
    decoded_color=PULSE_COLOR,
):
    """Render summary plots for the Doppler (velocity) estimation channel.

    Parameters
    ----------
    pulse_spikes_doppler, echo_spikes_doppler : Tensor [T, 1, F_doppler]
        Spike trains from the high-resolution DopplerEncoder.
    doppler_bank_spikes : Tensor [T, 1, N_doppler_neurons]
        Population spike output from the Doppler velocity bank.
    freq_shifts : 1-D array
        The frequency-shift values corresponding to each Doppler neuron.
    doppler_encoder : DopplerEncoder, optional
        Used to convert shifts → velocity for axis labels.
    speed_of_sound_m_s : float
        Speed of sound for velocity conversion.
    true_velocity_m_s : float or list, optional
        Ground-truth velocity (or velocities) for annotation.
    decoded_objects : list of DopplerDecodedObject, optional
        Population-decoded hypotheses to overlay.
    num_steps : int, optional
    """
    if num_steps is None:
        num_steps = pulse_spikes_doppler.shape[0]

    fig = plt.figure(constrained_layout=True, figsize=(14, 10))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1.5, 1.2])

    # --- 1. Narrowband raster (Doppler encoder) ---
    ax0 = fig.add_subplot(gs[0])
    plot_pulse_echo_raster(
        ax=ax0,
        pulse_spikes=pulse_spikes_doppler,
        echo_spikes=echo_spikes_doppler,
        title="DopplerEncoder Raster (Narrowband CF region)",
        pulse_color=pulse_color,
        echo_color=echo_color,
        total_time=num_steps,
    )

    # # --- 2. Doppler bank spike raster ---
    # ax1 = fig.add_subplot(gs[1])
    # spk_np = doppler_bank_spikes[:, 0, :].T.detach().cpu().numpy()
    # ax1.imshow(
    #     spk_np,
    #     aspect="auto",
    #     origin="lower",
    #     cmap="Greys",
    #     interpolation="nearest",
    # )
    # ax1.set_ylabel("Neuron (freq-shift index)")
    # ax1.set_xlabel("Time step")
    # ax1.set_title("Doppler Bank Response (Frequency-Shift-Tuned Neurons)")
    ax1 = fig.add_subplot(gs[1])
    spk_np = doppler_bank_spikes[:, 0, :].T.detach().cpu().numpy()
    n_neurons = spk_np.shape[0]

    # Choose y-axis units: velocity if encoder is available, else signed shift channels.
    if doppler_encoder is not None:
        y_axis = np.array([
            doppler_encoder.shift_channels_to_velocity(s, speed_of_sound_m_s)
            for s in freq_shifts
        ])
        y_label = "Preferred velocity (m/s)"
    else:
        y_axis = np.asarray(freq_shifts, dtype=float)
        y_label = "Preferred frequency shift (channels)"

    extent = [0, spk_np.shape[1], y_axis.min(), y_axis.max()]
    ax1.imshow(
        spk_np,
        aspect="auto",
        origin="lower",
        cmap="Greys",
        interpolation="nearest",
        extent=extent,
    )
    ax1.axhline(0.0, color="grey", lw=0.6, alpha=0.5)   # marks stationary
    ax1.set_ylabel(y_label)
    ax1.set_xlabel("Time step")
    ax1.set_title("Doppler Bank Response (Frequency-Shift-Tuned Neurons)")
    # --- 3. Population tuning curve ---
    ax2 = fig.add_subplot(gs[2])
    spike_counts = doppler_bank_spikes.sum(dim=0)[0].detach().cpu().numpy()

    if doppler_encoder is not None:
        velocities = np.array([
            doppler_encoder.shift_channels_to_velocity(s, speed_of_sound_m_s)
            for s in freq_shifts
        ])
        ax2.bar(velocities, spike_counts, width=(velocities[1] - velocities[0]) * 0.8,
                color=population_color, alpha=population_alpha, label="Spike count")
        ax2.set_xlabel("Velocity (m/s)  [positive = approaching]")
    else:
        ax2.bar(freq_shifts, spike_counts, width=0.8,
                color=population_color, alpha=population_alpha, label="Spike count")
        ax2.set_xlabel("Frequency shift (channels)")

    ax2.set_ylabel("Total spikes")
    ax2.set_title("Doppler Bank Population Response")

    # Annotate ground truth.
    if true_velocity_m_s is not None:
        if not hasattr(true_velocity_m_s, "__iter__"):
            true_velocity_m_s = [true_velocity_m_s]
        for v_true in true_velocity_m_s:
            ax2.axvline(v_true, color=true_color, linestyle="--", linewidth=1.5,
                        label=f"True v = {v_true:.2f} m/s")

    # Annotate decoded peaks.
    if decoded_objects is not None:
        for obj in decoded_objects:
            ax2.axvline(obj.velocity_m_s, color=decoded_color, linestyle="-.", linewidth=1.5,
                        label=f"Decoded v = {obj.velocity_m_s:.2f} m/s")

    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Doppler Velocity Estimation")
    plt.show()


def plot_doppler_scenario_results(
    pulse_spikes_doppler,
    echo_spikes_doppler,
    doppler_bank_spikes,
    num_steps,
    pulse_color=PULSE_COLOR,
    echo_color=ECHO_COLOR,
):
    """Render Doppler encoder raster and Doppler bank response in the range-figure style."""
    fig = plt.figure(constrained_layout=True, figsize=(14, 7))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0])
    plot_pulse_echo_raster(
        ax=ax1,
        pulse_spikes=pulse_spikes_doppler,
        echo_spikes=echo_spikes_doppler,
        title="DopplerEncoder Raster (Narrowband CF region)",
        pulse_color=pulse_color,
        echo_color=echo_color,
        total_time=num_steps,
    )

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(
        doppler_bank_spikes[:, 0, :].T.detach().cpu().numpy(),
        aspect="auto",
        origin="lower",
        cmap="Greys",
        interpolation="nearest",
    )
    ax2.set_ylabel("Neuron (Frequency-shift index)")
    ax2.set_xlabel("Time step")
    ax2.set_title("Doppler Bank Response (Frequency-Shift-Tuned Neurons)")

    plt.suptitle("Doppler Encoder and Frequency-Shift Bank Response")
    plt.show()


def plot_range_and_velocity(
    pulse_spikes,
    echo_spikes,
    delay_bank_spikes,
    doppler_bank_spikes,
    candidate_delays,
    freq_shifts,
    doppler_encoder=None,
    speed_of_sound_m_s=343.0,
    step_duration_ms=0.1,
    true_delays=None,
    true_velocity_m_s=None,
    decoded_delay_objects=None,
    decoded_doppler_objects=None,
    num_steps=None,
    range_color=PULSE_COLOR,
    velocity_color=ECHO_COLOR,
    true_color=ECHO_COLOR,
    decoded_color=PULSE_COLOR,
):
    """Combined range + velocity summary figure."""
    if num_steps is None:
        num_steps = pulse_spikes.shape[0]
    delay_step_to_m = 0.5 * speed_of_sound_m_s * (step_duration_ms * 1e-3)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

    # ---- Range (delay bank) population curve ----
    ax_range = axes[0]
    range_counts = delay_bank_spikes.sum(dim=0)[0].detach().cpu().numpy()
    distances = candidate_delays * delay_step_to_m
    ax_range.bar(distances, range_counts,
                 width=(distances[1] - distances[0]) * 0.8 if len(distances) > 1 else 0.01,
                 color=range_color, alpha=0.7, label="Spike count")
    ax_range.set_xlabel("Distance (m)")
    ax_range.set_ylabel("Total spikes")
    ax_range.set_title("Delay Bank — Range Estimation")
    if true_delays is not None:
        for td in (true_delays if hasattr(true_delays, "__iter__") else [true_delays]):
            ax_range.axvline(td * delay_step_to_m, color=true_color, ls="--", lw=1.5,
                             label=f"True d = {td * delay_step_to_m:.3f} m")
    if decoded_delay_objects is not None:
        for obj in decoded_delay_objects:
            ax_range.axvline(obj.delay_steps * delay_step_to_m, color=decoded_color, ls="-.", lw=1.5,
                             label=f"Dec. d = {obj.delay_steps * delay_step_to_m:.3f} m")
    ax_range.legend(loc="upper right", fontsize=8)
    ax_range.grid(True, alpha=0.3)

    # ---- Velocity (Doppler bank) population curve ----
    ax_vel = axes[1]
    vel_counts = doppler_bank_spikes.sum(dim=0)[0].detach().cpu().numpy()
    if doppler_encoder is not None:
        velocities = np.array([
            doppler_encoder.shift_channels_to_velocity(s, speed_of_sound_m_s)
            for s in freq_shifts
        ])
        ax_vel.bar(velocities, vel_counts,
                   width=(velocities[1] - velocities[0]) * 0.8 if len(velocities) > 1 else 0.01,
                   color=velocity_color, alpha=0.7, label="Spike count")
        ax_vel.set_xlabel("Velocity (m/s)  [positive = approaching]")
    else:
        ax_vel.bar(freq_shifts, vel_counts, width=0.8,
                   color=velocity_color, alpha=0.7, label="Spike count")
        ax_vel.set_xlabel("Frequency shift (channels)")
    ax_vel.set_ylabel("Total spikes")
    ax_vel.set_title("Doppler Bank — Velocity Estimation")
    if true_velocity_m_s is not None:
        for v in (true_velocity_m_s if hasattr(true_velocity_m_s, "__iter__") else [true_velocity_m_s]):
            ax_vel.axvline(v, color=true_color, ls="--", lw=1.5,
                           label=f"True v = {v:.2f} m/s")
    if decoded_doppler_objects is not None:
        for obj in decoded_doppler_objects:
            ax_vel.axvline(obj.velocity_m_s, color=decoded_color, ls="-.", lw=1.5,
                           label=f"Dec. v = {obj.velocity_m_s:.2f} m/s")
    ax_vel.legend(loc="upper right", fontsize=8)
    ax_vel.grid(True, alpha=0.3)

    plt.suptitle("Range + Velocity Estimation", fontsize=14)
    plt.show()
