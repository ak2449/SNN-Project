import torch
import torch.nn.functional as F

"""Simpler method of echo generation - Takes pulse spike train and performs transformations on it"""
def generate_fm_echo(
    pulse_spikes,
    delay,
    attenuation=0.6,
    doppler_shift_khz=0,
    freq_bins=None,
    spectral_notches=None,
):
    """Generate an echo with delay, attenuation, optional doppler, and spectral notches."""
    t_steps, _, freq_channels = pulse_spikes.shape
    echo_spikes = torch.zeros_like(pulse_spikes)

    if delay > 0 and delay < t_steps:
        echo_spikes[delay:, :, :] = pulse_spikes[:-delay, :, :].clone()
    elif delay == 0:
        echo_spikes = pulse_spikes.clone()

    echo_spikes = (torch.rand_like(echo_spikes) < echo_spikes * attenuation).float()

    if doppler_shift_khz != 0 and freq_bins is not None:
        freq_spacing = (freq_bins[0] - freq_bins[-1]) / (len(freq_bins) - 1)
        shift_channels = int(doppler_shift_khz / freq_spacing)
        if shift_channels != 0:
            echo_spikes = torch.roll(echo_spikes, shifts=shift_channels, dims=2)
            if shift_channels > 0:
                echo_spikes[:, :, :shift_channels] = 0
            else:
                echo_spikes[:, :, shift_channels:] = 0

    if spectral_notches is not None:
        for notch_idx in spectral_notches:
            if 0 <= notch_idx < freq_channels:
                notch_width = 2
                start_idx = max(0, notch_idx - notch_width // 2)
                end_idx = min(freq_channels, notch_idx + notch_width // 2 + 1)
                echo_spikes[:, :, start_idx:end_idx] *= 0.2

    return echo_spikes


def _apply_air_absorption(waveform, sample_rate_hz, path_length_m, air_absorption_db_per_m_100khz):
    """Approximate frequency-dependent air attenuation in the ultrasonic regime."""
    if path_length_m <= 0 or air_absorption_db_per_m_100khz <= 0:
        return waveform

    n = waveform.numel()
    spectrum = torch.fft.rfft(waveform)
    freqs_hz = torch.fft.rfftfreq(n, d=1.0 / float(sample_rate_hz)).to(waveform.device)

    attenuation_db = air_absorption_db_per_m_100khz * path_length_m * (freqs_hz / 100000.0)
    attenuation_amp = torch.pow(10.0, -attenuation_db / 20.0)

    spectrum = spectrum * attenuation_amp
    return torch.fft.irfft(spectrum, n=n)


def _apply_surface_reflection_coloration(waveform, micro_delay_samples, roughness):
    """Simple two-path reflection model that creates notch-like spectral ripples."""
    if micro_delay_samples <= 0 or roughness <= 0:
        return waveform

    kernel_len = int(micro_delay_samples) + 1
    kernel = torch.zeros(1, 1, kernel_len, device=waveform.device, dtype=waveform.dtype)
    kernel[0, 0, 0] = 1.0
    kernel[0, 0, -1] = -float(roughness)

    x = waveform.view(1, 1, -1)
    y = F.conv1d(x, kernel, padding=kernel_len - 1).squeeze(0).squeeze(0)
    return y[: waveform.numel()]


def _generate_multipath_clutter_waveform(
    pulse_waveform,
    total_samples,
    sample_rate_hz,
    speed_of_sound_m_s,
    air_absorption_db_per_m_100khz,
    num_paths,
    min_distance_m,
    max_distance_m,
    min_reflectivity,
    max_reflectivity,
    min_roughness,
    max_roughness,
    min_texture_scale_m,
    max_texture_scale_m,
):
    """Synthesize clutter as many weak delayed reflection paths."""
    clutter = torch.zeros_like(pulse_waveform)
    if num_paths <= 0:
        return clutter

    max_distance_m = max(min_distance_m, max_distance_m)
    for _ in range(int(num_paths)):
        u = torch.rand(1, device=pulse_waveform.device).item()
        distance_m = min_distance_m + (max_distance_m - min_distance_m) * u
        round_trip_m = 2.0 * max(0.0, float(distance_m))

        delay_s = round_trip_m / max(1e-6, float(speed_of_sound_m_s))
        delay_samples = int(round(delay_s * sample_rate_hz))
        if delay_samples >= total_samples:
            continue

        reflectivity = min_reflectivity + (max_reflectivity - min_reflectivity) * torch.rand(
            1, device=pulse_waveform.device
        ).item()
        roughness = min_roughness + (max_roughness - min_roughness) * torch.rand(
            1, device=pulse_waveform.device
        ).item()
        texture_scale_m = min_texture_scale_m + (max_texture_scale_m - min_texture_scale_m) * torch.rand(
            1, device=pulse_waveform.device
        ).item()

        geometric_atten = 1.0 / ((1.0 + round_trip_m) ** 2)
        path = pulse_waveform * float(reflectivity) * geometric_atten
        path = _apply_air_absorption(
            path,
            sample_rate_hz=sample_rate_hz,
            path_length_m=round_trip_m,
            air_absorption_db_per_m_100khz=air_absorption_db_per_m_100khz,
        )

        micro_delay_samples = int(round(texture_scale_m / max(1e-6, speed_of_sound_m_s) * sample_rate_hz))
        path = _apply_surface_reflection_coloration(
            path,
            micro_delay_samples=micro_delay_samples,
            roughness=float(roughness),
        )

        if torch.rand(1, device=pulse_waveform.device).item() < 0.5:
            path = -path

        src_len = total_samples - delay_samples
        clutter[delay_samples:] += path[:src_len]

    return clutter


def generate_fm_echo_from_waveform(
    pulse_waveform,
    encoder,
    num_steps,
    distance_m,
    reflectivity=0.7,
    max_spike_prob=0.9,
    sample_rate_hz=250_000,
    step_duration_ms=0.1,
    speed_of_sound_m_s=343.0,
    air_absorption_db_per_m_100khz=1.2,
    surface_roughness=0.25,
    surface_texture_scale_m=0.002,
    filter_taps=63,
    bandwidth_khz=8.0,
    smoothing_frac=0.8,
    clutter_fraction=0.8,
    num_clutter_paths=40,
    clutter_min_distance_m=0.05,
    clutter_max_distance_m=None,
    clutter_reflectivity_range=(0.02, 0.5),
    clutter_roughness_range=(0.1, 0.6),
    clutter_texture_scale_range_m=(0.01, 0.004),
    device="cpu",
):
    """Generate a biologically-inspired echo by propagating a pulse waveform through space."""
    pulse_waveform = pulse_waveform.to(device=device, dtype=torch.float32).flatten()

    total_samples = max(1, int(round(num_steps * (step_duration_ms * 1e-3) * sample_rate_hz)))
    if pulse_waveform.numel() < total_samples:
        pulse_waveform = F.pad(pulse_waveform, (0, total_samples - pulse_waveform.numel()))
    elif pulse_waveform.numel() > total_samples:
        pulse_waveform = pulse_waveform[:total_samples]

    round_trip_m = max(0.0, 2.0 * float(distance_m))
    delay_s = round_trip_m / max(1e-6, float(speed_of_sound_m_s))
    delay_samples = int(round(delay_s * sample_rate_hz))
    delay_steps = int(round(delay_s / (step_duration_ms * 1e-3)))

    # Distance attenuation with reflectivity scaling.
    geometric_atten = 1.0 / ((1.0 + round_trip_m) ** 2)
    amplitude_scale = float(reflectivity) * geometric_atten

    propagated = pulse_waveform * amplitude_scale
    propagated = _apply_air_absorption(
        propagated,
        sample_rate_hz=sample_rate_hz,
        path_length_m=round_trip_m,
        air_absorption_db_per_m_100khz=air_absorption_db_per_m_100khz,
    )

    micro_delay_samples = int(round(surface_texture_scale_m / max(1e-6, speed_of_sound_m_s) * sample_rate_hz))
    propagated = _apply_surface_reflection_coloration(
        propagated,
        micro_delay_samples=micro_delay_samples,
        roughness=float(surface_roughness),
    )

    echo_waveform = torch.zeros_like(pulse_waveform)
    if delay_samples < total_samples:
        src_len = total_samples - delay_samples
        echo_waveform[delay_samples:] = propagated[:src_len]

    if clutter_fraction > 0 and num_clutter_paths > 0:
        max_observable_distance_m = 0.5 * float(speed_of_sound_m_s) * (
            total_samples / float(sample_rate_hz)
        )
        clutter_max_distance_m = (
            max_observable_distance_m if clutter_max_distance_m is None else float(clutter_max_distance_m)
        )
        clutter_max_distance_m = max(float(clutter_min_distance_m), clutter_max_distance_m)

        min_reflectivity, max_reflectivity = clutter_reflectivity_range
        min_roughness, max_roughness = clutter_roughness_range
        min_texture_scale_m, max_texture_scale_m = clutter_texture_scale_range_m

        clutter_waveform = _generate_multipath_clutter_waveform(
            pulse_waveform=pulse_waveform,
            total_samples=total_samples,
            sample_rate_hz=sample_rate_hz,
            speed_of_sound_m_s=speed_of_sound_m_s,
            air_absorption_db_per_m_100khz=air_absorption_db_per_m_100khz,
            num_paths=int(num_clutter_paths),
            min_distance_m=float(clutter_min_distance_m),
            max_distance_m=float(clutter_max_distance_m),
            min_reflectivity=float(min_reflectivity),
            max_reflectivity=float(max_reflectivity),
            min_roughness=float(min_roughness),
            max_roughness=float(max_roughness),
            min_texture_scale_m=float(min_texture_scale_m),
            max_texture_scale_m=float(max_texture_scale_m),
        )

        clutter_peak = clutter_waveform.abs().max().clamp(min=1e-6)
        main_echo_peak = echo_waveform.abs().max().clamp(min=1e-6)
        clutter_target_peak = main_echo_peak * float(clutter_fraction)
        echo_waveform = echo_waveform + clutter_waveform * (clutter_target_peak / clutter_peak)

    echo_spikes = encoder.encode_waveform_to_spikes(
        waveform=echo_waveform,
        num_steps=num_steps,
        max_spike_prob=max_spike_prob,
        sample_rate_hz=sample_rate_hz,
        step_duration_ms=step_duration_ms,
        filter_taps=filter_taps,
        bandwidth_khz=bandwidth_khz,
        smoothing_frac=smoothing_frac,
        device=device,
    )

    return echo_spikes, echo_waveform, delay_steps
