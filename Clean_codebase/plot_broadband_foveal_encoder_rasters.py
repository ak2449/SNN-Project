import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sim_modules import TonotopicEncoder


BROADBAND_COLOR = "#3B6CA8"
FOVEAL_COLOR = "#D86B6B"


CONFIG = {
    "seed": 11,
    "simulation": {
        "device": "cpu",
        "num_steps": 130,
        "sample_rate_hz": 1_000_000,
        "step_duration_ms": 0.1,
        "speed_of_sound_m_s": 343.0,
        "snr_db": 0,
        "clutter_fraction": 0,
    },
    "broadband_encoder": {
        "freq_channels": 30,
        "f_min": 30,
        "f_max": 90,
    },
    "foveal_encoder": {
        "cf_freq_khz": 60.0,
        "half_range_khz": 3.0,
        "freq_channels": 60,
    },
    "call": {
        "start_time": 20,
        "cf_duration": 70,
        "fm_start_time": 65,
        "fm_duration": 35,
        "fm_start_khz": 90,
        "fm_end_khz": 30,
        "cf_amplitude": 1.0,
        "fm_amplitude": 0.85,
        "target_velocity_m_s": 1.2,
    },
    "encoding": {
        "max_spike_prob": 0.9,
        "broadband_filter_taps": 63,
        "broadband_bandwidth_khz": 8.0,
        "foveal_filter_taps": 223,
        "smoothing_frac": 0.8,
        "foveal_sharpen_power": 8.0,
    },
}


def load_doppler_encoder():
    """Load the velocity pathway's DopplerEncoder without shadowing sim_modules."""
    repo_root = Path(__file__).resolve().parents[1]
    encoder_path = repo_root / "Velocity_v1" / "sim_modules" / "encoder_vel.py"
    spec = importlib.util.spec_from_file_location("velocity_encoder_vel", encoder_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load DopplerEncoder from {encoder_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DopplerEncoder


DopplerEncoder = load_doppler_encoder()


def combine_call_components(*waveforms):
    """Combine CF and FM components into one emitted call waveform."""
    combined = torch.stack(waveforms, dim=0).sum(dim=0)
    peak = combined.abs().max().clamp_min(1e-8)
    return combined / peak if peak.item() > 1.0 else combined


def sharpen_spike_probabilities(spike_probs, max_spike_prob, power):
    """Sharpen foveal activity so the narrowband place code is visually crisp."""
    if power <= 1.0:
        return spike_probs

    normalized = (spike_probs / max(float(max_spike_prob), 1e-8)).clamp(0.0, 1.0)
    return (normalized.pow(float(power)) * float(max_spike_prob)).clamp(0.0, 1.0)


def frequency_to_channel(freq_bins, freq_khz):
    """Return the fractional channel position for a frequency in kHz."""
    bins = freq_bins.detach().cpu().numpy()
    channels = np.arange(bins.size, dtype=float)
    return float(np.interp(float(freq_khz), bins[::-1], channels[::-1]))


def set_frequency_ticks(ax, freq_bins, tick_freqs):
    tick_positions = [frequency_to_channel(freq_bins, freq) for freq in tick_freqs]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f"{freq:g}" for freq in tick_freqs])
    ax.set_ylabel("Centre frequency (kHz)")


def plot_raster(ax, spikes, color, title, freq_bins, num_steps, marker_size):
    time_indices, _, freq_indices = torch.where(spikes > 0)
    ax.scatter(
        time_indices.cpu().numpy(),
        freq_indices.cpu().numpy(),
        s=marker_size,
        c=color,
        alpha=0.72,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlim(0, num_steps)
    ax.set_ylim(spikes.shape[2] - 1, 0)
    ax.set_xlabel("Time step")
    ax.grid(True, alpha=0.24)

    top_freq = float(freq_bins[0].item())
    middle_freq = 0.5 * (float(freq_bins[0].item()) + float(freq_bins[-1].item()))
    bottom_freq = float(freq_bins[-1].item())
    set_frequency_ticks(ax, freq_bins, [top_freq, middle_freq, bottom_freq])


def build_call_waveform(broadband_encoder, foveal_encoder, sim_cfg, call_cfg):
    velocity_m_s = float(call_cfg["target_velocity_m_s"])
    speed_of_sound_m_s = float(sim_cfg["speed_of_sound_m_s"])
    cf_shift_khz = foveal_encoder.cf_freq_khz * 2.0 * velocity_m_s / speed_of_sound_m_s
    shifted_cf_khz = foveal_encoder.cf_freq_khz + cf_shift_khz

    cf_waveform = foveal_encoder.generate_tone_waveform(
        num_steps=sim_cfg["num_steps"],
        start_time=call_cfg["start_time"],
        duration=call_cfg["cf_duration"],
        freq_khz=shifted_cf_khz,
        sample_rate_hz=sim_cfg["sample_rate_hz"],
        step_duration_ms=sim_cfg["step_duration_ms"],
        amplitude=call_cfg["cf_amplitude"],
        device=sim_cfg["device"],
    )
    fm_waveform = broadband_encoder.generate_fm_sweep_waveform(
        num_steps=sim_cfg["num_steps"],
        start_time=call_cfg["fm_start_time"],
        duration=call_cfg["fm_duration"],
        f_start=call_cfg["fm_start_khz"],
        f_end=call_cfg["fm_end_khz"],
        sample_rate_hz=sim_cfg["sample_rate_hz"],
        step_duration_ms=sim_cfg["step_duration_ms"],
        amplitude=call_cfg["fm_amplitude"],
        device=sim_cfg["device"],
    )
    return combine_call_components(cf_waveform, fm_waveform), shifted_cf_khz


def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    sim_cfg = CONFIG["simulation"]
    encoding_cfg = CONFIG["encoding"]

    broadband_encoder = TonotopicEncoder(**CONFIG["broadband_encoder"])
    foveal_encoder = DopplerEncoder(**CONFIG["foveal_encoder"])
    call_waveform, shifted_cf_khz = build_call_waveform(
        broadband_encoder=broadband_encoder,
        foveal_encoder=foveal_encoder,
        sim_cfg=sim_cfg,
        call_cfg=CONFIG["call"],
    )

    broadband_probs = broadband_encoder.compute_spike_probabilities_from_waveform(
        waveform=call_waveform,
        num_steps=sim_cfg["num_steps"],
        max_spike_prob=encoding_cfg["max_spike_prob"],
        sample_rate_hz=sim_cfg["sample_rate_hz"],
        step_duration_ms=sim_cfg["step_duration_ms"],
        filter_taps=encoding_cfg["broadband_filter_taps"],
        bandwidth_khz=encoding_cfg["broadband_bandwidth_khz"],
        smoothing_frac=encoding_cfg["smoothing_frac"],
        device=sim_cfg["device"],
    )
    foveal_probs = foveal_encoder.compute_spike_probabilities_from_waveform(
        waveform=call_waveform,
        num_steps=sim_cfg["num_steps"],
        max_spike_prob=encoding_cfg["max_spike_prob"],
        sample_rate_hz=sim_cfg["sample_rate_hz"],
        step_duration_ms=sim_cfg["step_duration_ms"],
        filter_taps=encoding_cfg["foveal_filter_taps"],
        smoothing_frac=encoding_cfg["smoothing_frac"],
        device=sim_cfg["device"],
    )
    foveal_probs = sharpen_spike_probabilities(
        foveal_probs,
        max_spike_prob=encoding_cfg["max_spike_prob"],
        power=encoding_cfg["foveal_sharpen_power"],
    )

    broadband_spikes = broadband_encoder.sample_spikes_from_probabilities(
        spike_probs=broadband_probs,
        batch_size=1,
        device=sim_cfg["device"],
    )
    foveal_spikes = foveal_encoder.sample_spikes_from_probabilities(
        spike_probs=foveal_probs,
        batch_size=1,
        device=sim_cfg["device"],
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True, sharex=True)
    plot_raster(
        axes[0],
        broadband_spikes,
        BROADBAND_COLOR,
        "Broadband TonotopicEncoder raster (30 channels, 30-90 kHz)",
        broadband_encoder.freq_bins,
        sim_cfg["num_steps"],
        marker_size=9,
    )
    plot_raster(
        axes[1],
        foveal_spikes,
        FOVEAL_COLOR,
        "Foveal DopplerEncoder raster (60 channels, 57-63 kHz)",
        foveal_encoder.freq_bins,
        sim_cfg["num_steps"],
        marker_size=7,
    )

    for ax, encoder in zip(axes, [broadband_encoder, foveal_encoder]):
        cf_channel = frequency_to_channel(encoder.freq_bins, shifted_cf_khz)
        ax.axhline(cf_channel, color="0.25", linestyle=":", linewidth=1.0, alpha=0.65)

    axes[0].set_xlabel("")
    fig.suptitle("Figure 6: Same CF-FM call -> two parallel encodings", fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()
