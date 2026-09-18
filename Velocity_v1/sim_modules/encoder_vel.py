import numpy as np
import torch
import torch.nn.functional as F


class TonotopicEncoder:
    """Encode FM sweeps into tonotopic spike trains."""

    def __init__(self, freq_channels=30, f_min=20, f_max=100, tuning_width=5.0):
        self.freq_channels = freq_channels
        self.f_min = f_min
        self.f_max = f_max
        self.tuning_width = tuning_width
        self.freq_bins = torch.linspace(f_max, f_min, freq_channels)

    @staticmethod
    def _fir_bandpass_kernel(low_hz, high_hz, sample_rate_hz, num_taps, device):
        low = float(low_hz) / float(sample_rate_hz)
        high = float(high_hz) / float(sample_rate_hz)
        n = torch.arange(num_taps, device=device, dtype=torch.float32)
        mid = (num_taps - 1) / 2.0
        t = n - mid

        h = 2.0 * high * torch.sinc(2.0 * high * t) - 2.0 * low * torch.sinc(2.0 * low * t)
        window = 0.54 - 0.46 * torch.cos(2.0 * np.pi * n / (num_taps - 1))
        h = h * window
        h = h / (h.abs().sum() + 1e-8)
        return h

    def _build_filterbank(self, sample_rate_hz, bandwidth_hz, num_taps, device):
        nyquist = sample_rate_hz * 0.5
        kernels = []
        for center_khz in self.freq_bins:
            center_hz = float(center_khz.item()) * 1000.0
            low_hz = max(200.0, center_hz - bandwidth_hz * 0.5)
            high_hz = min(nyquist - 200.0, center_hz + bandwidth_hz * 0.5)
            if high_hz <= low_hz:
                high_hz = min(nyquist - 200.0, low_hz + 1000.0)

            h = self._fir_bandpass_kernel(
                low_hz=low_hz,
                high_hz=high_hz,
                sample_rate_hz=sample_rate_hz,
                num_taps=num_taps,
                device=device,
            )
            kernels.append(h)

        return torch.stack(kernels, dim=0).unsqueeze(1)

    def compute_filterbank_response(
        self,
        waveform,
        sample_rate_hz=250_000,
        filter_taps=63,
        bandwidth_khz=8.0,
        device="cpu",
    ):
        """Return per-channel bandpass responses with shape [F, samples]."""
        waveform = waveform.to(device=device, dtype=torch.float32).flatten()
        x = waveform.view(1, 1, -1)
        filters = self._build_filterbank(
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=bandwidth_khz * 1000.0,
            num_taps=filter_taps,
            device=device,
        )
        return F.conv1d(x, filters, padding=filter_taps // 2).squeeze(0)

    def generate_fm_sweep_waveform(
        self,
        num_steps,
        start_time,
        duration,
        f_start=None,
        f_end=None,
        sample_rate_hz=250_000,
        step_duration_ms=0.1,
        amplitude=1.0,
        device="cpu",
    ):
        if f_start is None:
            f_start = self.f_max
        if f_end is None:
            f_end = self.f_min

        step_duration_s = step_duration_ms * 1e-3
        total_duration_s = num_steps * step_duration_s
        total_samples = max(1, int(round(total_duration_s * sample_rate_hz)))
        waveform = torch.zeros(total_samples, device=device, dtype=torch.float32)

        chirp_duration_s = max(step_duration_s, duration * step_duration_s)
        chirp_samples = max(2, int(round(chirp_duration_s * sample_rate_hz)))
        t = torch.arange(chirp_samples, device=device, dtype=torch.float32) / sample_rate_hz

        f0_hz = float(f_start) * 1000.0
        f1_hz = float(f_end) * 1000.0
        k = (f1_hz - f0_hz) / chirp_duration_s
        phase = 2.0 * np.pi * (f0_hz * t + 0.5 * k * t.pow(2))
        chirp = amplitude * torch.sin(phase)

        start_sample = int(round(start_time * step_duration_s * sample_rate_hz))
        end_sample = min(total_samples, start_sample + chirp_samples)
        if start_sample < total_samples:
            waveform[start_sample:end_sample] = chirp[: end_sample - start_sample]

        return waveform

    def generate_tone_waveform(
        self,
        num_steps,
        start_time,
        duration,
        freq_khz,
        sample_rate_hz=250_000,
        step_duration_ms=0.1,
        amplitude=1.0,
        device="cpu",
    ):
        """Generate a constant-frequency tone burst."""
        step_duration_s = step_duration_ms * 1e-3
        total_duration_s = num_steps * step_duration_s
        total_samples = max(1, int(round(total_duration_s * sample_rate_hz)))
        waveform = torch.zeros(total_samples, device=device, dtype=torch.float32)

        tone_duration_s = max(step_duration_s, duration * step_duration_s)
        tone_samples = max(2, int(round(tone_duration_s * sample_rate_hz)))
        t = torch.arange(tone_samples, device=device, dtype=torch.float32) / sample_rate_hz
        tone = amplitude * torch.sin(2.0 * np.pi * (float(freq_khz) * 1000.0) * t)

        start_sample = int(round(start_time * step_duration_s * sample_rate_hz))
        end_sample = min(total_samples, start_sample + tone_samples)
        if start_sample < total_samples:
            waveform[start_sample:end_sample] = tone[: end_sample - start_sample]

        return waveform

    def compute_step_energy_from_waveform(
        self,
        waveform,
        num_steps,
        sample_rate_hz=250_000,
        step_duration_ms=0.1,
        filter_taps=63,
        bandwidth_khz=8.0,
        smoothing_frac=0.8,
        device="cpu",
    ):
        """Convert an audio waveform into tonotopic spike probabilities."""
        filtered = self.compute_filterbank_response(
            waveform=waveform,
            sample_rate_hz=sample_rate_hz,
            filter_taps=filter_taps,
            bandwidth_khz=bandwidth_khz,
            device=device,
        )
        energy = filtered.pow(2)

        samples_per_step = max(1, int(round(sample_rate_hz * (step_duration_ms * 1e-3))))
        smooth_len = max(1, int(round(samples_per_step * smoothing_frac)))
        smooth_kernel = torch.ones(
            self.freq_channels, 1, smooth_len, device=device, dtype=torch.float32
        ) / float(smooth_len)
        # Causal smoothing: each time index uses only current and past samples.
        energy = F.pad(energy.unsqueeze(0), (smooth_len - 1, 0))
        energy = F.conv1d(
            energy, smooth_kernel, padding=0, groups=self.freq_channels
        ).squeeze(0)

        total_audio_samples = energy.shape[1]
        target_audio_samples = num_steps * samples_per_step
        if total_audio_samples < target_audio_samples:
            energy = torch.nn.functional.pad(energy, (0, target_audio_samples - total_audio_samples))
        elif total_audio_samples > target_audio_samples:
            energy = energy[:, :target_audio_samples]

        step_energy = energy.view(self.freq_channels, num_steps, samples_per_step).mean(dim=2).T
        return step_energy

    def compute_spike_probabilities_from_waveform(
        self,
        waveform,
        num_steps,
        max_spike_prob=0.9,
        sample_rate_hz=250_000,
        step_duration_ms=0.1,
        filter_taps=63,
        bandwidth_khz=8.0,
        smoothing_frac=0.8,
        device="cpu",
    ):
        """Convert an audio waveform into tonotopic spike probabilities."""
        step_energy = self.compute_step_energy_from_waveform(
            waveform=waveform,
            num_steps=num_steps,
            sample_rate_hz=sample_rate_hz,
            step_duration_ms=step_duration_ms,
            filter_taps=filter_taps,
            bandwidth_khz=bandwidth_khz,
            smoothing_frac=smoothing_frac,
            device=device,
        )
        step_energy = step_energy / (step_energy.max(dim=0, keepdim=True).values + 1e-8)
        return (step_energy * max_spike_prob).clamp(0.0, 1.0)

    def sample_spikes_from_probabilities(
        self,
        spike_probs,
        batch_size=1,
        device="cpu",
    ):
        """Sample one or more spike trains from cached spike probabilities."""
        spike_probs = spike_probs.to(device=device, dtype=torch.float32)
        if spike_probs.ndim != 2 or spike_probs.shape[1] != self.freq_channels:
            raise ValueError("Spike probabilities must have shape [time, freq_channels].")

        batch_size = max(1, int(batch_size))
        random_values = torch.rand(batch_size, spike_probs.shape[0], self.freq_channels, device=device)
        spikes = (random_values < spike_probs.unsqueeze(0)).float()
        return spikes.permute(1, 0, 2).contiguous()

    def encode_waveform_to_spikes(
        self,
        waveform,
        num_steps,
        max_spike_prob=0.9,
        sample_rate_hz=250_000,
        step_duration_ms=0.1,
        filter_taps=63,
        bandwidth_khz=8.0,
        smoothing_frac=0.8,
        device="cpu",
    ):
        """Convert an audio waveform into tonotopic spikes."""
        spike_probs = self.compute_spike_probabilities_from_waveform(
            waveform=waveform,
            num_steps=num_steps,
            max_spike_prob=max_spike_prob,
            sample_rate_hz=sample_rate_hz,
            step_duration_ms=step_duration_ms,
            filter_taps=filter_taps,
            bandwidth_khz=bandwidth_khz,
            smoothing_frac=smoothing_frac,
            device=device,
        )
        return self.sample_spikes_from_probabilities(
            spike_probs=spike_probs,
            batch_size=1,
            device=device,
        )

# ---------------------------------------------------------------------------
# High-resolution "auditory fovea" encoder for Doppler velocity estimation
# ---------------------------------------------------------------------------

class DopplerEncoder(TonotopicEncoder):
    """Narrowband, high-resolution tonotopic encoder for Doppler analysis.

    Real CF-FM bats (e.g. *Pteronotus parnellii*, *Rhinolophus ferrumequinum*)
    have an **auditory fovea** — an over-represented cortical region with
    extremely sharp frequency tuning (~100 Hz bandwidth) centred on the
    resting CF frequency.  This class models that specialisation.

    Typical usage::

        # Broadband encoder for range (FM component)
        range_enc = TonotopicEncoder(freq_channels=30, f_min=30, f_max=90)

        # Narrowband encoder for velocity (CF component)
        doppler_enc = DopplerEncoder(cf_freq_khz=60.0, half_range_khz=3.0,
                                     freq_channels=60)
        # → 60 channels spanning 57–63 kHz  →  100 Hz per channel

    The same waveform is fed into both encoders.  The broadband spikes drive
    the delay bank (range estimation); the narrowband spikes drive the
    Doppler bank (velocity estimation).

    Parameters
    ----------
    cf_freq_khz : float
        Centre frequency of the CF component in kHz.
    half_range_khz : float
        The encoder covers ``[cf - half_range, cf + half_range]``.
    freq_channels : int
        Number of frequency channels (higher → finer Doppler resolution).
    bandwidth_per_channel_khz : float
        Bandwidth of each individual bandpass filter in kHz.
        Defaults to twice the channel spacing for mild overlap.
    """

    def __init__(
        self,
        cf_freq_khz=60.0,
        half_range_khz=3.0,
        freq_channels=60,
        bandwidth_per_channel_khz=None,
        tuning_width=0.5,
    ):
        f_min = cf_freq_khz - half_range_khz
        f_max = cf_freq_khz + half_range_khz
        super().__init__(
            freq_channels=freq_channels,
            f_min=f_min,
            f_max=f_max,
            tuning_width=tuning_width,
        )
        self.cf_freq_khz = float(cf_freq_khz)
        self.half_range_khz = float(half_range_khz)

        # Channel spacing in kHz (used for shift↔velocity conversion).
        self.channel_spacing_khz = (2.0 * half_range_khz) / max(1, freq_channels - 1)

        # Default per-filter bandwidth: 2× channel spacing for mild overlap.
        if bandwidth_per_channel_khz is None:
            self.bandwidth_per_channel_khz = 2.0 * self.channel_spacing_khz
        else:
            self.bandwidth_per_channel_khz = float(bandwidth_per_channel_khz)

    # --- Doppler ↔ velocity helpers ------------------------------------------

    def velocity_to_shift_channels(self, velocity_m_s, speed_of_sound_m_s=343.0):
        """Convert radial velocity (m/s) → expected frequency shift (channels).

        Positive velocity (approaching) → positive Doppler shift (higher freq)
        → **negative** channel index shift (because channel 0 = highest freq).
        """
        delta_f_khz = self.cf_freq_khz * 2.0 * velocity_m_s / speed_of_sound_m_s
        return delta_f_khz / max(self.channel_spacing_khz, 1e-12)

    def shift_channels_to_velocity(self, shift_channels, speed_of_sound_m_s=343.0):
        """Convert frequency shift (channels) → radial velocity (m/s)."""
        delta_f_khz = shift_channels * self.channel_spacing_khz
        return delta_f_khz * speed_of_sound_m_s / (2.0 * max(self.cf_freq_khz, 1e-12))

    @staticmethod
    def _local_peak_position(row, radius=4):
        """Return a sub-channel peak estimate using a small local centroidx."""
        peak_idx = int(row.argmax())
        start = max(0, peak_idx - int(radius))
        stop = min(row.numel(), peak_idx + int(radius) + 1)
        indices = torch.arange(start, stop, device=row.device, dtype=row.dtype)
        weights = row[start:stop].clamp_min(0.0)
        denom = weights.sum().clamp_min(1e-12)
        return float((indices * weights).sum() / denom)

    def compute_spike_probabilities_from_waveform(
        self,
        waveform,
        num_steps,
        max_spike_prob=0.9,
        sample_rate_hz=250_000,
        step_duration_ms=0.1,
        filter_taps=127,
        bandwidth_khz=None,
        smoothing_frac=0.8,
        device="cpu",
    ):
        """Encode narrowband CF activity while preserving spectral shape per frame."""
        if bandwidth_khz is None:
            bandwidth_khz = self.bandwidth_per_channel_khz
        step_energy = self.compute_step_energy_from_waveform(
            waveform=waveform,
            num_steps=num_steps,
            sample_rate_hz=sample_rate_hz,
            step_duration_ms=step_duration_ms,
            filter_taps=filter_taps,
            bandwidth_khz=bandwidth_khz,
            smoothing_frac=smoothing_frac,
            device=device,
        )
        step_energy = step_energy / (step_energy.max(dim=1, keepdim=True).values + 1e-8)
        return (step_energy * max_spike_prob).clamp(0.0, 1.0)

    def estimate_shift_from_waveforms(
        self,
        pulse_waveform,
        echo_waveform,
        delay_steps,
        num_steps,
        sample_rate_hz=250_000,
        step_duration_ms=0.1,
        filter_taps=223,
        bandwidth_khz=None,
        smoothing_frac=0.8,
        activity_threshold_frac=0.15,
        peak_radius=4,
        device="cpu",
    ):
        """Estimate Doppler shift from aligned CF waveforms via peak tracking."""
        if bandwidth_khz is None:
            bandwidth_khz = self.bandwidth_per_channel_khz

        pulse_step_energy = self.compute_step_energy_from_waveform(
            waveform=pulse_waveform,
            num_steps=num_steps,
            sample_rate_hz=sample_rate_hz,
            step_duration_ms=step_duration_ms,
            filter_taps=filter_taps,
            bandwidth_khz=bandwidth_khz,
            smoothing_frac=smoothing_frac,
            device=device,
        )
        echo_step_energy = self.compute_step_energy_from_waveform(
            waveform=echo_waveform,
            num_steps=num_steps,
            sample_rate_hz=sample_rate_hz,
            step_duration_ms=step_duration_ms,
            filter_taps=filter_taps,
            bandwidth_khz=bandwidth_khz,
            smoothing_frac=smoothing_frac,
            device=device,
        )

        pulse_peak_threshold = float(pulse_step_energy.max().item()) * float(activity_threshold_frac)
        echo_peak_threshold = float(echo_step_energy.max().item()) * float(activity_threshold_frac)

        shift_estimates = []
        weights = []
        delay_steps = int(round(delay_steps))

        for t in range(int(num_steps)):
            pulse_t = t - delay_steps
            if pulse_t < 0 or pulse_t >= num_steps:
                continue

            pulse_row = pulse_step_energy[pulse_t]
            echo_row = echo_step_energy[t]
            pulse_peak = float(pulse_row.max().item())
            echo_peak = float(echo_row.max().item())
            if pulse_peak < pulse_peak_threshold or echo_peak < echo_peak_threshold:
                continue

            pulse_pos = self._local_peak_position(pulse_row, radius=peak_radius)
            echo_pos = self._local_peak_position(echo_row, radius=peak_radius)
            shift_estimates.append(pulse_pos - echo_pos)
            weights.append(pulse_peak * echo_peak)

        if not shift_estimates:
            return None

        return float(np.average(shift_estimates, weights=weights))

    def estimate_velocity_from_waveforms(
        self,
        pulse_waveform,
        echo_waveform,
        delay_steps,
        num_steps,
        sample_rate_hz=250_000,
        step_duration_ms=0.1,
        filter_taps=223,
        bandwidth_khz=None,
        smoothing_frac=0.8,
        activity_threshold_frac=0.15,
        peak_radius=4,
        speed_of_sound_m_s=343.0,
        device="cpu",
    ):
        """Convenience wrapper around ``estimate_shift_from_waveforms``."""
        shift_channels = self.estimate_shift_from_waveforms(
            pulse_waveform=pulse_waveform,
            echo_waveform=echo_waveform,
            delay_steps=delay_steps,
            num_steps=num_steps,
            sample_rate_hz=sample_rate_hz,
            step_duration_ms=step_duration_ms,
            filter_taps=filter_taps,
            bandwidth_khz=bandwidth_khz,
            smoothing_frac=smoothing_frac,
            activity_threshold_frac=activity_threshold_frac,
            peak_radius=peak_radius,
            device=device,
        )
        if shift_channels is None:
            return None
        return self.shift_channels_to_velocity(
            shift_channels, speed_of_sound_m_s=speed_of_sound_m_s
        )

    def encode_waveform_to_spikes(
        self,
        waveform,
        num_steps,
        max_spike_prob=0.9,
        sample_rate_hz=250_000,
        step_duration_ms=0.1,
        filter_taps=127,
        bandwidth_khz=None,
        smoothing_frac=0.8,
        device="cpu",
    ):
        """Encode using the narrow per-channel bandwidth by default.

        ``filter_taps`` defaults to 127 (longer kernel for sharper filters).
        ``bandwidth_khz`` defaults to ``self.bandwidth_per_channel_khz``.
        """
        if bandwidth_khz is None:
            bandwidth_khz = self.bandwidth_per_channel_khz
        return super().encode_waveform_to_spikes(
            waveform=waveform,
            num_steps=num_steps,
            max_spike_prob=max_spike_prob,
            sample_rate_hz=sample_rate_hz,
            step_duration_ms=step_duration_ms,
            filter_taps=filter_taps,
            bandwidth_khz=bandwidth_khz,
            smoothing_frac=smoothing_frac,
            device=device,
        )
