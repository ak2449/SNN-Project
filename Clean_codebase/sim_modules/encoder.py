import numpy as np
import torch


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
        window = torch.hann_window(chirp_samples, periodic=False, device=device, dtype=torch.float32)
        chirp = amplitude * torch.sin(phase) * window

        start_sample = int(round(start_time * step_duration_s * sample_rate_hz))
        end_sample = min(total_samples, start_sample + chirp_samples)
        if start_sample < total_samples:
            waveform[start_sample:end_sample] = chirp[: end_sample - start_sample]

        return waveform

    def generate_fm_sweep_from_waveform(
        self,
        num_steps,
        start_time,
        duration,
        f_start=None,
        f_end=None,
        max_spike_prob=0.9,
        sample_rate_hz=250_000,
        step_duration_ms=0.1,
        filter_taps=63,
        bandwidth_khz=8.0,
        smoothing_frac=0.8,
        device="cpu",
    ):
        waveform = self.generate_fm_sweep_waveform(
            num_steps=num_steps,
            start_time=start_time,
            duration=duration,
            f_start=f_start,
            f_end=f_end,
            sample_rate_hz=sample_rate_hz,
            step_duration_ms=step_duration_ms,
            device=device,
        )
        return self.encode_waveform_to_spikes(
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
        waveform = waveform.to(device=device, dtype=torch.float32).flatten()
        x = waveform.view(1, 1, -1)
        filters = self._build_filterbank(
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=bandwidth_khz * 1000.0,
            num_taps=filter_taps,
            device=device,
        )

        filtered = torch.nn.functional.conv1d(x, filters, padding=filter_taps // 2).squeeze(0)
        energy = filtered.pow(2)

        samples_per_step = max(1, int(round(sample_rate_hz * (step_duration_ms * 1e-3))))
        smooth_len = max(1, int(round(samples_per_step * smoothing_frac)))
        smooth_kernel = torch.ones(
            self.freq_channels, 1, smooth_len, device=device, dtype=torch.float32
        ) / float(smooth_len)
        energy = torch.nn.functional.conv1d(
            energy.unsqueeze(0), smooth_kernel, padding=smooth_len // 2, groups=self.freq_channels
        ).squeeze(0)

        total_audio_samples = energy.shape[1]
        target_audio_samples = num_steps * samples_per_step
        if total_audio_samples < target_audio_samples:
            energy = torch.nn.functional.pad(energy, (0, target_audio_samples - total_audio_samples))
        elif total_audio_samples > target_audio_samples:
            energy = energy[:, :target_audio_samples]

        step_energy = energy.view(self.freq_channels, num_steps, samples_per_step).mean(dim=2).T
        step_energy = step_energy / (step_energy.max(dim=0, keepdim=True).values + 1e-8)
        spike_probs = (step_energy * max_spike_prob).clamp(0.0, 1.0)
        spikes = (torch.rand(num_steps, self.freq_channels, device=device) < spike_probs).float()
        return spikes.unsqueeze(1)
    
    '''Old pulse spike train generataion'''
    def generate_fm_sweep(
        self,
        num_steps,
        start_time,
        duration,
        f_start=None,
        f_end=None,
        max_spike_prob=0.9,
        device="cpu",
    ):
        if f_start is None:
            f_start = self.f_max
        if f_end is None:
            f_end = self.f_min

        spikes = torch.zeros(num_steps, 1, self.freq_channels, device=device)
        for t in range(duration):
            progress = t / duration
            current_freq = f_start - (f_start - f_end) * progress
            freq_distance = (self.freq_bins.to(device) - current_freq) ** 2
            tuning_response = torch.exp(-freq_distance / (2 * self.tuning_width**2))
            spike_probs = tuning_response * max_spike_prob
            spikes[start_time + t, 0, :] = (
                torch.rand(self.freq_channels, device=device) < spike_probs
            ).float()

        return spikes
