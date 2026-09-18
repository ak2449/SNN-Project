import numpy as np
import torch

from sim_modules import (
    DecodedObject,
    FMSweepBankNetwork,
    TonotopicEncoder,
    DopplerEncoder,
    DopplerDecodedObject,
    FovealDopplerBankNetwork,
    generate_fm_echo_from_waveform,
    plot_scenario_results,
)
from sim_modules.visualization_vel import (
    plot_doppler_results,
    plot_doppler_scenario_results,
    plot_range_and_velocity,
)


def run_fm_sweep_demo():
    """Run a modular FM sweep echolocation demo with range AND velocity."""
    # ---------------------------------------------------------------------
    # Tunable parameters
    # ---------------------------------------------------------------------
    sim_cfg = {
        "device": "cpu",
        "num_steps": 250,
        "sample_rate_hz": 1_000_000,
        "step_duration_ms": 0.1,
        "seed": np.random.randint(0, 1000000),
    }

    # --- Broadband encoder (range / FM component) ---
    encoder_cfg = {
        "freq_channels": 30,
        "f_min": 30,
        "f_max": 90,
    } 

    # --- Narrowband "auditory fovea" encoder (velocity / CF component) ---
    #     60 channels over ±3 kHz around 60 kHz CF  →  100 Hz/channel
    doppler_encoder_cfg = {
        "cf_freq_khz": 60.0,
        "half_range_khz": 3.0,
        "freq_channels": 60,
    }

    network_cfg = {
        "delays_start": 1,
        "delays_stop": 200,
        "weight": 0.1,
        "beta": 0.1,
        "threshold": 0.9,
        "refractory_steps": 2,
        "lateral_inhibition_strength": 3,
        "lateral_inhibition_radius": 8,
        "inhibition_decay": 0.1,
        "decode_radius_channels": 3,
    }
    # --- Doppler bank configuration ---
    doppler_net_cfg = {
        "shift_range": 25,          # neurons tuned from -25 to +25 CF channels
        "input_batch_size": 64,     # independent auditory spike samples for population decoding
        "spectral_sharpen_power": 12.0,
        "decode_peak_fraction": 0.75,
        "tuning_width_channels": 2.0,
        "gate_threshold": 1,
        "decode_radius_channels": 3,
        "weight": 0.15,
        "beta": 0.15,
        "threshold": 0.8,
        "refractory_steps": 2,
        "lateral_inhibition_strength": 2.0,
        "lateral_inhibition_radius": 6,
        "inhibition_decay": 0.1,
    }

    # A single CF-FM call: a constant-frequency component followed by a
    # terminal FM sweep. Both range and velocity pathways read this waveform.
    cf_pulse_cfg = {
        "pulse_start_time": 5,
        "duration": 8,
        "tone_freq_khz": doppler_encoder_cfg["cf_freq_khz"],
        "amplitude": 1.0,
    }
    pulse_cfg = {
        "pulse_start_time": 13,
        "duration": 4,
        "f_start": 90,
        "f_end": 30,
        "amplitude": 0.9,
    }
    velocity_decode_cfg = {
        "filter_taps": 223,
        "smoothing_frac": 0.8,
    }

    pulse_encoding_cfg = {
        "max_spike_prob": 0.9,
        "filter_taps": 63,
        "bandwidth_khz": 8.0,
        "smoothing_frac": 0.8,
    }
    echo_default_cfg = {
        "max_spike_prob": 0.9,
        "speed_of_sound_m_s": 343.0,
        "air_absorption_db_per_m_100khz": 0.05,
        "surface_roughness": 0.25,
        "surface_texture_scale_m": 0.002,
        "filter_taps": 63,
        "bandwidth_khz": 8.0,
        "smoothing_frac": 0.8,
        "snr_db": None,
        "clutter_fraction": 0,
        "num_clutter_paths": 30,
        "clutter_min_distance_m": 0.05,
        "clutter_max_distance_m": 3.4,
        "clutter_reflectivity_range": (0.2, 0.9),
        "clutter_roughness_range": (0.1, 0.3),
        "clutter_texture_scale_range_m": (0.001, 0.004),
    }
    plot_cfg = {
        "pulse_color": "#3B6CA8",
        "echo_color": "#D86B6B",
        "range_color": "#3B6CA8",
        "velocity_color": "#D86B6B",
        "doppler_bar_color": "black",
        "doppler_bar_alpha": 0.2,
        "true_color": "#D86B6B",
        "decoded_color": "#3B6CA8",
    }

    # --- Scene: one object with known range and velocity ---
    object_cfgs = [
        {
            "distance_m": 1.15,
            "reflectivity": 0.9,
            "surface_roughness": 0.15,
            "velocity_m_s": 3.0,      # approaching at 3 m/s
        },
    ]

    detect_cfg = {
        "min_spikes": 2,
        "min_spikes_pop": 1,
    }

    device = sim_cfg["device"]
    num_steps = sim_cfg["num_steps"]
    sample_rate_hz = sim_cfg["sample_rate_hz"]
    step_duration_ms = sim_cfg["step_duration_ms"]
    pulse_start_time = min(cf_pulse_cfg["pulse_start_time"], pulse_cfg["pulse_start_time"])
    speed_of_sound_m_s = echo_default_cfg["speed_of_sound_m_s"]
    seed = sim_cfg["seed"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    def sharpen_doppler_probabilities(spike_probs):
        """Sharpen broad CF filter responses before stochastic spike sampling."""
        power = float(doppler_net_cfg["spectral_sharpen_power"])
        if power <= 1.0:
            return spike_probs

        max_spike_prob = max(float(pulse_encoding_cfg["max_spike_prob"]), 1e-8)
        normalized = (spike_probs / max_spike_prob).clamp(0.0, 1.0)
        return (normalized.pow(power) * max_spike_prob).clamp(0.0, 1.0)

    def combine_call_components(*waveforms):
        """Combine CF and FM components into one emitted call waveform."""
        combined = torch.stack(waveforms, dim=0).sum(dim=0)
        peak = combined.abs().max().clamp_min(1e-8)
        return combined / peak if peak.item() > 1.0 else combined

    # =====================================================================
    # Build encoders
    # =====================================================================
    encoder = TonotopicEncoder(**encoder_cfg)
    doppler_encoder = DopplerEncoder(**doppler_encoder_cfg)

    # =====================================================================
    # Build delay bank (range)
    # =====================================================================
    candidate_delays = np.arange(network_cfg["delays_start"], network_cfg["delays_stop"])
    net = FMSweepBankNetwork(
        delays=candidate_delays,
        freq_channels=encoder_cfg["freq_channels"],
        weight=network_cfg["weight"],
        beta=network_cfg["beta"],
        threshold=network_cfg["threshold"],
        refractory_steps=network_cfg["refractory_steps"],
        lateral_inhibition_strength=network_cfg["lateral_inhibition_strength"],
        lateral_inhibition_radius=network_cfg["lateral_inhibition_radius"],
        inhibition_decay=network_cfg["inhibition_decay"],
        decode_radius_channels=network_cfg["decode_radius_channels"],
    ).to(device)

    # =====================================================================
    # Prepare Doppler bank candidates (velocity)
    # =====================================================================
    shift_range = doppler_net_cfg["shift_range"]
    candidate_shifts = np.arange(-shift_range, shift_range + 1)

    def build_doppler_net(delay_steps):
        return FovealDopplerBankNetwork(
            freq_shifts=candidate_shifts,
            delay=delay_steps,
            freq_channels=doppler_encoder_cfg["freq_channels"],
            weight=doppler_net_cfg["weight"],
            beta=doppler_net_cfg["beta"],
            threshold=doppler_net_cfg["threshold"],
            refractory_steps=doppler_net_cfg["refractory_steps"],
            lateral_inhibition_strength=doppler_net_cfg["lateral_inhibition_strength"],
            lateral_inhibition_radius=doppler_net_cfg["lateral_inhibition_radius"],
            inhibition_decay=doppler_net_cfg["inhibition_decay"],
            tuning_width_channels=doppler_net_cfg["tuning_width_channels"],
            gate_threshold=doppler_net_cfg["gate_threshold"],
            decode_radius_channels=doppler_net_cfg["decode_radius_channels"],
        ).to(device)

    # =====================================================================
    # Generate pulse waveform
    # =====================================================================
    print("=" * 70)
    print("CF-FM BAT ECHOLOCATION — RANGE + VELOCITY ESTIMATION")
    print("=" * 70)

    pulse_waveform_cf = doppler_encoder.generate_tone_waveform(
        num_steps=num_steps,
        start_time=cf_pulse_cfg["pulse_start_time"],
        duration=cf_pulse_cfg["duration"],
        freq_khz=cf_pulse_cfg["tone_freq_khz"],
        sample_rate_hz=sample_rate_hz,
        step_duration_ms=step_duration_ms,
        amplitude=cf_pulse_cfg["amplitude"],
        device=device,
    )
    pulse_waveform_fm = encoder.generate_fm_sweep_waveform(
        num_steps=num_steps,
        start_time=pulse_cfg["pulse_start_time"],
        duration=pulse_cfg["duration"],
        f_start=pulse_cfg["f_start"],
        f_end=pulse_cfg["f_end"],
        sample_rate_hz=sample_rate_hz,
        step_duration_ms=step_duration_ms,
        amplitude=pulse_cfg["amplitude"],
        device=device,
    )
    pulse_waveform = combine_call_components(pulse_waveform_cf, pulse_waveform_fm)

    # Broadband spike encoding of the combined CF-FM pulse (for range bank).
    pulse_broadband = encoder.encode_waveform_to_spikes(
        waveform=pulse_waveform,
        num_steps=num_steps,
        max_spike_prob=pulse_encoding_cfg["max_spike_prob"],
        sample_rate_hz=sample_rate_hz,
        step_duration_ms=step_duration_ms,
        filter_taps=pulse_encoding_cfg["filter_taps"],
        bandwidth_khz=pulse_encoding_cfg["bandwidth_khz"],
        smoothing_frac=pulse_encoding_cfg["smoothing_frac"],
        device=device,
    )

    # Narrowband spike encoding of the same combined CF-FM pulse (for Doppler bank).
    pulse_doppler_probs = doppler_encoder.compute_spike_probabilities_from_waveform(
        waveform=pulse_waveform,
        num_steps=num_steps,
        max_spike_prob=pulse_encoding_cfg["max_spike_prob"],
        sample_rate_hz=sample_rate_hz,
        step_duration_ms=step_duration_ms,
        filter_taps=velocity_decode_cfg["filter_taps"],
        smoothing_frac=velocity_decode_cfg["smoothing_frac"],
        device=device,
    )
    pulse_doppler = doppler_encoder.sample_spikes_from_probabilities(
        spike_probs=sharpen_doppler_probabilities(pulse_doppler_probs),
        batch_size=doppler_net_cfg["input_batch_size"],
        device=device,
    )

    # =====================================================================
    # Generate echoes (with Doppler!)
    # =====================================================================
    torch.manual_seed(seed)
    np.random.seed(seed)

    object_echoes_broad = []
    object_echoes_narrow = []
    true_delays = []
    true_velocities = []

    for obj_cfg in object_cfgs:
        obj_echo_cfg = dict(echo_default_cfg)
        obj_echo_cfg.update(obj_cfg)
        velocity = obj_echo_cfg.pop("velocity_m_s", 0.0)
        true_velocities.append(velocity)

        # --- Combined CF-FM echo. The range and velocity pathways both read this echo. ---
        echo_broad_i, echo_waveform_i, true_delay_i = generate_fm_echo_from_waveform(
            pulse_waveform=pulse_waveform,
            encoder=encoder,
            num_steps=num_steps,
            distance_m=obj_echo_cfg["distance_m"],
            reflectivity=obj_echo_cfg["reflectivity"],
            velocity_m_s=velocity,
            max_spike_prob=obj_echo_cfg["max_spike_prob"],
            sample_rate_hz=sample_rate_hz,
            step_duration_ms=step_duration_ms,
            speed_of_sound_m_s=obj_echo_cfg["speed_of_sound_m_s"],
            air_absorption_db_per_m_100khz=obj_echo_cfg["air_absorption_db_per_m_100khz"],
            surface_roughness=obj_echo_cfg["surface_roughness"],
            surface_texture_scale_m=obj_echo_cfg["surface_texture_scale_m"],
            filter_taps=obj_echo_cfg["filter_taps"],
            bandwidth_khz=obj_echo_cfg["bandwidth_khz"],
            smoothing_frac=obj_echo_cfg["smoothing_frac"],
            snr_db=obj_echo_cfg["snr_db"],
            clutter_fraction=obj_echo_cfg["clutter_fraction"],
            num_clutter_paths=obj_echo_cfg["num_clutter_paths"],
            clutter_min_distance_m=obj_echo_cfg["clutter_min_distance_m"],
            clutter_max_distance_m=obj_echo_cfg["clutter_max_distance_m"],
            clutter_reflectivity_range=obj_echo_cfg["clutter_reflectivity_range"],
            clutter_roughness_range=obj_echo_cfg["clutter_roughness_range"],
            clutter_texture_scale_range_m=obj_echo_cfg["clutter_texture_scale_range_m"],
            device=device,
        )
        object_echoes_broad.append(echo_broad_i)
        true_delays.append(true_delay_i)

        # --- Narrowband spikes from the same echo for the foveal Doppler pathway. ---
        echo_narrow_probs_i = doppler_encoder.compute_spike_probabilities_from_waveform(
            waveform=echo_waveform_i,
            num_steps=num_steps,
            max_spike_prob=obj_echo_cfg["max_spike_prob"],
            sample_rate_hz=sample_rate_hz,
            step_duration_ms=step_duration_ms,
            filter_taps=velocity_decode_cfg["filter_taps"],
            smoothing_frac=velocity_decode_cfg["smoothing_frac"],
            device=device,
        )
        echo_narrow_i = doppler_encoder.sample_spikes_from_probabilities(
            spike_probs=sharpen_doppler_probabilities(echo_narrow_probs_i),
            batch_size=doppler_net_cfg["input_batch_size"],
            device=device,
        )
        object_echoes_narrow.append(echo_narrow_i)

    echo_broadband = torch.clamp(sum(object_echoes_broad), 0, 1)
    echo_doppler = torch.clamp(sum(object_echoes_narrow), 0, 1)

    # =====================================================================
    # 1. RANGE estimation (delay bank)
    # =====================================================================
    print("\n[RANGE ESTIMATION]  Delay bank")
    print("-" * 70)

    bank_spikes = net.forward(pulse_broadband, echo_broadband)
    spike_counts_np = bank_spikes.sum(dim=0)[0].detach().cpu().numpy()

    decoded_delay_objects = net.decode_population_response(
        spike_counts_np,
        min_peak_spikes=detect_cfg["min_spikes"],
        min_object_spikes=detect_cfg["min_spikes_pop"],
    )
    decoded_delay_objects.sort(
        key=lambda obj: (obj.peak_spikes, obj.total_spikes),
        reverse=True,
    )
    decoded_delay_objects = decoded_delay_objects[: len(object_cfgs)]
    if not decoded_delay_objects and spike_counts_np.max() > 0:
        best_idx = int(np.argmax(spike_counts_np))
        best_delay = float(candidate_delays[best_idx])
        best_spikes = float(spike_counts_np[best_idx])
        decoded_delay_objects = [
            DecodedObject(
                delay_steps=best_delay,
                peak_delay_steps=int(best_delay),
                start_delay_steps=int(best_delay),
                end_delay_steps=int(best_delay),
                total_spikes=best_spikes,
                peak_spikes=best_spikes,
            )
        ]

    delay_step_to_m = 0.5 * speed_of_sound_m_s * (step_duration_ms * 1e-3)

    for i, obj in enumerate(decoded_delay_objects):
        dist = obj.delay_steps * delay_step_to_m
        print(f"  Object {i}: decoded delay = {obj.delay_steps:.1f} steps  →  "
              f"distance = {dist:.4f} m  (peak spikes = {obj.peak_spikes:.0f})")
    print(f"  True delays: {true_delays} steps")

    # =====================================================================
    # 2. VELOCITY estimation (Doppler bank)
    # =====================================================================
    print("\n[VELOCITY ESTIMATION]  Foveal Doppler bank")
    print("-" * 70)

    decoded_doppler_objects = []
    if decoded_delay_objects:
        decoded_delay = int(round(decoded_delay_objects[0].delay_steps))
        print(f"  Using decoded delay = {decoded_delay} steps for echo-time gating")
        doppler_net = build_doppler_net(decoded_delay)

        doppler_bank_spikes = doppler_net.forward(pulse_doppler, echo_doppler)
        doppler_population_spikes = doppler_bank_spikes.sum(dim=1, keepdim=True)
        doppler_spike_counts_np = doppler_population_spikes.sum(dim=0)[0].detach().cpu().numpy()
        doppler_peak_threshold = max(
            float(detect_cfg["min_spikes"]),
            float(doppler_spike_counts_np.max()) * float(doppler_net_cfg["decode_peak_fraction"]),
        )
        decoded_doppler_objects = doppler_net.decode_population_response(
            doppler_spike_counts_np,
            min_peak_spikes=doppler_peak_threshold,
            min_object_spikes=max(float(detect_cfg["min_spikes_pop"]), doppler_peak_threshold),
            doppler_encoder=doppler_encoder,
            speed_of_sound_m_s=speed_of_sound_m_s,
        )
        decoded_doppler_objects.sort(key=lambda obj: obj.peak_spikes, reverse=True)
        decoded_doppler_objects = decoded_doppler_objects[: len(decoded_delay_objects)]
        if not decoded_doppler_objects and doppler_spike_counts_np.max() > 0:
            best_idx = int(np.argmax(doppler_spike_counts_np))
            best_shift = float(candidate_shifts[best_idx])
            best_spikes = float(doppler_spike_counts_np[best_idx])
            best_velocity = doppler_encoder.shift_channels_to_velocity(
                best_shift,
                speed_of_sound_m_s=speed_of_sound_m_s,
            )
            decoded_doppler_objects.append(
                DopplerDecodedObject(
                    shift_channels=best_shift,
                    peak_shift_channels=int(best_shift),
                    start_shift_channels=int(best_shift),
                    end_shift_channels=int(best_shift),
                    total_spikes=best_spikes,
                    peak_spikes=best_spikes,
                    velocity_m_s=best_velocity,
                )
            )
    else:
        print("  No decoded range available; skipping Doppler bank echo-time gating")
        doppler_population_spikes = torch.zeros(
            num_steps,
            1,
            len(candidate_shifts),
            device=device,
        )

    # Print expected shift for reference.
    for v_true in true_velocities:
        expected_shift = doppler_encoder.velocity_to_shift_channels(v_true, speed_of_sound_m_s)
        expected_df_khz = doppler_encoder.cf_freq_khz * 2.0 * v_true / speed_of_sound_m_s
        print(f"  True velocity = {v_true:.2f} m/s  →  Δf = {expected_df_khz * 1000:.0f} Hz  "
              f"→  expected shift = {expected_shift:.1f} channels")

    for i, obj in enumerate(decoded_doppler_objects):
        print(f"  Object {i}: decoded shift = {obj.shift_channels:.1f} ch  →  "
              f"velocity = {obj.velocity_m_s:.2f} m/s  "
              f"(peak spikes = {obj.peak_spikes:.0f})")

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, obj_cfg in enumerate(object_cfgs):
        print(f"\n  Object {i}:")
        print(f"    True  distance = {obj_cfg['distance_m']:.3f} m,  velocity = {obj_cfg.get('velocity_m_s', 0):.2f} m/s")
        if i < len(decoded_delay_objects):
            d = decoded_delay_objects[i].delay_steps * delay_step_to_m
            print(f"    Est.  distance = {d:.3f} m", end="")
        else:
            print(f"    Est.  distance = (not detected)", end="")
        if i < len(decoded_doppler_objects):
            print(f",  velocity = {decoded_doppler_objects[i].velocity_m_s:.2f} m/s")
        else:
            print(f",  velocity = (not detected)")

    # =====================================================================
    # Plots
    # =====================================================================
    plot_scenario_results(
        pulse_spikes=pulse_broadband,
        echo_spikes=echo_broadband,
        bank_spikes=bank_spikes,
        num_steps=num_steps,
        pulse_start_time=pulse_start_time,
        pulse_color=plot_cfg["pulse_color"],
        echo_color=plot_cfg["echo_color"],
    )

    plot_doppler_results(
        pulse_spikes_doppler=pulse_doppler[:, :1, :],
        echo_spikes_doppler=echo_doppler[:, :1, :],
        doppler_bank_spikes=doppler_population_spikes,
        freq_shifts=candidate_shifts,
        doppler_encoder=doppler_encoder,
        speed_of_sound_m_s=speed_of_sound_m_s,
        true_velocity_m_s=true_velocities,
        decoded_objects=decoded_doppler_objects,
        num_steps=num_steps,
        pulse_color=plot_cfg["pulse_color"],
        echo_color=plot_cfg["echo_color"],
        population_color=plot_cfg["doppler_bar_color"],
        population_alpha=plot_cfg["doppler_bar_alpha"],
        true_color=plot_cfg["true_color"],
        decoded_color=plot_cfg["decoded_color"],
    )

    plot_doppler_scenario_results(
        pulse_spikes_doppler=pulse_doppler[:, :1, :],
        echo_spikes_doppler=echo_doppler[:, :1, :],
        doppler_bank_spikes=doppler_population_spikes,
        num_steps=num_steps,
        pulse_color=plot_cfg["pulse_color"],
        echo_color=plot_cfg["echo_color"],
    )

    plot_range_and_velocity(
        pulse_spikes=pulse_broadband,
        echo_spikes=echo_broadband,
        delay_bank_spikes=bank_spikes,
        doppler_bank_spikes=doppler_population_spikes,
        candidate_delays=candidate_delays,
        freq_shifts=candidate_shifts,
        doppler_encoder=doppler_encoder,
        speed_of_sound_m_s=speed_of_sound_m_s,
        step_duration_ms=step_duration_ms,
        true_delays=true_delays,
        true_velocity_m_s=true_velocities,
        decoded_delay_objects=decoded_delay_objects,
        decoded_doppler_objects=decoded_doppler_objects,
        num_steps=num_steps,
        range_color=plot_cfg["range_color"],
        velocity_color=plot_cfg["velocity_color"],
        true_color=plot_cfg["true_color"],
        decoded_color=plot_cfg["decoded_color"],
    )


if __name__ == "__main__":
    run_fm_sweep_demo()
