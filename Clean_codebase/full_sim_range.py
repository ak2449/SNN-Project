import numpy as np
import torch

from sim_modules import (
    FMSweepBankNetwork,
    TonotopicEncoder,
    WTAReadoutLayer,
    generate_fm_echo_from_waveform,
    plot_scenario_results,
)


def run_fm_sweep_demo():
    """Run a modular FM sweep echolocation demo."""
    # ---------------------------------------------------------------------
    # Tunable parameters
    # ---------------------------------------------------------------------
    sim_cfg = {
        "device": "cpu",
        "num_steps": 120,
        "sample_rate_hz": 1000000,
        "step_duration_ms": 0.1,
    }

    encoder_cfg = {
        "freq_channels": 30,
        "f_min": 30,
        "f_max": 90,
    }

    network_cfg = {
        "delays_start": 1,
        "delays_stop": 100,  # np.arange stop (exclusive), supports ~10.0 m max
        "weight": 0.1,
        "beta": 0.1,
        "threshold": 0.7,
        "refractory_steps": 2,
        # Lateral inhibition across neighboring delay neurons.
        "lateral_inhibition_strength": 3,
        "lateral_inhibition_radius": 8,
        "inhibition_decay": 0.5,
    }
    readout_cfg = {
        "beta": 0.4,
        "threshold": 0.6,
        "inhibition_strength": 0,
        "inhibition_radius": 4,
        "inhibition_falloff": "gaussian",
        "sigma": 2.0,
    }

    pulse_cfg = {
        "pulse_start_time": 5,
        "duration": 8,
        "f_start": 90,
        "f_end": 30,
        "amplitude": 1.0,
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
        "snr_db": -20,
        "clutter_fraction": 0.6,
        "num_clutter_paths": 50,
        "clutter_min_distance_m": 0.05,
        "clutter_max_distance_m": 3.4,
        "clutter_reflectivity_range": (0.2, 0.9),
        "clutter_roughness_range": (0.1, 0.3),
        "clutter_texture_scale_range_m": (0.001, 0.004),
    }
    plot_cfg = {
        "pulse_color": "#3B6CA8",
        "echo_color": "#D86B6B",
    }

    object_cfgs = [
        {
            "distance_m": 1.4,
            "reflectivity": 0.9,
            "surface_roughness": 0.15,
        },
        {
            "distance_m": 0.5,
            "reflectivity": 0.75,
            "surface_roughness": 0.35,
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
    pulse_start_time = pulse_cfg["pulse_start_time"]

    encoder = TonotopicEncoder(**encoder_cfg)

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
    ).to(device)
    readout = WTAReadoutLayer(
        n_neurons=len(candidate_delays),
        beta=readout_cfg["beta"],
        threshold=readout_cfg["threshold"],
        inhibition_strength=readout_cfg["inhibition_strength"],
        inhibition_radius=readout_cfg["inhibition_radius"],
        inhibition_falloff=readout_cfg["inhibition_falloff"],
        sigma=readout_cfg["sigma"],
    ).to(device)

    print("=" * 60)
    print("FM SWEEP BAT ECHOLOCATION SIMULATION")
    print("=" * 60)

    print("\n[SCENARIO] Two static objects")
    print("-" * 60)

    pulse_waveform = encoder.generate_fm_sweep_waveform(
        num_steps=num_steps,
        start_time=pulse_start_time,
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
        num_steps=num_steps,
        max_spike_prob=pulse_encoding_cfg["max_spike_prob"],
        sample_rate_hz=sample_rate_hz,
        step_duration_ms=step_duration_ms,
        filter_taps=pulse_encoding_cfg["filter_taps"],
        bandwidth_khz=pulse_encoding_cfg["bandwidth_khz"],
        smoothing_frac=pulse_encoding_cfg["smoothing_frac"],
        device=device,
    )

    object_echoes = []
    true_delays = []
    for obj_cfg in object_cfgs:
        obj_echo_cfg = dict(echo_default_cfg)
        obj_echo_cfg.update(obj_cfg)
        echo_i, _, true_delay_i = generate_fm_echo_from_waveform(
            pulse_waveform=pulse_waveform,
            encoder=encoder,
            num_steps=num_steps,
            distance_m=obj_echo_cfg["distance_m"],
            reflectivity=obj_echo_cfg["reflectivity"],
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
        object_echoes.append(echo_i)
        true_delays.append(true_delay_i)

    echo = torch.clamp(sum(object_echoes), 0, 1)

    bank_spikes = net.forward(pulse, echo)
    wta_spikes = readout(bank_spikes)
    spike_counts = wta_spikes.sum(dim=0)
    if spike_counts.ndim != 2 or spike_counts.shape[0] != 1:
        raise ValueError("Demo currently expects batch size 1.")

    spike_counts_np = spike_counts[0].detach().cpu().numpy()
    active_indices = np.flatnonzero(spike_counts_np >= float(detect_cfg["min_spikes"]))
    detected_delays = candidate_delays[active_indices]
    counts = spike_counts_np[active_indices]
    decoded_objects = net.decode_population_response(
        spike_counts_np,
        min_peak_spikes=detect_cfg["min_spikes"],
        min_object_spikes=detect_cfg["min_spikes_pop"],
    )
    speed_of_sound_m_s = echo_default_cfg["speed_of_sound_m_s"]
    delay_step_to_distance_m = 0.5 * speed_of_sound_m_s * (step_duration_ms * 1e-3)

    print(f"True delays: {true_delays} steps")
    print(f"Detected delays: {detected_delays}")
    print(f"Spike counts: {counts}")
    print("\nDetection table")
    print("bank_index | delay_steps | delay_ms | distance_m | spike_count")
    if len(detected_delays) == 0:
        print("(none)")
    else:
        for delay_steps, spike_count in zip(detected_delays, counts):
            bank_idx = int(np.where(candidate_delays == int(delay_steps))[0][0])
            delay_ms = float(delay_steps) * step_duration_ms
            distance_m = float(delay_steps) * delay_step_to_distance_m
            print(
                f"{bank_idx:10d} | {int(delay_steps):11d} | {delay_ms:8.3f} | "
                f"{distance_m:10.4f} | {float(spike_count):10.2f}"
            )

    print("\nPopulation decoding table")
    print(
        "peak_index | peak_delay_steps | decoded_delay_steps | delay_ms | distance_m | "
        "support_range | peak_spikes | total_spikes"
    )
    if len(decoded_objects) == 0:
        print("(none)")
    else:
        for obj in decoded_objects:
            peak_idx = int(np.argmin(np.abs(candidate_delays - obj.peak_delay_steps)))
            delay_ms = obj.delay_steps * step_duration_ms
            distance_m = obj.delay_steps * delay_step_to_distance_m
            support_range = f"{obj.start_delay_steps}:{obj.end_delay_steps}"
            print(
                f"{peak_idx:10d} | {obj.peak_delay_steps:16d} | {obj.delay_steps:19.3f} | "
                f"{delay_ms:8.3f} | {distance_m:10.4f} | {support_range:13s} | "
                f"{obj.peak_spikes:11.2f} | {obj.total_spikes:12.2f}"
            )

    plot_scenario_results(
        pulse_spikes=pulse,
        echo_spikes=echo,
        bank_spikes=wta_spikes,
        num_steps=num_steps,
        pulse_start_time=pulse_start_time,
        pulse_color=plot_cfg["pulse_color"],
        echo_color=plot_cfg["echo_color"],
    )


if __name__ == "__main__":
    run_fm_sweep_demo()
