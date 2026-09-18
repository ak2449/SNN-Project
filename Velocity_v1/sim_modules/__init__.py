from .echo_vel import generate_fm_echo_from_waveform
from .encoder_vel import TonotopicEncoder, DopplerEncoder
from .network_vel import (
    DecodedObject,
    DopplerDecodedObject,
    FovealDopplerBankNetwork,
    FMSweepBankNetwork,
)


def plot_pulse_echo_raster(*args, **kwargs):
    from .visualization_vel import plot_pulse_echo_raster as _plot_pulse_echo_raster

    return _plot_pulse_echo_raster(*args, **kwargs)


def plot_scenario_results(*args, **kwargs):
    from .visualization_vel import plot_scenario_results as _plot_scenario_results

    return _plot_scenario_results(*args, **kwargs)


def plot_doppler_results(*args, **kwargs):
    from .visualization_vel import plot_doppler_results as _plot_doppler_results

    return _plot_doppler_results(*args, **kwargs)


def plot_doppler_scenario_results(*args, **kwargs):
    from .visualization_vel import plot_doppler_scenario_results as _plot_doppler_scenario_results

    return _plot_doppler_scenario_results(*args, **kwargs)


def plot_range_and_velocity(*args, **kwargs):
    from .visualization_vel import plot_range_and_velocity as _plot_range_and_velocity

    return _plot_range_and_velocity(*args, **kwargs)


__all__ = [
    "TonotopicEncoder",
    "DopplerEncoder",
    "generate_fm_echo_from_waveform",
    "FMSweepBankNetwork",
    "FovealDopplerBankNetwork",
    "DecodedObject",
    "DopplerDecodedObject",
    "plot_pulse_echo_raster",
    "plot_scenario_results",
    "plot_doppler_results",
    "plot_doppler_scenario_results",
    "plot_range_and_velocity",
]
