"""
PulseGuard AI Configuration
Centralized constants and configuration for the simulation engine.
"""

import numpy as np

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Default signal parameters
DEFAULT_FREQUENCY = 5.0e6  # Hz (Carrier Frequency slider default)
DEFAULT_AMPLITUDE = 3.3  # Vpp (Vpp slider default)
DEFAULT_OFFSET = 0.0  # Vdc (Voffset slider default)
DEFAULT_SAMPLE_RATE = 1.0e9  # Samples/sec (1 ns timestep)
DEFAULT_DURATION = 1.0e-6  # Seconds to simulate

# Rise time parameters
DEFAULT_RISE_TIME_TARGET = 10.0e-9  # seconds (Rise Time slider default)
RISE_TIME_CONSTANT = np.log(0.9) - np.log(0.1)  # for exponential rise/fall calculation

# Fault injection parameters
DEFAULT_IMPEDANCE_MISMATCH = 0.0  # 0.0V to 1.0V (Impedance Mismatch slider)
DEFAULT_NOISE_FLOOR = 0.0  # 0.0 to 1.0 (Noise Floor slider -> 0.00V to 0.50V)

# Digital Twin Effects
ADC_BITS = 10  # 10-bit ADC
THERMAL_DRIFT_COEFFICIENT = 0.02  # ±2% drift
THERMAL_WARMUP_TIME = 3600.0  # seconds (1 hour)
VNA_DEFAULT_DIRECTIVITY = 0.01 + 0.01j
VNA_DEFAULT_SOURCE_MATCH = 0.005 - 0.005j
SCPI_CALIBRATION_INTERVAL = 86400.0  # 24 hours in seconds

# =============================================================================
# SIGNAL PROCESSING CONSTANTS
# =============================================================================

# Mathematical constants
PI = np.pi
TWO_PI = 2 * np.pi

# For rise time calculation
RISE_TIME_LOW_THRESHOLD = 0.1  # 10%
RISE_TIME_HIGH_THRESHOLD = 0.9  # 90%

# Fault detection thresholds (from specs)
IMPEDANCE_MISMATCH_OVERSHOOT_THRESHOLD = 0.10  # > 10%
POWER_SUPPLI_VPP_THRESHOLD = 0.10  # > 100mV
PROBE_COMPENSATION_RISE_TIME_MULTIPLIER = 1.2  # > 1.2 × target

# =============================================================================
# STREAMLIT WIDGET MAPPING
# =============================================================================

# Widget ranges and steps (from CLAUDE.md section 9)
WIDGET_RANGES = {
    "carrier_frequency": (1.0e6, 25.0e6, 0.1e6),  # min, max, step in Hz
    "amplitude": (0.02, 5.0, 0.01),  # min, max, step in V (20mV to 5.0V)
    "offset": (-2.5, 2.5, 0.1),  # min, max, step in V
    "noise_floor": (0.0, 100.0, 1.0),  # min, max, step in %
    "rise_time_target": (1.0e-9, 100.0e-9, 1.0e-9),  # min, max, step in s (1-100 ns)
    "impedance_mismatch": (0.0, 1.0, 0.01),  # min, max, step in V
}

# Widget defaults
WIDGET_DEFAULTS = {
    "carrier_frequency": DEFAULT_FREQUENCY,
    "amplitude": DEFAULT_AMPLITUDE,
    "offset": DEFAULT_OFFSET,
    "noise_floor": DEFAULT_NOISE_FLOOR,
    "rise_time_target": DEFAULT_RISE_TIME_TARGET,
    "impedance_mismatch": DEFAULT_IMPEDANCE_MISMATCH,
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def calculate_rise_time_tau(rise_time_target):
    """
    Calculate tau constant for exponential rise/fall from rise time (10%-90%).

    Args:
        rise_time_target: Desired rise time in seconds (10% to 90%)

    Returns:
        Tau constant for exponential charging/discharging
    """
    return rise_time_target / RISE_TIME_CONSTANT


def map_noise_floor_to_std(noise_floor_percent):
    """
    Map Noise Floor slider percentage to noise standard deviation.

    Args:
        noise_floor_percent: Slider value from 0-100%

    Returns:
        noise_std: Standard deviation in volts (0.00V to 0.50V)
    """
    return (noise_floor_percent / 100.0) * 0.5


def get_ch2_dc_level(offset):
    """
    Determine CH2 DC level based on offset setting.

    Args:
        offset: DC offset voltage

    Returns:
        DC voltage level for CH2 (3.3V or 5.0V)
    """
    if abs(offset) < 0.1:  # Near zero offset
        return 3.3
    else:
        return 5.0  # Significant offset uses high rail
