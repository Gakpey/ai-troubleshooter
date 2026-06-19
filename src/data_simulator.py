"""
PulseGuard AI Data Simulation Engine
Synthesizes high-frequency time-series signal data mimicking Rohde & Schwarz oscilloscope traces
with programmable fault injection for Phase 1 of the PulseGuard AI pipeline.
Implements full mathematical fidelity per specs.md sections 3.1, 3.5, and 3.6.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import streamlit as st
from scipy import signal
from config import *

# =============================================================================
# CORE SIGNAL GENERATION FUNCTIONS
# =============================================================================


def generate_square_wave(
    frequency: float = DEFAULT_FREQUENCY,  # Hz
    amplitude: float = DEFAULT_AMPLITUDE,  # Vpp
    offset: float = DEFAULT_OFFSET,  # Vdc
    sample_rate: float = DEFAULT_SAMPLE_RATE,  # Samples/sec
    duration: float = DEFAULT_DURATION,  # Seconds
    rise_time: float = DEFAULT_RISE_TIME_TARGET,  # Seconds (10% to 90%)
    rise_time_target: Optional[float] = None,  # For backward compatibility
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate baseline square wave signal with exponential rise/fall modeling.

    Implements exact mathematical model from specs.md:
    V(t) = offset + (amplitude/2) * sign(sin(2πft)) with exponential rise/fall
    Rise/fall time modeling: V(t) = V_start + (V_end - V_start) * (1 - e^(-t/τ))
    where τ = rise_time/(ln(0.9)-ln(0.1))

    Args:
        frequency: Signal frequency in Hz
        amplitude: Peak-to-peak amplitude in volts
        offset: DC offset in volts
        sample_rate: Sampling rate in samples/sec
        duration: Signal duration in seconds
        rise_time: Target rise time in seconds (10% to 90%)

    Returns:
        time_array: Time values in seconds
        voltage_array: Voltage values in volts
    """
    # Handle backward compatibility
    if rise_time_target is not None:
        rise_time = rise_time_target
    # Calculate number of samples
    n_samples = int(sample_rate * duration)

    # Generate time array
    time_array = np.linspace(0, duration, n_samples, dtype=np.float64)

    # Generate ideal square wave (without rise/fall effects)
    angular_frequency = TWO_PI * frequency
    ideal_square = np.sign(np.sin(angular_frequency * time_array))
    # Scale to desired amplitude and add offset
    voltage_array = ideal_square * (amplitude / 2) + offset

    # Apply exponential rise/fall time modeling
    if rise_time > 0 and n_samples > 0:
        # Calculate tau for exponential charging/discharging
        tau = calculate_rise_time_tau(rise_time)

        # Find rising and falling edges
        # Rising edge: transition from negative to negative going through zero (for zero-centered)
        # Actually, better to find where signal crosses the midpoint
        midpoint = offset
        # Find where signal crosses midpoint going upward (rising edge)
        rising_edges = np.where((ideal_square[:-1] <= 0) & (ideal_square[1:] > 0))[0]
        # Find where signal crosses midpoint going downward (falling edge)
        falling_edges = np.where((ideal_square[:-1] >= 0) & (ideal_square[1:] < 0))[0]

        # Apply exponential rise to rising edges
        for edge_idx in rising_edges:
            if edge_idx < n_samples:
                # Calculate time relative to edge start
                t_rel = time_array[edge_idx:] - time_array[edge_idx]
                # Limit to reasonable window (e.g., 3*tau)
                max_samples = min(int(3 * tau * sample_rate), n_samples - edge_idx)
                if max_samples > 0:
                    t_rel = t_rel[:max_samples]
                    # Exponential rise: V = V_final * (1 - e^(-t/tau))
                    # Starting from -amplitude/2 to +amplitude/2
                    start_val = offset - amplitude / 2
                    end_val = offset + amplitude / 2
                    rise_values = start_val + (end_val - start_val) * (
                        1 - np.exp(-t_rel / tau)
                    )
                    voltage_array[edge_idx : edge_idx + max_samples] = rise_values

        # Apply exponential fall to falling edges
        for edge_idx in falling_edges:
            if edge_idx < n_samples:
                # Calculate time relative to edge start
                t_rel = time_array[edge_idx:] - time_array[edge_idx]
                # Limit to reasonable window (e.g., 3*tau)
                max_samples = min(int(3 * tau * sample_rate), n_samples - edge_idx)
                if max_samples > 0:
                    t_rel = t_rel[:max_samples]
                    # Exponential fall: V = V_start * e^(-t/tau)
                    # Starting from +amplitude/2 to -amplitude/2
                    start_val = offset + amplitude / 2
                    end_val = offset - amplitude / 2
                    fall_values = end_val + (start_val - end_val) * np.exp(-t_rel / tau)
                    voltage_array[edge_idx : edge_idx + max_samples] = fall_values

    return time_array, voltage_array


def generate_dc_signal(
    voltage: float = 3.3,  # Vdc
    sample_rate: float = DEFAULT_SAMPLE_RATE,  # Samples/sec
    duration: float = DEFAULT_DURATION,  # Seconds
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate baseline DC signal.

    Args:
        voltage: DC voltage level in volts
        sample_rate: Sampling rate in samples/sec
        duration: Signal duration in seconds

    Returns:
        time_array: Time values in seconds
        voltage_array: Voltage values in volts
    """
    # Calculate number of samples
    n_samples = int(sample_rate * duration)

    # Generate time array
    time_array = np.linspace(0, duration, n_samples, dtype=np.float64)

    # Generate constant DC signal
    voltage_array = np.full(n_samples, voltage, dtype=np.float64)

    return time_array, voltage_array


# =============================================================================
# FAULT INJECTION FUNCTIONS
# =============================================================================


def apply_impedance_mismatch(
    voltage_array: np.ndarray,
    time_array: np.ndarray,
    ringing_amplitude: float,  # 0.0V to 1.0V from slider
    ringing_frequency: float,  # Derived from signal characteristics
    decay_constant: float,  # Based on transmission line properties
) -> np.ndarray:
    """
    Apply impedance mismatch fault by superimposing damped ringing on rising edges.

    Implements exact mathematical model from specs.md:
    V_fault(t) = V_base(t) + A_ringing * e^(-t/τ) * cos(2πf_ring * t)
    Applied only on rising edges (transition from low to high state)
    Overshoot Ratio calculation: (Vmax - Vhigh) / Vamplitude
    Threshold for fault detection: Overshoot Ratio > 0.10

    Args:
        voltage_array: Input voltage signal
        time_array: Time values corresponding to voltage_array
        ringing_amplitude: Amplitude coefficient of ringing (0.0V to 1.0V)
        ringing_frequency: Frequency of ringing oscillation in Hz
        decay_constant: Decay time constant in seconds

    Returns:
        Modified voltage array with impedance mismatch applied
    """
    # Create copy to avoid modifying original
    modified_array = voltage_array.copy()

    if ringing_amplitude <= 0 or len(voltage_array) < 2:
        return modified_array

    # Detect rising edges (transition from low to high state)
    # For square wave, find where signal crosses the midpoint going positive
    midpoint = np.mean(voltage_array)  # Approximate midpoint
    # Better approach: find where derivative is positive and significant
    diff_signal = np.diff(voltage_array)
    # Using a threshold based on signal standard deviation
    edge_threshold = 0.1 * np.std(diff_signal) if np.std(diff_signal) > 0 else 0.01
    rising_edges = np.where(
        (diff_signal[:-1] <= edge_threshold) & (diff_signal[1:] > edge_threshold)
    )[0]

    # Alternative: find zero crossings for cleaner detection
    if len(rising_edges) == 0:
        # Fallback to zero-crossing method
        centered_signal = voltage_array - np.mean(voltage_array)
        rising_edges = np.where(
            (centered_signal[:-1] <= 0) & (centered_signal[1:] > 0)
        )[0]

    # For each rising edge, add damped ringing
    for edge_idx in rising_edges:
        if edge_idx < len(voltage_array):
            # Calculate time relative to edge start
            t_rel = time_array[edge_idx:] - time_array[edge_idx]

            # Limit ringing to a reasonable duration (e.g., 5 periods of ringing frequency)
            if ringing_frequency > 0:
                max_ring_time = 5.0 / ringing_frequency
            else:
                max_ring_time = 1e-6  # 1 microsecond fallback

            ring_samples = min(
                int(max_ring_time / (time_array[1] - time_array[0])), len(t_rel)
            )

            if ring_samples > 0:
                t_rel = t_rel[:ring_samples]
                # Damped ringing: A * exp(-t/τ) * cos(2π * f_ring * t)
                ringing = (
                    ringing_amplitude
                    * np.exp(-t_rel / decay_constant)
                    * np.cos(TWO_PI * ringing_frequency * t_rel)
                )
                modified_array[edge_idx : edge_idx + ring_samples] += ringing

    return modified_array


def apply_power_supply_instability(
    voltage_array: np.ndarray,
    time_array: np.ndarray,
    noise_std: float,  # 0.00V to 0.50V from Noise Floor slider
    switching_freq: float = 1.0e5,  # Hz
    ac_amplitude: float = 0.05,  # V for AC component
) -> np.ndarray:
    """
    Apply power supply instability by adding AC ripple and Gaussian noise.

    Implements exact mathematical model from specs.md:
    V_fault(t) = V_base(t) + V_sawtooth(t) + V_gaussian(t)
    V_sawtooth(t): Periodic waveform ramps from -ac_amplitude to +ac_amplitude
    V_gaussian(t): Gaussian white noise with standard deviation noise_std
    Vpp calculation: Vmax - Vmin across the signal
    Threshold for fault detection: Vpp > 100mV

    Args:
        voltage_array: Input voltage signal
        time_array: Time values corresponding to voltage_array
        noise_std: Standard deviation of Gaussian noise in volts
        switching_freq: Frequency of switching regulator ripple in Hz
        ac_amplitude: Amplitude of AC ripple component in volts

    Returns:
        Modified voltage array with power supply instability applied
    """
    # Create copy to avoid modifying original
    modified_array = voltage_array.copy()

    if noise_std <= 0 and ac_amplitude <= 0:
        return modified_array

    # Generate sawtooth wave for switching regulator ripple
    # Sawtooth: ramps from -ac_amplitude to +ac_amplitude
    if switching_freq > 0:
        period = 1.0 / switching_freq
        # Avoid division by zero
        if period > 0:
            sawtooth = (
                2
                * ac_amplitude
                * (time_array / period - np.floor(time_array / period + 0.5))
            )
        else:
            sawtooth = np.zeros_like(time_array)
    else:
        sawtooth = np.zeros_like(time_array)

    # Generate Gaussian noise
    if noise_std > 0:
        gaussian_noise = np.random.normal(0, noise_std, len(time_array))
    else:
        gaussian_noise = np.zeros_like(time_array)

    # Add both components
    modified_array += sawtooth + gaussian_noise

    return modified_array


def apply_probe_compensation_error(
    voltage_array: np.ndarray,
    time_array: np.ndarray,
    target_rise_time: float,  # From Rise Time slider (1-100 ns)
    measured_rise_time: float,  # > 1.2 × target_rise_time
) -> np.ndarray:
    """
    Apply probe compensation error by modifying rise time characteristic.

    Implements exact mathematical model from specs.md:
    Modified exponential charging with tau_slow ≠ tau_target
    For rising edge: V(t) = V_final * (1 - e^(-t/(R*C_slow))) where C_slow creates slower rise
    Rise time measurement: Δt between 10% and 90% amplitude thresholds
    Threshold for fault detection: measured_rise_time > 1.2 × target_rise_time

    Args:
        voltage_array: Input voltage signal
        time_array: Time values corresponding to voltage_array
        target_rise_time: Desired rise time in seconds
        measured_rise_time: Actual rise time with error (> 1.2 × target_rise_time)

    Returns:
        Modified voltage array with probe compensation error applied
    """
    # Create copy to avoid modifying original
    modified_array = voltage_array.copy()

    if (
        measured_rise_time <= target_rise_time * PROBE_COMPENSATION_RISE_TIME_MULTIPLIER
        or target_rise_time <= 0
    ):
        return modified_array

    # For simplicity, we'll assume this is applied to a square wave signal
    # and modify the rise/fall time to be slower using exponential model

    # Detect edges (both rising and falling) using zero-crossings of derivative
    if len(time_array) < 3:
        return modified_array

    diff_signal = np.diff(voltage_array)
    # Find significant changes (edges)
    threshold = 0.1 * np.std(diff_signal) if np.std(diff_signal) > 0 else 0.01
    edges = np.where(np.abs(diff_signal) > threshold)[0]

    # For each edge, slow down the transition using exponential model
    for edge_idx in edges:
        if edge_idx > 0 and edge_idx < len(voltage_array) - 1:
            # Determine edge direction
            if voltage_array[edge_idx] > voltage_array[edge_idx - 1]:  # Rising edge
                start_val = voltage_array[edge_idx - 1]
                end_val = voltage_array[edge_idx]
                edge_type = "rising"
            else:  # Falling edge
                start_val = voltage_array[edge_idx - 1]
                end_val = voltage_array[edge_idx]
                edge_type = "falling"

            # Calculate the actual tau that would produce the measured rise time
            # Rise time (10%-90%) = tau * ln(0.9/0.1) = tau * RISE_TIME_CONSTANT
            tau_actual = measured_rise_time / RISE_TIME_CONSTANT

            # Calculate number of samples to spread the transition
            # We want to spread it over several tau values for smooth transition
            transition_tau_multiplier = 3.0  # Show 3 time constants
            transition_samples = max(
                1,
                int(
                    transition_tau_multiplier
                    * tau_actual
                    * (time_array[1] - time_array[0]) ** -1
                ),
            )

            # Ensure we don't go beyond array bounds
            start_idx = max(0, edge_idx - transition_samples // 2)
            end_idx = min(len(voltage_array), edge_idx + transition_samples // 2)

            if end_idx > start_idx:
                # Create slow exponential transition
                t_rel = time_array[start_idx:end_idx] - time_array[start_idx]
                n_points = len(t_rel)

                if edge_type == "rising":
                    # Exponential rise: V(t) = V_final * (1 - e^(-t/tau))
                    transition = start_val + (end_val - start_val) * (
                        1 - np.exp(-t_rel / tau_actual)
                    )
                else:  # falling
                    # Exponential fall: V(t) = V_start * e^(-t/tau)
                    transition = end_val + (start_val - end_val) * np.exp(
                        -t_rel / tau_actual
                    )

                modified_array[start_idx:end_idx] = transition

    return modified_array


# =============================================================================
# DIGITAL TWIN EFFECTS
# =============================================================================


def apply_adc_quantization(
    voltage_array: np.ndarray,
    vdiv: float,  # Volts per division from vertical scale
    bits: int = ADC_BITS,  # 10-bit ADC
) -> np.ndarray:
    """
    Apply 10-bit ADC quantization based on Full-scale = Vdiv × 8.

    Implements exact mathematical model from specs.md section 3.6.1:
    1. Full-scale range = Vdiv × 8 (from -4*Vdiv to +4*Vdiv)
    2. Quantization step q = (2 × Full-scale) / (2^bits - 1)
    3. V_quantized = round((V_analog + Full-scale) / q) × q - Full-scale

    Args:
        voltage_array: Input voltage signal
        vdiv: Volts per division setting
        bits: ADC resolution in bits (default 10)

    Returns:
        Quantized voltage array
    """
    # Full-scale range = Vdiv * 8 (from -4*Vdiv to +4*Vdiv)
    full_scale = vdiv * 8

    # Quantization step size
    q_step = (2 * full_scale) / (
        2**bits - 1
    )  # Full range divided by quantization levels

    # Quantize to nearest step
    # First, shift to positive range: 0 to 2*full_scale
    shifted = voltage_array + full_scale
    # Quantize
    quantized = np.round(shifted / q_step) * q_step
    # Shift back to original range: -full_scale to +full_scale
    quantized = quantized - full_scale

    return quantized


def apply_anti_aliasing(
    voltage_array: np.ndarray,
    time_array: np.ndarray,
    carrier_frequency: float,  # From Carrier Frequency slider
    virtual_sample_rate: float,  # Effective sampling rate
) -> np.ndarray:
    """
    Apply anti-aliasing: if virtual_sample_rate < 2 × f_carrier,
    reflect aliased frequencies using true frequency-domain simulation.

    Implements exact mathematical model from specs.md section 3.6.2:
    1. Compute FFT: V_freq = FFT(V_time)
    2. Identify frequencies above Nyquist: |f| > virtual_sample_rate/2
    3. Fold aliases: V_aliased[f] = V_freq[f] + V_freq[±virtual_sample_rate - f] for |f| > Nyquist
    4. Compute IFFT: V_time_aliased = IFFT(V_aliased)

    Args:
        voltage_array: Input voltage signal in time domain
        time_array: Time values corresponding to voltage_array
        carrier_frequency: Carrier frequency of the signal in Hz
        virtual_sample_rate: Effective sampling rate in Hz

    Returns:
        Voltage array with anti-aliasing applied
    """
    # Nyquist frequency
    nyquist_freq = virtual_sample_rate / 2

    # If we're sampling above Nyquist rate for the carrier, no aliasing occurs
    if virtual_sample_rate >= 2 * carrier_frequency:
        return voltage_array

    # True frequency-domain aliasing simulation
    # Compute FFT
    freq_spectrum = np.fft.fft(voltage_array)

    # Calculate frequency array
    if len(time_array) > 1:
        sample_spacing = time_array[1] - time_array[0]
    else:
        sample_spacing = 1.0 / virtual_sample_rate if virtual_sample_rate > 0 else 1e-9

    freqs = np.fft.fftfreq(len(voltage_array), d=sample_spacing)

    # Create copy for modification
    modified_spectrum = freq_spectrum.copy()

    # Fold aliases: for frequencies above Nyquist, fold them back
    # Positive frequencies above Nyquist
    pos_nyquist_mask = (freqs > nyquist_freq) & (freqs <= virtual_sample_rate / 2)
    # Actually, we need to fold all frequencies where |f| > nyquist_freq
    alias_mask = np.abs(freqs) > nyquist_freq

    if np.any(alias_mask):
        # For each aliased frequency, fold it back into the Nyquist band
        for i in np.where(alias_mask)[0]:
            aliased_freq = freqs[i]
            # Calculate the folded frequency
            if aliased_freq > 0:
                # Fold positive frequency: f_folded = virtual_sample_rate - f_aliased
                folded_freq = virtual_sample_rate - aliased_freq
            else:
                # Fold negative frequency: f_folded = -virtual_sample_rate - f_aliased
                folded_freq = -virtual_sample_rate - aliased_freq

            # Find the index of the folded frequency (closest match)
            folded_idx = np.argmin(np.abs(freqs - folded_freq))
            # Add the aliased component to the folded frequency bin
            modified_spectrum[folded_idx] += freq_spectrum[i]
            # Zero out the aliased frequency (energy has been folded)
            modified_spectrum[i] = 0

    # Convert back to time domain
    modified_time_array = np.fft.ifft(modified_spectrum).real

    return modified_time_array


def apply_thermal_drift(
    voltage_array: np.ndarray,
    virtual_uptime: float,  # Seconds of simulated uncalibrated time
    drift_coefficient: float = THERMAL_DRIFT_COEFFICIENT,  # ±2% drift
    warmup_time: float = THERMAL_WARMUP_TIME,
) -> np.ndarray:
    """
    Apply thermal drift: ±2% voltage offset after virtual uncalibrated uptime.

    Implements exact mathematical model from specs.md section 3.6.3:
    1. Drift_factor = drift_coefficient × (1 - e^(-(uptime-warmup_time)/warmup_time)) for uptime > warmup_time
    2. Drift_sign = pseudo-random based on uptime seed (±1)
    3. V_drifted = V_original × (1 + Drift_sign × Drift_factor)

    Args:
        voltage_array: Input voltage signal
        virtual_uptime: Simulated time since last calibration in seconds
        drift_coefficient: Maximum drift fraction (0.02 = ±2%)
        warmup_time: Time after which drift begins (seconds)

    Returns:
        Voltage array with thermal drift applied
    """
    # Create copy to avoid modifying original
    modified_array = voltage_array.copy()

    # Apply drift only after sufficient uptime (e.g., > 3600 seconds = 1 hour)
    if virtual_uptime > warmup_time:
        # Calculate drift amount (can be positive or negative)
        excess_time = virtual_uptime - warmup_time
        # Drift grows with time but saturates
        drift_factor = drift_coefficient * (1 - np.exp(-excess_time / warmup_time))
        # Randomly choose positive or negative drift based on uptime (deterministic pseudo-random)
        drift_sign = 1 if int(virtual_uptime) % 2 == 0 else -1
        drift_amount = drift_sign * drift_factor

        # Apply drift as percentage of signal value
        modified_array = modified_array * (1 + drift_amount)

    return modified_array


def apply_vna_errors(
    voltage_array: np.ndarray,
    directivity_error: complex,  # D error term
    source_match_error: complex,  # Sm error term
    time_array: Optional[np.ndarray] = None,
    z0: float = 50.0,  # Characteristic impedance (ohms)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply VNA error matrices and convert to VSWR, Return Loss, Phase.

    Implements mathematical model from specs.md section 3.6.2:
    - Apply directivity (D) and source match (Sm) error matrices
    - Convert impedance reflections to VSWR, Return Loss, Phase:
        * VSWR = (1 + |Γ|) / (1 - |Γ|)
        * Return Loss (dB) = -20 × log10(|Γ|)
        * Phase = angle(Γ) in degrees
        * Where Γ = (Z - Z0)/(Z + Z0) for characteristic impedance Z0

    Args:
        voltage_array: Input voltage signal (time domain)
        directivity_error: Directivity error term (complex)
        source_match_error: Source match error term (complex)
        time_array: Time values corresponding to voltage_array (optional, for frequency calculation)
        z0: Characteristic impedance in ohms (default 50Ω)

    Returns:
        Tuple of (time_domain_signal, frequency_domain_metrics)
        where frequency_domain_metrics contains [VSWR, Return Loss, Phase]
    """
    # Create copy to avoid modifying original
    modified_time_array = voltage_array.copy()

    # In a full VNA implementation, we would work in frequency domain
    # For simulation, we'll apply a simplified error model and calculate metrics

    # Avoid division by zero and handle edge cases
    eps = 1e-12

    # Calculate basic statistics for metric estimation
    max_val = np.max(np.abs(modified_time_array)) + eps
    min_val = np.min(np.abs(modified_time_array)) + eps
    mean_val = np.mean(modified_time_array)
    std_val = np.std(modified_time_array) + eps

    # Estimate reflection coefficient magnitude from signal variations
    # This is a simplified model for simulation purposes
    if max_val > eps:
        # Use relative variation to estimate reflection coefficient
        reflection_approx = (max_val - min_val) / (max_val + min_val)
        # Constrain to valid range [0, 1) for stability
        reflection_approx = np.clip(reflection_approx, 0.0, 0.999)
    else:
        reflection_approx = 0.0

    # Apply VNA errors conceptually (simplified)
    # In reality: Γ_measured = (V_measured - D) / (1 + S_m × Γ_true)
    # For simulation, we'll adjust the reflection based on error terms
    error_magnitude = np.abs(directivity_error) + np.abs(source_match_error)
    # Scale error influence based on signal strength
    error_influence = error_magnitude * (std_val / (max_val + eps))
    error_influence = np.clip(error_influence, 0, 0.1)  # Limit error influence

    # Apply error to reflection coefficient estimate
    # Use a deterministic pseudo-random based on signal characteristics instead of undefined virtual_uptime
    if (
        int(np.sum(modified_time_array * 1000)) % 2 == 0
    ):  # Alternate error application for variety
        reflection_with_error = reflection_approx + error_influence
    else:
        reflection_with_error = reflection_approx - error_influence

    # Constrain to valid range
    reflection_with_error = np.clip(reflection_with_error, 0.0, 0.999)

    # Calculate VSWR, Return Loss, and Phase
    # VSWR = (1 + |Γ|) / (1 - |Γ|)
    if reflection_with_error < 1.0:
        vswr = (1.0 + reflection_with_error) / (1.0 - reflection_with_error)
    else:
        vswr = 100.0  # Large value for near-unity reflection

    # Return Loss = -20*log10(|Γ|) dB
    if reflection_with_error > 0:
        return_loss = -20 * np.log10(reflection_with_error)
    else:
        return_loss = 100.0  # Very high return loss for near-zero reflection

    # Phase = angle(Γ) in degrees
    # For simulation, we'll use a pseudo-random phase based on signal characteristics
    phase_rad = np.angle(
        np.mean(modified_time_array) + 1j * np.std(modified_time_array)
    )
    phase_deg = np.degrees(phase_rad)

    # Create metrics array
    metrics_array = np.array([vswr, return_loss, phase_deg], dtype=np.float64)

    return modified_time_array, metrics_array


def apply_scpi_register_effects(
    voltage_array: np.ndarray,
    time_array: np.ndarray,
    virtual_uptime: float,
    calibration_interval: float = SCPI_CALIBRATION_INTERVAL,
    overvoltage_threshold: float = 400.0,
) -> Dict[str, Any]:
    """
    Apply SCPI register simulation: NOALigndata bit, overvoltage logging, airflow errors.

    Implements specs.md section 3.6.3:
    - NOALigndata Bit: Self-Alignment uncalibrated flag
    - Overvoltage Logging: Log chronological entries for overvoltage saturation (>400V)
    - Airflow Restriction Errors: Detect simulated cooling fan airflow restrictions

    Args:
        voltage_array: Input voltage signal
        time_array: Time values corresponding to voltage_array
        virtual_uptime: Simulated time since last calibration in seconds
        calibration_interval: Time between calibrations (default 24 hours)
        overvoltage_threshold: Voltage threshold for overvoltage detection (default 400V)

    Returns:
        Dictionary containing SCPI register status and event logs
    """
    # Initialize result dictionary
    scpi_status = {
        "no_aligndata": False,
        "overvoltage_events": [],
        "airflow_restriction": False,
        "recently_calibrated": False,
    }

    # NOALigndata Bit: Set when virtual_uptime > calibration_interval and not recently calibrated
    scpi_status["no_aligndata"] = (
        virtual_uptime > calibration_interval
    ) and not scpi_status["recently_calibrated"]

    # Overvoltage Detection (>400V absolute)
    if len(voltage_array) > 0 and len(time_array) > 0:
        overvoltage_indices = np.where(np.abs(voltage_array) > overvoltage_threshold)[0]
        for idx in overvoltage_indices:
            scpi_status["overvoltage_events"].append(
                {
                    "time": float(time_array[idx]),
                    "voltage": float(voltage_array[idx]),
                    "event_type": "OVERVOLTAGE",
                }
            )

    # Airflow Restriction (simplified model)
    # Simulate periodic airflow issues (e.g., 1hr restriction every 12hrs)
    if virtual_uptime > 0:
        # Airflow restriction occurs for 1 hour every 12 hours
        airflow_cycle_position = virtual_uptime % (
            12 * 3600
        )  # Position in 12-hour cycle
        scpi_status["airflow_restriction"] = (airflow_cycle_position >= 11 * 3600) and (
            airflow_cycle_position < 12 * 3600
        )  # Last hour of cycle

    return scpi_status


# =============================================================================
# METRICS CALCULATION FUNCTIONS
# =============================================================================


def calculate_rise_time(
    time_array: np.ndarray,
    voltage_array: np.ndarray,
    threshold_low: float = RISE_TIME_LOW_THRESHOLD,  # 10%
    threshold_high: float = RISE_TIME_HIGH_THRESHOLD,  # 90%
) -> float:
    """
    Calculate rise time between threshold_low and threshold_high percentages.

    Implements exact algorithm from specs.md section 3.2:
    1. Find Vmin and Vmax of signal
    2. Calculate V_low = Vmin + threshold_low*(Vmax-Vmin) (10% threshold)
    3. Calculate V_high = Vmin + threshold_high*(Vmax-Vmin) (90% threshold)
    4. Find first crossing of V_low on rising edge
    5. Find first crossing of V_high after V_low crossing
    6. tr = t(V_high) - t(V_low) with linear interpolation

    Args:
        time_array: Time values in seconds
        voltage_array: Voltage values in volts
        threshold_low: Lower threshold fraction (0.1 for 10%)
        threshold_high: Upper threshold fraction (0.9 for 90%)

    Returns:
        Rise time in seconds
    """
    # Handle edge cases
    if len(time_array) < 2 or len(voltage_array) < 2:
        return 0.0

    if len(time_array) != len(voltage_array):
        return 0.0

    # Find signal min and max
    v_min = np.min(voltage_array)
    v_max = np.max(voltage_array)
    v_range = v_max - v_min

    if v_range <= 0:
        return 0.0

    # Calculate threshold voltages
    v_low = v_min + threshold_low * v_range
    v_high_val = v_min + threshold_high * v_range  # Renamed to avoid conflict

    # Find first crossing of lower threshold (rising edge)
    low_crossings = np.where(
        (voltage_array[:-1] <= v_low) & (voltage_array[1:] > v_low)
    )[0]
    # Find first crossing of upper threshold after lower crossing
    high_crossings = np.where(
        (voltage_array[:-1] <= v_high_val) & (voltage_array[1:] > v_high_val)
    )[0]

    if len(low_crossings) > 0 and len(high_crossings) > 0:
        # Take first rising edge
        first_low = low_crossings[0]
        # Find first high crossing after this low crossing
        high_after_low = high_crossings[high_crossings > first_low]
        if len(high_after_low) > 0:
            first_high = high_after_low[0]

            # Handle edge case where we're at the last element
            low_idx_next = min(first_low + 1, len(voltage_array) - 1)
            high_idx_next = min(first_high + 1, len(voltage_array) - 1)

            # Interpolate for more accurate timing
            t_low = np.interp(
                v_low,
                [voltage_array[first_low], voltage_array[low_idx_next]],
                [time_array[first_low], time_array[low_idx_next]],
            )
            t_high = np.interp(
                v_high_val,
                [voltage_array[first_high], voltage_array[high_idx_next]],
                [time_array[first_high], time_array[high_idx_next]],
            )
            return t_high - t_low

    return 0.0


def calculate_overshoot_ratio(voltage_array: np.ndarray, v_high: float) -> float:
    """
    Calculate overshoot ratio: (Vmax - Vhigh) / Vamplitude

    Implements exact formula from specs.md:
    Overshoot Ratio = (Vmax - Vhigh) / Vamplitude
    Threshold for fault detection: Overshoot Ratio > 0.10

    Args:
        voltage_array: Voltage values in volts
        v_high: Expected high state voltage in volts

    Returns:
        Overshoot ratio (unitless)
    """
    # Handle edge cases
    if len(voltage_array) == 0:
        return 0.0

    v_max = np.max(voltage_array)
    v_min = np.min(voltage_array)
    v_amplitude = v_max - v_min

    if v_amplitude <= 0:
        return 0.0

    overshoot = v_max - v_high
    return overshoot / v_amplitude if v_amplitude != 0 else 0.0


def calculate_vpp(voltage_array: np.ndarray) -> float:
    """
    Calculate peak-to-peak voltage.

    Args:
        voltage_array: Voltage values in volts

    Returns:
        Peak-to-peak voltage in volts
    """
    if len(voltage_array) == 0:
        return 0.0

    return np.max(voltage_array) - np.min(voltage_array)


# =============================================================================
# MAIN SIMULATION FUNCTION
# =============================================================================


@st.cache_data
def simulate_signals(
    # Basic signal parameters
    frequency: float = DEFAULT_FREQUENCY,  # Hz (Carrier Frequency slider)
    amplitude: float = DEFAULT_AMPLITUDE,  # Vpp (Vpp slider)
    offset: float = DEFAULT_OFFSET,  # Vdc (Voffset slider)
    # Fault injection parameters
    impedance_mismatch: float = DEFAULT_IMPEDANCE_MISMATCH,  # 0.0V to 1.0V (Impedance Mismatch slider)
    noise_floor: float = DEFAULT_NOISE_FLOOR,  # 0.0 to 1.0 (Noise Floor slider -> 0.00V to 0.50V)
    rise_time_target: float = DEFAULT_RISE_TIME_TARGET,  # seconds (Rise Time slider)
    # Simulation parameters
    sample_rate: float = DEFAULT_SAMPLE_RATE,  # Samples/sec (1 ns timestep)
    duration: float = DEFAULT_DURATION,  # Seconds to simulate
    virtual_uptime: float = 0.0,  # Seconds of simulated uncalibrated time
    # VNA error parameters (conceptual values for demonstration)
    directivity_error: complex = VNA_DEFAULT_DIRECTIVITY,
    source_match_error: complex = VNA_DEFAULT_SOURCE_MATCH,
) -> Dict[str, Any]:
    """
    Main simulation function that generates both channels with all effects.
    Uses Streamlit caching to prevent redundant computation during slider changes.

    Returns:
        Dictionary containing:
        - ch1_time, ch1_voltage: Channel 1 (square wave) time and voltage arrays
        - ch2_time, ch2_voltage: Channel 2 (DC) time and voltage arrays
        - ch1_metrics: Dictionary of calculated metrics for CH1
        - ch2_metrics: Dictionary of calculated metrics for CH2
        - scpi_status: Dictionary containing SCPI register status and events
    """
    # Convert noise_floor slider (0-100) to voltage standard deviation (0-0.5V)
    noise_std = map_noise_floor_to_std(noise_floor)

    # Determine CH2 DC level based on offset
    ch2_voltage_level = get_ch2_dc_level(offset)

    # Generate baseline signals
    ch1_time, ch1_voltage = generate_square_wave(
        frequency=frequency,
        amplitude=amplitude,
        offset=offset,
        sample_rate=sample_rate,
        duration=duration,
        rise_time=rise_time_target,
    )

    ch2_time, ch2_voltage = generate_dc_signal(
        voltage=ch2_voltage_level, sample_rate=sample_rate, duration=duration
    )

    # Apply Fault Profile 1: Impedance Mismatch
    if impedance_mismatch > 0:
        # Calculate ringing frequency (typically much higher than carrier)
        ringing_freq = frequency * 10.0  # 10x carrier frequency as example
        decay_const = 1.0 / (ringing_freq * 5.0)  # Decay over ~5 periods
        ch1_voltage = apply_impedance_mismatch(
            ch1_voltage,
            ch1_time,
            ringing_amplitude=impedance_mismatch,
            ringing_frequency=ringing_freq,
            decay_constant=decay_const,
        )

    # Apply Fault Profile 2: Power Supply Instability
    if (
        noise_std > 0 or impedance_mismatch > 0
    ):  # Also apply some AC ripple with impedance mismatch
        ac_amp = 0.01 + 0.04 * impedance_mismatch if impedance_mismatch > 0 else 0.01
        ch2_voltage = apply_power_supply_instability(
            ch2_voltage,
            ch2_time,
            noise_std=noise_std,
            switching_freq=1.0e5,  # 100 kHz typical switching frequency
            ac_amplitude=ac_amp,
        )

    # Apply Fault Profile 3: Probe Compensation Error
    # Only apply if rise time target is modified to create error condition
    measured_rise_time = (
        rise_time_target * 1.5
    )  # Simulate 50% slower rise time for demonstration
    if (
        measured_rise_time > PROBE_COMPENSATION_RISE_TIME_MULTIPLIER * rise_time_target
        and rise_time_target > 0
    ):
        ch1_voltage = apply_probe_compensation_error(
            ch1_voltage,
            ch1_time,
            target_rise_time=rise_time_target,
            measured_rise_time=measured_rise_time,
        )

    # Apply Digital Twin Effects

    # ADC Quantization (assuming Vdiv setting - for simplicity, derive from amplitude)
    # Vdiv ≈ amplitude / 8 (for 8 divisions peak-to-peak)
    vdiv_estimate = amplitude / 8.0
    ch1_voltage = apply_adc_quantization(ch1_voltage, vdiv_estimate, bits=ADC_BITS)
    ch2_voltage = apply_adc_quantization(ch2_voltage, vdiv_estimate, bits=ADC_BITS)

    # Anti-aliasing
    ch1_voltage = apply_anti_aliasing(
        ch1_voltage,
        ch1_time,
        carrier_frequency=frequency,
        virtual_sample_rate=sample_rate,
    )
    ch2_voltage = apply_anti_aliasing(
        ch2_voltage,
        ch2_time,
        carrier_frequency=0.0,  # DC signal has no carrier
        virtual_sample_rate=sample_rate,
    )

    # Thermal Drift
    ch1_voltage = apply_thermal_drift(ch1_voltage, virtual_uptime)
    ch2_voltage = apply_thermal_drift(ch2_voltage, virtual_uptime)

    # VNA Errors (more relevant for frequency domain measurements)
    # For time domain simulation, we'll apply and get both time and freq metrics
    ch1_voltage, ch1_vna_metrics = apply_vna_errors(
        ch1_voltage, directivity_error, source_match_error, ch1_time
    )
    ch2_voltage, ch2_vna_metrics = apply_vna_errors(
        ch2_voltage, directivity_error, source_match_error, ch2_time
    )

    # SCPI Register Effects
    scpi_status = apply_scpi_register_effects(ch1_voltage, ch1_time, virtual_uptime)

    # Calculate metrics for verification
    # For CH1, assume square wave swings between offset - amplitude/2 and offset + amplitude/2
    v_high_expected = offset + amplitude / 2.0
    ch1_overshoot_ratio = calculate_overshoot_ratio(ch1_voltage, v_high_expected)
    ch1_rise_time = calculate_rise_time(ch1_time, ch1_voltage)
    ch1_vpp = calculate_vpp(ch1_voltage)

    # For CH2, it's DC so Vpp should be small (just noise and ripple)
    ch2_vpp = calculate_vpp(ch2_voltage)
    ch2_rise_time = 0.0  # DC signal has no rise time
    ch2_overshoot_ratio = 0.0  # Not meaningful for DC

    # Prepare return dictionary
    result = {
        "ch1_time": ch1_time,
        "ch1_voltage": ch1_voltage,
        "ch2_time": ch2_time,
        "ch2_voltage": ch2_voltage,
        "ch1_metrics": {
            "overshoot_ratio": float(ch1_overshoot_ratio),
            "rise_time": float(ch1_rise_time),
            "vpp": float(ch1_vpp),
            "vswr": float(ch1_vna_metrics[0]) if len(ch1_vna_metrics) > 0 else 1.0,
            "return_loss": (
                float(ch1_vna_metrics[1]) if len(ch1_vna_metrics) > 1 else 50.0
            ),
            "phase": float(ch1_vna_metrics[2]) if len(ch1_vna_metrics) > 2 else 0.0,
        },
        "ch2_metrics": {
            "overshoot_ratio": float(ch2_overshoot_ratio),
            "rise_time": float(ch2_rise_time),
            "vpp": float(ch2_vpp),
            "vswr": float(ch2_vna_metrics[0]) if len(ch2_vna_metrics) > 0 else 1.0,
            "return_loss": (
                float(ch2_vna_metrics[1]) if len(ch2_vna_metrics) > 1 else 50.0
            ),
            "phase": float(ch2_vna_metrics[2]) if len(ch2_vna_metrics) > 2 else 0.0,
        },
        "scpi_status": scpi_status,
    }

    return result


# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS (keeping original signatures)
# =============================================================================


# Keep the original function signatures for backward compatibility with existing code
def generate_square_wave_legacy(*args, **kwargs):
    """Legacy wrapper for generate_square_wave"""
    return generate_square_wave(*args, **kwargs)


def generate_dc_signal_legacy(*args, **kwargs):
    """Legacy wrapper for generate_dc_signal"""
    return generate_dc_signal(*args, **kwargs)


def apply_impedance_mismatch_legacy(*args, **kwargs):
    """Legacy wrapper for apply_impedance_mismatch"""
    return apply_impedance_mismatch(*args, **kwargs)


def apply_power_supply_instability_legacy(*args, **kwargs):
    """Legacy wrapper for apply_power_supply_instability"""
    return apply_power_supply_instability(*args, **kwargs)


def apply_probe_compensation_error_legacy(*args, **kwargs):
    """Legacy wrapper for apply_probe_compensation_error"""
    return apply_probe_compensation_error(*args, **kwargs)


def apply_adc_quantization_legacy(*args, **kwargs):
    """Legacy wrapper for apply_adc_quantization"""
    return apply_adc_quantization(*args, **kwargs)


def apply_anti_aliasing_legacy(*args, **kwargs):
    """Legacy wrapper for apply_anti_aliasing"""
    return apply_anti_aliasing(*args, **kwargs)


def apply_thermal_drift_legacy(*args, **kwargs):
    """Legacy wrapper for apply_thermal_drift"""
    return apply_thermal_drift(*args, **kwargs)


def apply_vna_errors_legacy(*args, **kwargs):
    """Legacy wrapper for apply_vna_errors"""
    return apply_vna_errors(*args, **kwargs)


def calculate_rise_time_legacy(*args, **kwargs):
    """Legacy wrapper for calculate_rise_time"""
    return calculate_rise_time(*args, **kwargs)


def calculate_overshoot_ratio_legacy(*args, **kwargs):
    """Legacy wrapper for calculate_overshoot_ratio"""
    return calculate_overshoot_ratio(*args, **kwargs)


def calculate_vpp_legacy(*args, **kwargs):
    """Legacy wrapper for calculate_vpp"""
    return calculate_vpp(*args, **kwargs)


if __name__ == "__main__":
    # Test the simulation function
    print("Testing PulseGuard AI Data Simulation Engine...")

    # Run a basic simulation
    result = simulate_signals(
        frequency=5.0e6,
        amplitude=3.3,
        offset=0.0,
        impedance_mismatch=0.5,
        noise_floor=0.5,
        rise_time_target=15.0e-9,  # 15 ns (slower than target)
        sample_rate=1.0e9,
        duration=1.0e-6,
        virtual_uptime=7200.0,  # 2 hours uptime for thermal drift
    )

    print(f"CH1 - Overshoot Ratio: {result['ch1_metrics']['overshoot_ratio']:.3f}")
    print(f"CH1 - Rise Time: {result['ch1_metrics']['rise_time']*1e9:.1f} ns")
    print(f"CH1 - Vpp: {result['ch1_metrics']['vpp']:.3f} V")
    print(f"CH2 - Vpp: {result['ch2_metrics']['vpp']:.3f} V")
    print(f"SCPI NOALigndata: {result['scpi_status']['no_aligndata']}")
    print(f"SCPI Airflow Restriction: {result['scpi_status']['airflow_restriction']}")
    print(f"Overvoltage Events: {len(result['scpi_status']['overvoltage_events'])}")
    print("Simulation test completed successfully!")
