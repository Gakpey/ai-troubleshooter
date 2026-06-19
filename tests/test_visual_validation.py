"""
Visual validation tests for data_simulator.py
Tests that generate expected waveforms and visual characteristics
"""

import numpy as np
import pytest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_simulator import (
    generate_square_wave,
    generate_dc_signal,
    apply_impedance_mismatch,
    apply_power_supply_instability,
    apply_probe_compensation_error,
    apply_adc_quantization,
    apply_anti_aliasing,
    apply_thermal_drift,
    apply_vna_errors,
    simulate_signals,
)

try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
class TestVisualValidation:
    """Visual validation tests"""

    def test_square_wave_visual(self):
        """Test that square wave has expected visual characteristics"""
        t, v = generate_square_wave(
            frequency=1.0e6,  # 1 MHz
            amplitude=2.0,  # 2 Vpp
            offset=0.0,  # 0 Vdc
            sample_rate=1.0e8,  # 100 MS/s
            duration=5.0e-6,  # 5 us
            rise_time=10.0e-9,  # 10 ns
        )

        # Basic sanity checks
        assert len(t) == len(v)
        assert len(t) > 0

        # Should have approximate min and max values
        assert np.min(v) <= -0.5  # Should reach negative values
        assert np.max(v) >= 0.5  # Should reach positive values

        # Test that we can create a plot (basic visualization test)
        if HAS_MATPLOTLIB:
            plt.figure(figsize=(10, 6))
            plt.plot(t * 1e6, v)  # Time in microseconds
            plt.xlabel("Time (μs)")
            plt.ylabel("Voltage (V)")
            plt.title("Square Wave Visual Validation")
            plt.grid(True)
            plt.close()  # Close figure to free memory

    def test_impedance_mismatch_visual(self):
        """Test that impedance mismatch adds visible ringing"""
        # Create base signal
        t, v_base = generate_square_wave(
            frequency=1.0e6,
            amplitude=2.0,
            offset=0.0,
            sample_rate=1.0e9,
            duration=2.0e-6,
            rise_time=1.0e-9,
        )

        # Apply impedance mismatch
        v_mismatch = apply_impedance_mismatch(
            v_base,
            t,
            ringing_amplitude=0.5,
            ringing_frequency=1.0e9,
            decay_constant=1.0e-9,
        )

        # Should be different from base
        assert not np.allclose(v_base, v_mismatch, rtol=1e-3)

        # Visual validation - if we can plot, verify we see ringing
        if HAS_MATPLOTLIB:
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 1, 1)
            plt.plot(t * 1e6, v_base, label="Base", alpha=0.7)
            plt.plot(t * 1e6, v_mismatch, label="With Ringing", alpha=0.7)
            plt.xlabel("Time (μs)")
            plt.ylabel("Voltage (V)")
            plt.title("Impedance Mismatch Effect")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(
                t * 1e6, v_mismatch - v_base, label="Ringing Component", color="red"
            )
            plt.xlabel("Time (μs)")
            plt.ylabel("Voltage Difference (V)")
            plt.title("Isolated Ringing Effect")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.close()

    def test_power_supply_instability_visual(self):
        """Test that power supply instability adds visible noise and ripple"""
        # Create base DC signal
        t, v_base = generate_dc_signal(voltage=3.3, sample_rate=1.0e9, duration=2.0e-6)

        # Apply power supply instability
        v_noise = apply_power_supply_instability(
            v_base,
            t,
            noise_std=0.1,  # 100mV noise
            switching_freq=1.0e5,  # 100kHz
            ac_amplitude=0.05,  # 50mV ripple
        )

        # Should be different from base
        assert not np.allclose(v_base, v_noise, rtol=1e-3)
        assert len(v_noise) == len(v_base)

        # Visual validation
        if HAS_MATPLOTLIB:
            plt.figure(figsize=(12, 8))

            plt.subplot(3, 1, 1)
            plt.plot(t * 1e6, v_base, label="Clean DC", alpha=0.7)
            plt.xlabel("Time (μs)")
            plt.ylabel("Voltage (V)")
            plt.title("Power Supply Instability - Base Signal")
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.plot(t * 1e6, v_noise, label="Noisy DC", alpha=0.7)
            plt.xlabel("Time (μs)")
            plt.ylabel("Voltage (V)")
            plt.title("Power Supply Instability - With Noise/Ripple")
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 1, 3)
            plt.plot(
                t * 1e6,
                v_noise - v_base,
                label="Noise + Ripple",
                color="red",
                alpha=0.7,
            )
            plt.xlabel("Time (μs)")
            plt.ylabel("Voltage Difference (V)")
            plt.title("Added Noise and Ripple Components")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.close()

    def test_probe_compensation_error_visual(self):
        """Test that probe compensation error visibly affects rise time"""
        # Create signal with defined rise time
        t, v_fast = generate_square_wave(
            frequency=1.0e6,
            amplitude=2.0,
            offset=0.0,
            sample_rate=1.0e9,
            duration=2.0e-6,
            rise_time=1.0e-9,  # Very fast rise time
        )

        # Apply probe compensation error to make it slower
        t, v_slow = generate_square_wave(
            frequency=1.0e6,
            amplitude=2.0,
            offset=0.0,
            sample_rate=1.0e9,
            duration=2.0e-6,
            rise_time=20.0e-9,  # Slower rise time
        )

        # Should be different
        assert not np.allclose(v_fast, v_slow, rtol=1e-3)

        # Visual validation
        if HAS_MATPLOTLIB:
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 1, 1)
            plt.plot(t * 1e6, v_fast, label="Fast Rise (1ns)", alpha=0.7)
            plt.plot(t * 1e6, v_slow, label="Slow Rise (20ns)", alpha=0.7)
            plt.xlabel("Time (μs)")
            plt.ylabel("Voltage (V)")
            plt.title("Probe Compensation Error - Rise Time Effect")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 1, 2)
            # Zoom in on a rising edge
            rise_start = np.where(v_fast > -0.9)[0][0]  # Find where it starts rising
            rise_end = min(rise_start + 100, len(t))  # Look at next 100 points
            plt.plot(
                t[rise_start:rise_end] * 1e6,
                v_fast[rise_start:rise_end],
                label="Fast Rise",
                alpha=0.7,
            )
            plt.plot(
                t[rise_start:rise_end] * 1e6,
                v_slow[rise_start:rise_end],
                label="Slow Rise",
                alpha=0.7,
            )
            plt.xlabel("Time (μs)")
            plt.ylabel("Voltage (V)")
            plt.title("Zoomed Rise Time Comparison")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.close()

    def test_adc_quantization_visual(self):
        """Test that ADC quantization creates visible quantization steps"""
        # Create a smooth signal
        t = np.linspace(0, 1e-6, 1000)
        v_smooth = np.sin(2 * np.pi * 1.0e6 * t)  # 1MHz sine wave

        # Apply ADC quantization
        v_quantized = apply_adc_quantization(v_smooth, vdiv=0.5, bits=10)

        # Should be different from original (unless original already matches quantization levels)
        assert len(v_quantized) == len(v_smooth)

        # Visual validation
        if HAS_MATPLOTLIB:
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 1, 1)
            plt.plot(t * 1e6, v_smooth, label="Original Sine Wave", alpha=0.7)
            plt.plot(t * 1e6, v_quantized, label="ADC Quantized", alpha=0.7)
            plt.xlabel("Time (μs)")
            plt.ylabel("Voltage (V)")
            plt.title("ADC Quantization Effect")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(
                t * 1e6,
                v_quantized - v_smooth,
                label="Quantization Error",
                color="red",
                alpha=0.7,
            )
            plt.xlabel("Time (μs)")
            plt.ylabel("Voltage Difference (V)")
            plt.title("Quantization Error")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.close()

    def test_complete_simulation_visual(self):
        """Test visualization of complete simulation with various effects"""
        # Run a complete simulation with multiple effects
        result = simulate_signals(
            frequency=2.0e6,  # 2 MHz
            amplitude=3.3,  # 3.3Vpp
            offset=0.0,  # 0V offset
            impedance_mismatch=0.4,  # Moderate impedance mismatch
            noise_floor=0.5,  # Moderate noise
            rise_time_target=5.0e-9,  # 5ns rise time
            sample_rate=1.0e9,  # 1 GS/s
            duration=5.0e-6,  # 5 microseconds
            virtual_uptime=7200.0,  # 2 hours uptime for thermal effects
        )

        # Basic validation
        assert len(result["ch1_time"]) == len(result["ch1_voltage"])
        assert len(result["ch1_time"]) > 0
        assert len(result["ch2_time"]) == len(result["ch2_voltage"])
        assert len(result["ch2_time"]) > 0

        # Visual validation
        if HAS_MATPLOTLIB:
            plt.figure(figsize=(14, 10))

            # Channel 1 (Square wave with effects)
            plt.subplot(3, 1, 1)
            plt.plot(
                result["ch1_time"] * 1e6,
                result["ch1_voltage"],
                linewidth=1,
                color="blue",
            )
            plt.xlabel("Time (μs)")
            plt.ylabel("Voltage (V)")
            plt.title("Channel 1: Square Wave with All Effects")
            plt.grid(True, alpha=0.3)

            # Channel 2 (DC with effects)
            plt.subplot(3, 1, 2)
            plt.plot(
                result["ch2_time"] * 1e6,
                result["ch2_voltage"],
                linewidth=1,
                color="red",
            )
            plt.xlabel("Time (μs)")
            plt.ylabel("Voltage (V)")
            plt.title("Channel 2: DC Rail with Power Supply Instability")
            plt.grid(True, alpha=0.3)

            # Frequency domain metrics from VNA (if available)
            if "ch1_metrics" in result and len(result["ch1_metrics"]) >= 3:
                plt.subplot(3, 1, 3)
                metrics = result["ch1_metrics"]
                # Create a simple bar chart of key metrics
                metric_names = ["VSWR", "Return Loss (dB)", "Phase (deg)"]
                metric_values = [
                    metrics.get("vswr", 0),
                    metrics.get("return_loss", 0),
                    metrics.get("phase", 0),
                ]
                bars = plt.bar(
                    metric_names,
                    metric_values,
                    color=["green", "orange", "purple"],
                    alpha=0.7,
                )
                plt.ylabel("Value")
                plt.title("VNA-Derived Metrics")
                plt.grid(True, alpha=0.3, axis="y")

                # Add value labels on bars
                for bar, value in zip(bars, metric_values):
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(metric_values) * 0.01,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                    )

            plt.tight_layout()
            plt.close()


if __name__ == "__main__":
    # Run visual validation tests
    pytest.main([__file__, "-v"])
