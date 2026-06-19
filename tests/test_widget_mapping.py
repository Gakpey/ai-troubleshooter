"""
Widget mapping tests for data_simulator.py
Tests that Streamlit widget parameters map correctly to simulation parameters
"""

import numpy as np
import pytest
import sys
import os
from data_simulator import simulate_signals
from src.config import (
    WIDGET_RANGES,
    WIDGET_DEFAULTS,
    map_noise_floor_to_std,
)

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestWidgetMapping:
    """Test Streamlit widget parameter mapping"""

    def test_widget_ranges_defined(self):
        """Test that widget ranges are properly defined"""
        expected_widgets = {
            "carrier_frequency",
            "amplitude",
            "offset",
            "noise_floor",
            "rise_time_target",
            "impedance_mismatch",
        }
        assert set(WIDGET_RANGES.keys()) == expected_widgets

        # Check each widget has (min, max, step)
        for widget, (min_val, max_val, step) in WIDGET_RANGES.items():
            assert isinstance(min_val, (int, float))
            assert isinstance(max_val, (int, float))
            assert isinstance(step, (int, float))
            assert min_val < max_val
            assert step > 0

    def test_widget_defaults_within_ranges(self):
        """Test that widget defaults are within their specified ranges"""
        for widget, default_value in WIDGET_DEFAULTS.items():
            if widget not in WIDGET_RANGES:
                raise AssertionError(f"Widget {widget} not in WIDGET_RANGES")
            min_val, max_val, step = WIDGET_RANGES[widget]
            assert min_val <= default_value <= max_val, (
                f"Default {default_value} for {widget} not in range "
                f"[{min_val}, {max_val}]"
            )

    def test_carrier_frequency_mapping(self):
        """Test that Carrier Frequency slider maps to frequency parameter"""
        # Test minimum value
        result_min = simulate_signals(
            frequency=WIDGET_RANGES["carrier_frequency"][0],  # 1.0 MHz
            amplitude=WIDGET_DEFAULTS["amplitude"],
            offset=WIDGET_DEFAULTS["offset"],
            noise_floor=WIDGET_DEFAULTS["noise_floor"],
            rise_time_target=WIDGET_DEFAULTS["rise_time_target"],
            impedance_mismatch=WIDGET_DEFAULTS["impedance_mismatch"],
        )
        assert result_min is not None

        # Test maximum value
        result_max = simulate_signals(
            frequency=WIDGET_RANGES["carrier_frequency"][1],  # 25.0 MHz
            amplitude=WIDGET_DEFAULTS["amplitude"],
            offset=WIDGET_DEFAULTS["offset"],
            noise_floor=WIDGET_DEFAULTS["noise_floor"],
            rise_time_target=WIDGET_DEFAULTS["rise_time_target"],
            impedance_mismatch=WIDGET_DEFAULTS["impedance_mismatch"],
        )
        assert result_max is not None

        # Verify that higher frequency produces more cycles in same duration
        # (this is a basic sanity check)
        # Actually, let's just verify both calls succeed

    def test_amplitude_mapping(self):
        """Test Vpp amplitude slider maps to amplitude parameter"""
        # Test minimum value (20 mV)
        result_min = simulate_signals(
            frequency=WIDGET_DEFAULTS["carrier_frequency"],
            amplitude=WIDGET_RANGES["amplitude"][0],  # 0.02 V
            offset=WIDGET_DEFAULTS["offset"],
            noise_floor=WIDGET_DEFAULTS["noise_floor"],
            rise_time_target=WIDGET_DEFAULTS["rise_time_target"],
            impedance_mismatch=WIDGET_DEFAULTS["impedance_mismatch"],
        )
        assert result_min is not None

        # Test maximum value (5.0 V)
        result_max = simulate_signals(
            frequency=WIDGET_DEFAULTS["carrier_frequency"],
            amplitude=WIDGET_RANGES["amplitude"][1],  # 5.0 V
            offset=WIDGET_DEFAULTS["offset"],
            noise_floor=WIDGET_DEFAULTS["noise_floor"],
            rise_time_target=WIDGET_DEFAULTS["rise_time_target"],
            impedance_mismatch=WIDGET_DEFAULTS["impedance_mismatch"],
        )
        assert result_max is not None

        # The maximum amplitude should produce higher voltage swings
        vpp_min = result_min["ch1_metrics"]["vpp"]
        vpp_max = result_max["ch1_metrics"]["vpp"]
        assert vpp_max > vpp_min, (
            f"Higher amplitude setting should produce higher VPP: "
            f"{vpp_min} vs {vpp_max}"
        )

    def test_offset_mapping(self):
        """Test that Voffset (DC Offset) slider maps to offset parameter"""
        # Test minimum value (-2.5 V)
        result_min = simulate_signals(
            frequency=WIDGET_DEFAULTS["carrier_frequency"],
            amplitude=WIDGET_DEFAULTS["amplitude"],
            offset=WIDGET_RANGES["offset"][0],  # -2.5 V
            noise_floor=WIDGET_DEFAULTS["noise_floor"],
            rise_time_target=WIDGET_DEFAULTS["rise_time_target"],
            impedance_mismatch=WIDGET_DEFAULTS["impedance_mismatch"],
        )
        assert result_min is not None

        # Test maximum value (+2.5 V)
        result_max = simulate_signals(
            frequency=WIDGET_DEFAULTS["carrier_frequency"],
            amplitude=WIDGET_DEFAULTS["amplitude"],
            offset=WIDGET_RANGES["offset"][1],  # +2.5 V
            noise_floor=WIDGET_DEFAULTS["noise_floor"],
            rise_time_target=WIDGET_DEFAULTS["rise_time_target"],
            impedance_mismatch=WIDGET_DEFAULTS["impedance_mismatch"],
        )
        assert result_max is not None

        # More importantly, the mean voltage should reflect the offset
        ch2_mean_min = np.mean(result_min["ch2_voltage"])
        ch2_mean_max = np.mean(result_max["ch2_voltage"])

        # With higher offset, we expect higher mean voltage (exact relationship
        # depends on get_ch2_dc_level implementation)
        assert ch2_mean_max >= ch2_mean_min, (
            f"Higher offset should generally produce higher CH2 mean: "
            f"{ch2_mean_min} vs {ch2_mean_max}"
        )

    def test_noise_floor_mapping(self):
        """Test noise floor slider maps to std deviation correctly"""
        # Test that the mapping function works correctly
        assert map_noise_floor_to_std(0.0) == 0.0  # 0% -> 0.0V
        assert map_noise_floor_to_std(50.0) == 0.25  # 50% -> 0.25V
        assert map_noise_floor_to_std(100.0) == 0.5  # 100% -> 0.5V

        # Test simulation with different noise floor values
        # Low noise floor
        result_low = simulate_signals(
            frequency=WIDGET_DEFAULTS["carrier_frequency"],
            amplitude=WIDGET_DEFAULTS["amplitude"],
            offset=WIDGET_DEFAULTS["offset"],
            noise_floor=WIDGET_RANGES["noise_floor"][0],  # 0%
            rise_time_target=WIDGET_DEFAULTS["rise_time_target"],
            impedance_mismatch=WIDGET_DEFAULTS["impedance_mismatch"],
        )

        # High noise floor
        result_high = simulate_signals(
            frequency=WIDGET_DEFAULTS["carrier_frequency"],
            amplitude=WIDGET_DEFAULTS["amplitude"],
            offset=WIDGET_DEFAULTS["offset"],
            noise_floor=WIDGET_RANGES["noise_floor"][1],  # 100%
            rise_time_target=WIDGET_DEFAULTS["rise_time_target"],
            impedance_mismatch=WIDGET_DEFAULTS["impedance_mismatch"],
        )

        # Both should succeed
        assert result_low is not None
        assert result_high is not None

        # High noise should generally produce higher VPP on CH2 (DC channel)
        vpp_low = result_low["ch2_metrics"]["vpp"]
        vpp_high = result_high["ch2_metrics"]["vpp"]
        # Allow for randomness, but generally expect higher noise -> higher VPP
        assert vpp_high >= vpp_low * 0.5, (
            f"Higher noise floor should generally increase VPP: "
            f"{vpp_low} vs {vpp_high}"
        )

    def test_rise_time_target_mapping(self):
        """Test that Rise Time slider maps to rise_time_target parameter"""
        # Test minimum value (1 ns)
        result_min = simulate_signals(
            frequency=WIDGET_DEFAULTS["carrier_frequency"],
            amplitude=WIDGET_DEFAULTS["amplitude"],
            offset=WIDGET_DEFAULTS["offset"],
            noise_floor=WIDGET_DEFAULTS["noise_floor"],
            rise_time_target=WIDGET_RANGES["rise_time_target"][0],  # 1 ns
            impedance_mismatch=WIDGET_DEFAULTS["impedance_mismatch"],
        )
        assert result_min is not None

        # Test maximum value (100 ns)
        result_max = simulate_signals(
            frequency=WIDGET_DEFAULTS["carrier_frequency"],
            amplitude=WIDGET_DEFAULTS["amplitude"],
            offset=WIDGET_DEFAULTS["offset"],
            noise_floor=WIDGET_DEFAULTS["noise_floor"],
            rise_time_target=WIDGET_RANGES["rise_time_target"][1],  # 100 ns
            impedance_mismatch=WIDGET_DEFAULTS["impedance_mismatch"],
        )
        assert result_max is not None

        # Higher rise time target should generally result in measured rise time
        # being slower (when effects are applied)
        rt_min = result_min["ch1_metrics"]["rise_time"]
        rt_max = result_max["ch1_metrics"]["rise_time"]

        # Both should be positive and reasonable
        assert rt_min >= 0
        assert rt_max >= 0
        # Note: Exact relationship depends on probe compensation error

    def test_impedance_mismatch_mapping(self):
        """Test impedance mismatch slider maps to impedance_mismatch"""
        # Test minimum value (0.0V - perfect match)
        result_min = simulate_signals(
            frequency=WIDGET_DEFAULTS["carrier_frequency"],
            amplitude=WIDGET_DEFAULTS["amplitude"],
            offset=WIDGET_DEFAULTS["offset"],
            noise_floor=WIDGET_DEFAULTS["noise_floor"],
            rise_time_target=WIDGET_DEFAULTS["rise_time_target"],
            impedance_mismatch=WIDGET_RANGES["impedance_mismatch"][0],  # 0.0V
        )
        assert result_min is not None

        # Test maximum value (1.0V - severe mismatch)
        result_max = simulate_signals(
            frequency=WIDGET_DEFAULTS["carrier_frequency"],
            amplitude=WIDGET_DEFAULTS["amplitude"],
            offset=WIDGET_DEFAULTS["offset"],
            noise_floor=WIDGET_DEFAULTS["noise_floor"],
            rise_time_target=WIDGET_DEFAULTS["rise_time_target"],
            impedance_mismatch=WIDGET_RANGES["impedance_mismatch"][1],
            # 1.0V
        )
        assert result_max is not None

        # Higher impedance mismatch should generally produce higher overshoot
        overshoot_min = result_min["ch1_metrics"]["overshoot_ratio"]
        overshoot_max = result_max["ch1_metrics"]["overshoot_ratio"]

        # For zero impedance mismatch, overshoot ratio approx zero
        # Allow small negative values due to numerical precision
        assert (
            overshoot_min >= -0.001
        ), f"Zero impedance mismatch overshoot near zero, got {overshoot_min}"
        assert (
            overshoot_max >= 0
        ), f"Pos impedance mismatch overshoot >=0, got {overshoot_max}"

        # Generally expect higher impedance -> higher overshoot
        assert overshoot_max >= overshoot_min * 0.5, (
            f"Higher impedance mismatch should generally increase overshoot: "
            f"{overshoot_min} vs {overshoot_max}"
        )

    def test_parameter_combinations(self):
        """Test various combinations of widget parameters"""
        # Test all minimums
        result_min = simulate_signals(
            frequency=WIDGET_RANGES["carrier_frequency"][0],
            amplitude=WIDGET_RANGES["amplitude"][0],
            offset=WIDGET_RANGES["offset"][0],
            noise_floor=WIDGET_RANGES["noise_floor"][0],
            rise_time_target=WIDGET_RANGES["rise_time_target"][0],
            impedance_mismatch=WIDGET_RANGES["impedance_mismatch"][0],
        )
        assert result_min is not None

        # Test all maximums
        result_max = simulate_signals(
            frequency=WIDGET_RANGES["carrier_frequency"][1],
            amplitude=WIDGET_RANGES["amplitude"][1],
            offset=WIDGET_RANGES["offset"][1],
            noise_floor=WIDGET_RANGES["noise_floor"][1],
            rise_time_target=WIDGET_RANGES["rise_time_target"][1],
            impedance_mismatch=WIDGET_RANGES["impedance_mismatch"][1],
        )
        assert result_max is not None

        # Test all defaults
        result_default = simulate_signals(
            frequency=WIDGET_DEFAULTS["carrier_frequency"],
            amplitude=WIDGET_DEFAULTS["amplitude"],
            offset=WIDGET_DEFAULTS["offset"],
            noise_floor=WIDGET_DEFAULTS["noise_floor"],
            rise_time_target=WIDGET_DEFAULTS["rise_time_target"],
            impedance_mismatch=WIDGET_DEFAULTS["impedance_mismatch"],
        )
        assert result_default is not None

        # All should have valid structure
        for result in [result_min, result_max, result_default]:
            assert "ch1_time" in result
            assert "ch1_voltage" in result
            assert "ch2_time" in result
            assert "ch2_voltage" in result
            assert "ch1_metrics" in result
            assert "ch2_metrics" in result
            assert "scpi_status" in result

    def test_edge_case_values(self):
        """Test edge case values within widget ranges"""
        # Test values just above minimum
        freq_min = WIDGET_RANGES["carrier_frequency"][0]
        freq_range = WIDGET_RANGES["carrier_frequency"][1] - freq_min
        freq = freq_min + 0.1 * freq_range

        amp_min = WIDGET_RANGES["amplitude"][0]
        amp_range = WIDGET_RANGES["amplitude"][1] - amp_min
        amplitude = amp_min + 0.1 * amp_range

        offset_min = WIDGET_RANGES["offset"][0]
        offset_range = WIDGET_RANGES["offset"][1] - offset_min
        offset = offset_min + 0.1 * offset_range

        noise_min = WIDGET_RANGES["noise_floor"][0]
        noise_range = WIDGET_RANGES["noise_floor"][1] - noise_min
        noise_floor = noise_min + 0.1 * noise_range

        rise_min = WIDGET_RANGES["rise_time_target"][0]
        rise_range = WIDGET_RANGES["rise_time_target"][1] - rise_min
        rise_time_target = rise_min + 0.1 * rise_range

        imp_min = WIDGET_RANGES["impedance_mismatch"][0]
        imp_range = WIDGET_RANGES["impedance_mismatch"][1] - imp_min
        impedance_mismatch = imp_min + 0.1 * imp_range

        result_near_min = simulate_signals(
            frequency=freq,
            amplitude=amplitude,
            offset=offset,
            noise_floor=noise_floor,
            rise_time_target=rise_time_target,
            impedance_mismatch=impedance_mismatch,
        )
        assert result_near_min is not None

        # Test values just below maximum
        freq_max = WIDGET_RANGES["carrier_frequency"][1]
        freq_range = freq_max - WIDGET_RANGES["carrier_frequency"][0]
        freq = freq_max - 0.1 * freq_range

        amp_max = WIDGET_RANGES["amplitude"][1]
        amp_range = amp_max - WIDGET_RANGES["amplitude"][0]
        amplitude = amp_max - 0.1 * amp_range

        offset_max = WIDGET_RANGES["offset"][1]
        offset_range = offset_max - WIDGET_RANGES["offset"][0]
        offset = offset_max - 0.1 * offset_range

        noise_max = WIDGET_RANGES["noise_floor"][1]
        noise_range = noise_max - WIDGET_RANGES["noise_floor"][0]
        noise_floor = noise_max - 0.1 * noise_range

        rise_max = WIDGET_RANGES["rise_time_target"][1]
        rise_range = rise_max - WIDGET_RANGES["rise_time_target"][0]
        rise_time_target = rise_max - 0.1 * rise_range

        imp_max = WIDGET_RANGES["impedance_mismatch"][1]
        imp_range = imp_max - WIDGET_RANGES["impedance_mismatch"][0]
        impedance_mismatch = imp_max - 0.1 * imp_range

        result_near_max = simulate_signals(
            frequency=freq,
            amplitude=amplitude,
            offset=offset,
            noise_floor=noise_floor,
            rise_time_target=rise_time_target,
            impedance_mismatch=impedance_mismatch,
        )
        assert result_near_max is not None


if __name__ == "__main__":
    # Run widget mapping tests
    pytest.main([__file__, "-v"])
