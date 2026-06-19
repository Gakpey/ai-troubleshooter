"""
Test suite for verification functions
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from data_simulator import calculate_rise_time, calculate_overshoot_ratio, calculate_vpp


def test_calculate_vpp_basic():
    """Basic test for VPP calculation"""
    signal = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    vpp = calculate_vpp(signal)
    assert np.isclose(vpp, 4.0)


def test_calculate_overshoot_ratio_basic():
    """Basic test for overshoot ratio calculation"""
    signal = np.array([0.0, 3.0, 3.5, 3.3])
    ratio = calculate_overshoot_ratio(signal, 3.3)
    # Vmax=3.5, Vhigh=3.3, Vmin=0.0
    # Vamplitude=3.5, overshoot=0.2, ratio=0.2/3.5
    expected = 0.2 / 3.5
    assert np.isclose(ratio, expected)


def test_calculate_rise_time_basic():
    """Basic test for rise time calculation"""
    # Linear ramp from 0 to 1V over 10ns
    # Use matching array lengths
    n_points = 21
    time = np.linspace(0, 20e-9, n_points)
    # First half: 0V, Second half: linear ramp to 1V
    voltage = np.concatenate(
        [
            np.zeros(n_points // 2 + 1),  # 0V for first half
            np.linspace(0, 1.0, n_points // 2),  # 0V to 1V for second half
        ]
    )
    rise_time = calculate_rise_time(time, voltage)
    assert np.isclose(rise_time, 10.0e-9, rtol=1e-5)


if __name__ == "__main__":
    test_calculate_vpp_basic()
    test_calculate_overshoot_ratio_basic()
    test_calculate_rise_time_basic()
    print("Verification tests passed!")
