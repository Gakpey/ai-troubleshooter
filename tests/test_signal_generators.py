"""
Test suite for signal generators
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from data_simulator import generate_square_wave, generate_dc_signal


def test_square_wave_basic():
    """Basic test for square wave generation"""
    t, v = generate_square_wave(frequency=1e6, amplitude=2.0, duration=1e-6)
    assert len(t) == len(v)
    assert len(t) > 0


def test_dc_signal_basic():
    """Basic test for DC signal generation"""
    t, v = generate_dc_signal(voltage=3.3, duration=1e-6)
    assert len(t) == len(v)
    assert np.allclose(v, 3.3)


if __name__ == "__main__":
    test_square_wave_basic()
    test_dc_signal_basic()
    print("Signal generator tests passed!")
