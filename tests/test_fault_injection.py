"""
Test suite for fault injection
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from data_simulator import (
    apply_impedance_mismatch,
    apply_power_supply_instability,
    apply_probe_compensation_error,
)


def test_impedance_mismatch_basic():
    """Basic test for impedance mismatch"""
    signal = np.zeros(100)
    time = np.arange(100) * 1e-9
    result = apply_impedance_mismatch(signal, time, 0.5, 1e9, 1e-9)
    assert len(result) == len(signal)


def test_power_supply_instability_basic():
    """Basic test for power supply instability"""
    signal = np.full(100, 3.3)
    time = np.arange(100) * 1e-9
    result = apply_power_supply_instability(signal, time, 0.1, 1e5, 0.05)
    assert len(result) == len(signal)


def test_probe_compensation_error_basic():
    """Basic test for probe compensation error"""
    signal = np.array([0, 0, 1, 1, 1, 0, 0])
    time = np.arange(7) * 1e-9
    result = apply_probe_compensation_error(signal, time, 2e-9, 3e-9)
    assert len(result) == len(signal)


if __name__ == "__main__":
    test_impedance_mismatch_basic()
    test_power_supply_instability_basic()
    test_probe_compensation_error_basic()
    print("Fault injection tests passed!")
