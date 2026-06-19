"""
Test suite for digital twin effects
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from data_simulator import (
    apply_adc_quantization,
    apply_anti_aliasing,
    apply_thermal_drift,
    apply_vna_errors,
)


def test_adc_quantization_basic():
    """Basic test for ADC quantization"""
    signal = np.array([-1.0, 0.0, 1.0])
    result = apply_adc_quantization(signal, 0.5, bits=10)
    assert len(result) == len(signal)


def test_anti_aliasing_basic():
    """Basic test for anti-aliasing"""
    signal = np.ones(100)
    time = np.arange(100) * 1e-9
    result = apply_anti_aliasing(signal, time, 1e6, 1e9)
    assert len(result) == len(signal)


def test_thermal_drift_basic():
    """Basic test for thermal drift"""
    signal = np.array([1.0, 2.0, 3.0])
    result = apply_thermal_drift(signal, 0.0)
    assert len(result) == len(signal)


def test_vna_errors_basic():
    """Basic test for VNA errors"""
    signal = np.array([1.0, 2.0, 3.0])
    result = apply_vna_errors(signal, 0.01 + 0.01j, 0.005 - 0.005j)
    assert isinstance(result, tuple)
    assert len(result) == 2


if __name__ == "__main__":
    test_adc_quantization_basic()
    test_anti_aliasing_basic()
    test_thermal_drift_basic()
    test_vna_errors_basic()
    print("Digital twin effects tests passed!")
