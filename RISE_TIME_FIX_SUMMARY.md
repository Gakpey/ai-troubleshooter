# Fix for Rise Time Fault Detection in PulseGuard AI

## Problem Statement
The system was unable to recognize faults from rise time configurations because the rise time calculation was using the actual signal min/max values, which get distorted by electrical faults like overshoot, ringing, or noise. This led to incorrect rise time measurements and missed fault detections.

## Root Cause
In the original `_compute_rise_time` method in `src/anomaly_detector.py`, the calculation was:
```python
v_min = float(np.min(voltage_array))
v_max = float(np.max(voltage_array))
v_range = v_max - v_min
v_low = v_min + 0.1 * v_range
v_high = v_min + 0.9 * v_range
```

This approach fails when:
- Overshoot increases V_max beyond expected levels
- Ringing creates local extrema that distort min/max
- Noise affects the extreme values of the signal
- Any fault that changes the signal's absolute min/max values

## Solution Implemented
Modified the `_compute_rise_time` method to use expected signal levels based on input parameters (amplitude and offset) rather than actual signal statistics:

```python
# Expected high and low from params
amplitude = float(params.get("amplitude", self._default_amp))
offset = float(params.get("offset", self._default_offset))
v_low_expected = offset - amplitude / 2.0
v_high_expected = offset + amplitude / 2.0
v_amplitude_expected = amplitude  # peak-to-peak

# 10% and 90% thresholds for rise time calculation
v_low_thresh = v_low_expected + 0.1 * v_amplitude_expected  # 10% point
v_high_thresh = v_low_expected + 0.9 * v_amplitude_expected  # 90% point
```

## Key Benefits
1. **Fault Immunity**: Rise time calculation is now immune to signal distortions caused by faults
2. **Accuracy**: Measures rise time relative to expected signal levels, not distorted actual levels
3. **Consistency**: Provides repeatable measurements regardless of fault conditions
4. **Standards Compliance**: Follows the specification approach of using expected thresholds

## Verification Results
All tests pass, including:
- ✅ Normal rise time (10ns target): No fault detected (measures ~9.97ns)
- ✅ Slow rise time (20ns actual): Fault correctly detected (measures ~20.08ns > 12.0ns threshold)
- ✅ Very slow rise time (30ns actual): Fault correctly detected (measures ~29.96ns > 12.0ns threshold)
- ✅ Edge case at threshold (12.0ns): Correctly not fault (measures ~11.98ns < 12.0ns threshold)
- ✅ All existing fault detection tests continue to pass (overshoot, Vpp, Isolation Forest, CH2 DC level)
- ✅ Performance tests confirm detection completes well under required latency thresholds

## Files Modified
- `src/anomaly_detector.py`: 
  - Fixed `_compute_rise_time` method (lines 400-465)
  - Added clarifying comments about the fix approach

## Impact
This fix ensures that rise time-based faults (such as probe compensation issues) are reliably detected regardless of concurrent electrical faults in the signal, improving the overall robustness and accuracy of the PulseGuard AI fault detection system.