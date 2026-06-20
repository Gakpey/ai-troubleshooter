# Summary of Fixes for PulseGuard AI Fault Detection Issues

## Issue 1: Rise Time Fault Detection Not Working Properly
**Problem**: The system was unable to recognize faults from rise time configurations because the rise time calculation was using the actual min/max of the signal, which can be distorted by faults like overshoot or ringing.

**Solution**: Modified `_compute_rise_time` method in `src/anomaly_detector.py` to:
- Use expected signal levels based on parameters (amplitude and offset) instead of actual signal min/max
- Calculate thresholds relative to the expected high/low states:
  - v_low_thresh = v_low_expected + 0.1 * v_amplitude_expected  (10% point)
  - v_high_thresh = v_low_expected + 0.9 * v_amplitude_expected  (90% point)
- This ensures accurate rise time measurement even when faults distort the signal

## Issue 2: DC Offset Fault Detection Missing (TTD - Test Driven Development)
**Problem**: The system lacked test-driven development for detecting DC offset faults, specifically incorrect CH2 DC levels.

**Solution**: Implemented CH2 DC level error detection:

### Changes to `src/config.py`:
- Added `CH2_DC_LEVEL_ERROR_THRESHOLD = 0.15` (>150mV error threshold)

### Changes to `src/anomaly_detector.py`:
- Added import: `from config import get_ch2_dc_level`
- Enhanced `__init__` method to store `ch2_dc_level_error_threshold`
- Enhanced `_compute_ch2_metrics` method to:
  - Calculate expected DC level using `get_ch2_dc_level(offset)`
  - Calculate actual DC level as mean of `ch2_voltage`
  - Compute DC level error = |actual - expected|
  - Return additional metrics: `dc_level_error`, `expected_dc_level`, `actual_dc_level`
- Updated result structure to include new CH2 DC level metrics
- Added check in `_check_deterministic_rules` for:
  ```python
  if ch2_metrics["dc_level_error"] > self.ch2_dc_level_error_threshold:
      result["is_fault"] = True
      result["fault_reasons"].append("ch2_dc_level_error_exceeded")
  ```

### Changes to `tests/test_anomaly_detector.py`:
- Added `generate_ch2_dc_level_error_signal()` function that creates a signal with incorrect CH2 DC level
- Added `TestCh2DCLevelErrorDetection` test class with:
  - `test_ch2_dc_level_error_triggers_fault()`: Verifies that signals with incorrect CH2 DC level are properly detected as faults

## Verification
- All existing tests continue to pass (13/13)
- New test for CH2 DC level error detection passes
- Rise time fault detection test continues to pass
- Manual import tests confirm no syntax errors or import issues

## Files Modified
1. `src/anomaly_detector.py` - Fixed rise time calculation, added CH2 DC level error detection
2. `src/config.py` - Added CH2 DC level error threshold and imported get_ch2_dc_level
3. `tests/test_anomaly_detector.py` - Added test for CH2 DC level error detection

These fixes ensure that:
1. Rise time faults are accurately detected regardless of signal distortions
2. CH2 DC level faults are detectable with appropriate test coverage (addressing the TTd requirement)
3. All existing functionality remains intact