"""
Anomaly Detector for PulseGuard AI

Implements the Anomaly Detection & Feature Isolation Layer as specified in
@specs.md section 3.2 and @CLAUDE.md.

Design Decisions (locked):
1. ML features use CH1 only: raw amplitude + rolling variance (window=10 samples).
2. Rolling variance window is strictly integer size = 10 samples.
3. High-state block detection for overshoot uses epsilon = 0.01 * amplitude.

The detector combines:
- Isolation Forest (n_estimators=100, contamination=0.03, random_state=42)
- Deterministic rules:
    * Overshoot %: localized peak search on logical '1' blocks (CH1)
    * Vpp: delta bounds on DC arrays (CH2)
    * Rise time: 10% -> 90% amplitude crossing delta-t (CH1)

Fault is declared if ML anomaly OR any deterministic rule threshold is exceeded.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import IsolationForest
import logging
from config import get_ch2_dc_level

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Hybrid anomaly detector combining Isolation Forest and deterministic rules.

    Parameters
    ----------
    contamination : float, default=0.03
        Proportion of outliers in the data set.
    n_estimators : int, default=100
        Number of base estimators in the ensemble.
    random_state : int, default=42
        Seed for random number generator.
    window_size : int, default=10
        Size of rolling window for variance feature (in samples).
    """

    # Class-level cache for the Isolation Forest model (fitted on baseline normal signal)
    _isolation_forest: Optional[IsolationForest] = None
    _fitted: bool = False

    def __init__(
        self,
        contamination: float = 0.03,
        n_estimators: int = 50,
        random_state: int = 42,
        window_size: int = 10,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.window_size = window_size

        # Thresholds from config
        from config import (
            DEFAULT_FREQUENCY,
            DEFAULT_AMPLITUDE,
            DEFAULT_OFFSET,
            DEFAULT_RISE_TIME_TARGET,
            DEFAULT_SAMPLE_RATE,
            DEFAULT_DURATION,
            DEFAULT_IMPEDANCE_MISMATCH,
            DEFAULT_NOISE_FLOOR,
            IMPEDANCE_MISMATCH_OVERSHOOT_THRESHOLD,
            POWER_SUPPLI_VPP_THRESHOLD,
            PROBE_COMPENSATION_RISE_TIME_MULTIPLIER,
            CH2_DC_LEVEL_ERROR_THRESHOLD,
        )

        self.overshoot_threshold = IMPEDANCE_MISMATCH_OVERSHOOT_THRESHOLD
        self.vpp_threshold = POWER_SUPPLI_VPP_THRESHOLD
        self.rise_time_multiplier = PROBE_COMPENSATION_RISE_TIME_MULTIPLIER
        self.ch2_dc_level_error_threshold = CH2_DC_LEVEL_ERROR_THRESHOLD

        # Store defaults for baseline generation
        self._default_freq = DEFAULT_FREQUENCY
        self._default_amp = DEFAULT_AMPLITUDE
        self._default_offset = DEFAULT_OFFSET
        self._default_rise = DEFAULT_RISE_TIME_TARGET
        self._default_sr = DEFAULT_SAMPLE_RATE
        self._default_dur = DEFAULT_DURATION
        self._default_impedance_mismatch = DEFAULT_IMPEDANCE_MISMATCH
        self._default_noise_floor = DEFAULT_NOISE_FLOOR

        # Cache for the last feature vector (for debugging/inspection)
        self._last_feature_vector: Optional[np.ndarray] = None

        # Fit the model on a baseline normal signal (only once per class)
        if not AnomalyDetector._fitted:
            AnomalyDetector._isolation_forest = self._fit_baseline_model()
            AnomalyDetector._fitted = True

        # Use the cached model
        self._isolation_forest = AnomalyDetector._isolation_forest

    def _generate_baseline_signal(self) -> Dict[str, np.ndarray]:
        """Generate a baseline normal signal with default parameters and no faults."""
        from data_simulator import simulate_signals

        return simulate_signals(
            frequency=self._default_freq,
            amplitude=self._default_amp,
            offset=self._default_offset,
            impedance_mismatch=0.0,
            noise_floor=0.0,
            rise_time_target=self._default_rise,
            sample_rate=self._default_sr,
            duration=self._default_dur,
            virtual_uptime=0.0,
        )

    def _fit_baseline_model(self) -> IsolationForest:
        """Fit the Isolation Forest on a baseline normal signal."""
        baseline = self._generate_baseline_signal()
        ch1_voltage = baseline["ch1_voltage"]
        feature_vector = self._engineer_features(ch1_voltage)
        isolation_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        isolation_forest.fit(feature_vector)
        logger.debug(
            "Isolation Forest fitted on baseline %d samples", feature_vector.shape[0]
        )
        return isolation_forest

    def detect_anomalies(
        self,
        ch1_time: np.ndarray,
        ch1_voltage: np.ndarray,
        ch2_time: np.ndarray,
        ch2_voltage: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Detect anomalies in the signal data.

        Parameters
        ----------
        ch1_time : np.ndarray
            Time values for CH1 (square wave) in seconds.
        ch1_voltage : np.ndarray
            Voltage values for CH1 in volts.
        ch2_time : np.ndarray
            Time values for CH2 (DC rail) in seconds.
        ch2_voltage : np.ndarray
            Voltage values for CH2 in volts.
        params : dict, optional
            Dictionary of simulation parameters (frequency, amplitude, offset,
            rise_time_target) used for adaptive thresholding.

        Returns
        -------
        dict
            Dictionary with keys:
            - is_fault : bool
            - ml_anomaly_score : float
            - computed_metrics : dict
            - fault_reasons : list of str
            - feature_vector : np.ndarray
        """
        # Initialize return structure
        result: Dict[str, Any] = {
            "is_fault": False,
            "ml_anomaly_score": 0.0,
            "computed_metrics": {
                "ch1": {
                    "overshoot_ratio": 0.0,
                    "rise_time": 0.0,
                    "vpp": 0.0,
                },
                "ch2": {
                    "overshoot_ratio": 0.0,
                    "rise_time": 0.0,
                    "vpp": 0.0,
                    "dc_level_error": 0.0,
                    "expected_dc_level": 0.0,
                    "actual_dc_level": 0.0,
                },
            },
            "fault_reasons": [],
            "feature_vector": None,
        }

        # Validate inputs
        if len(ch1_voltage) == 0 or len(ch2_voltage) == 0:
            logger.warning("Empty voltage arrays provided")
            return result

        # Use default params if not provided
        if params is None:
            params = {
                "frequency": self._default_freq,
                "amplitude": self._default_amp,
                "offset": self._default_offset,
                "rise_time_target": self._default_rise,
            }

        # Step 1: Compute deterministic metrics
        ch1_metrics = self._compute_ch1_metrics(ch1_time, ch1_voltage, params)
        ch2_metrics = self._compute_ch2_metrics(ch2_time, ch2_voltage, params)

        result["computed_metrics"]["ch1"] = ch1_metrics
        result["computed_metrics"]["ch2"] = ch2_metrics

        # Step 2: Check deterministic rules
        self._check_deterministic_rules(ch1_metrics, ch2_metrics, result, params)

        # Step 3: Compute ML anomaly score (using pre-fitted model)
        feature_vector = self._engineer_features(ch1_voltage)
        result["feature_vector"] = feature_vector

        # Subsample features for fast Isolation Forest inference.
        # For repetitive signals (e.g. 5 MHz square wave over 1 µs),
        # a representative subsample is sufficient.  Target ~2000 samples.
        fv_len = feature_vector.shape[0]
        if fv_len > 2000:
            step = fv_len // 2000
            fv_sub = feature_vector[::step]
        else:
            fv_sub = feature_vector

        # score_samples returns per-sample anomaly scores (lower = more anomalous).
        ml_scores = self._isolation_forest.score_samples(fv_sub)
        result["ml_anomaly_score"] = float(np.mean(ml_scores))

        # Predict labels on the same subsample (1 = inlier, -1 = outlier).
        ml_prediction = self._isolation_forest.predict(fv_sub)
        outlier_proportion = np.mean(ml_prediction == -1)
        ml_anomaly_detected = outlier_proportion > 0.10

        if ml_anomaly_detected:
            result["is_fault"] = True
            result["fault_reasons"].append("isolation_forest_anomaly")

        # Add active parameters for diagnostic agent
        result["active_parameters"] = {
            "frequency": params.get("frequency", self._default_freq),
            "amplitude": params.get("amplitude", self._default_amp),
            "offset": params.get("offset", self._default_offset),
            "noise_floor": params.get("noise_floor", self._default_noise_floor),
            "rise_time_target": params.get("rise_time_target", self._default_rise),
            "impedance_mismatch": params.get(
                "impedance_mismatch", self._default_impedance_mismatch
            ),
        }

        return result

    # ------------------------------------------------------------------
    # CH1 metrics
    # ------------------------------------------------------------------

    def _compute_ch1_metrics(
        self,
        time_array: np.ndarray,
        voltage_array: np.ndarray,
        params: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute CH1-specific metrics: overshoot, rise time, Vpp."""
        vpp = float(np.ptp(voltage_array))
        overshoot_ratio = self._compute_overshoot_ratio(
            time_array, voltage_array, params
        )
        rise_time = self._compute_rise_time(time_array, voltage_array, params)
        return {
            "overshoot_ratio": overshoot_ratio,
            "rise_time": rise_time,
            "vpp": vpp,
        }

    # ------------------------------------------------------------------
    # CH2 metrics
    # ------------------------------------------------------------------

    def _compute_ch2_metrics(
        self,
        time_array: np.ndarray,
        voltage_array: np.ndarray,
        params: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute CH2-specific metrics: Vpp and DC level error."""
        vpp = float(np.ptp(voltage_array))
        # Calculate DC level error: difference between actual mean and expected level
        offset = float(params.get("offset", self._default_offset))
        expected_dc_level = get_ch2_dc_level(offset)
        actual_dc_level = float(np.mean(voltage_array))
        dc_level_error = abs(actual_dc_level - expected_dc_level)

        return {
            "overshoot_ratio": 0.0,
            "rise_time": 0.0,
            "vpp": vpp,
            "dc_level_error": dc_level_error,
            "expected_dc_level": expected_dc_level,
            "actual_dc_level": actual_dc_level,
        }

    # ------------------------------------------------------------------
    # Overshoot ratio
    # ------------------------------------------------------------------

    def _compute_overshoot_ratio(
        self,
        time_array: np.ndarray,
        voltage_array: np.ndarray,
        params: Dict[str, Any],
    ) -> float:
        """
        Compute overshoot ratio as localized peak search on logical '1' blocks.

        Implements specs.md: Overshoot Ratio = (Vmax - Vhigh) / Vamplitude
        Threshold for fault detection: Overshoot Ratio > 0.10

        We find each LOW→HIGH transition (using 10% and 90% thresholds) and
        search a post-transition window (e.g., next 150-250 samples) for the
        local maximum voltage. The overshoot is calculated relative to the
        expected high state from params.

        Returns:
            Overshoot ratio (float)
        """
        if len(voltage_array) < 2 or len(time_array) < 2:
            return 0.0

        # Expected high and low from params
        amplitude = float(params.get("amplitude", self._default_amp))
        offset = float(params.get("offset", self._default_offset))
        v_high_expected = offset + amplitude / 2.0
        v_low_expected = offset - amplitude / 2.0
        v_amplitude_expected = amplitude  # peak-to-peak

        if v_amplitude_expected <= 0:
            return 0.0

        # 10% and 90% thresholds for rise time calculation
        v_low_thresh = v_low_expected + 0.1 * v_amplitude_expected  # 10% point
        v_high_thresh = v_low_expected + 0.9 * v_amplitude_expected  # 90% point

        # Find rising edges: crossing of 10% threshold going up
        low_crossings = np.where(
            (voltage_array[:-1] <= v_low_thresh) & (voltage_array[1:] > v_low_thresh)
        )[0]
        # Find rising edges: crossing of 90% threshold going up
        high_crossings = np.where(
            (voltage_array[:-1] <= v_high_thresh) & (voltage_array[1:] > v_high_thresh)
        )[0]

        if len(low_crossings) == 0 or len(high_crossings) == 0:
            return 0.0

        max_overshoot = 0.0
        # Time step
        dt = time_array[1] - time_array[0]
        # Window after 90% point to search for peak (150 ns)
        window_time = 150.0e-9
        window_samples = max(1, int(window_time / dt))

        # We'll pair each low crossing with the next high crossing (same edge)
        for low_idx in low_crossings:
            # Find the first high crossing after this low crossing
            high_after = high_crossings[high_crossings > low_idx]
            if len(high_after) == 0:
                continue
            high_idx = high_after[0]

            # Search window for local maximum after the 90% point
            start_idx = high_idx
            end_idx = min(len(voltage_array), high_idx + window_samples)
            if end_idx <= start_idx:
                continue

            # Find the maximum voltage in the window
            window_voltage = voltage_array[start_idx:end_idx]
            local_max = np.max(window_voltage)

            # Overshoot = (local_max - settled_high) / amplitude
            # Use expected high state as the settled high
            overshoot = (local_max - v_high_expected) / v_amplitude_expected
            if overshoot > max_overshoot:
                max_overshoot = overshoot

        return float(max(max_overshoot, 0.0))  # Ensure non-negative

    # ------------------------------------------------------------------
    # Rise time
    # ------------------------------------------------------------------

    def _compute_rise_time(
        self,
        time_array: np.ndarray,
        voltage_array: np.ndarray,
        params: Dict[str, Any],
    ) -> float:
        """
        Compute rise time as dt between 10% and 90% thresholds.

        Uses linear interpolation for sub-sample precision.
        Uses expected signal levels based on params, not actual min/max,
        to avoid distortion from faults like overshoot or ringing.
        """
        if len(time_array) < 2 or len(voltage_array) < 2:
            return 0.0

        # Expected high and low from params
        amplitude = float(params.get("amplitude", self._default_amp))
        offset = float(params.get("offset", self._default_offset))
        v_low_expected = offset - amplitude / 2.0
        v_high_expected = offset + amplitude / 2.0
        v_amplitude_expected = amplitude  # peak-to-peak

        if v_amplitude_expected <= 0:
            return 0.0

        # 10% and 90% thresholds for rise time calculation
        v_low_thresh = v_low_expected + 0.1 * v_amplitude_expected  # 10% point
        v_high_thresh = v_low_expected + 0.9 * v_amplitude_expected  # 90% point

        # Find first crossing of v_low on rising edge
        low_cross = np.where(
            (voltage_array[:-1] <= v_low_thresh) & (voltage_array[1:] > v_low_thresh)
        )[0]
        if len(low_cross) == 0:
            return 0.0

        first_low = low_cross[0]

        # Find first crossing of v_high after first_low
        high_cross = np.where(
            (voltage_array[:-1] <= v_high_thresh) & (voltage_array[1:] > v_high_thresh)
        )[0]
        high_after = high_cross[high_cross > first_low]
        if len(high_after) == 0:
            return 0.0

        first_high = high_after[0]

        # Linear interpolation for precise crossing times
        t_low = float(
            np.interp(
                v_low_thresh,
                [voltage_array[first_low], voltage_array[first_low + 1]],
                [time_array[first_low], time_array[first_low + 1]],
            )
        )
        t_high = float(
            np.interp(
                v_high_thresh,
                [voltage_array[first_high], voltage_array[first_high + 1]],
                [time_array[first_high], time_array[first_high + 1]],
            )
        )

        return float(max(t_high - t_low, 0.0))

    # ------------------------------------------------------------------
    # Feature engineering  (CH1 only, window=10)
    # ------------------------------------------------------------------

    def _engineer_features(self, ch1_voltage: np.ndarray) -> np.ndarray:
        """
        Engineer features for Isolation Forest: raw amplitude + rolling variance.

        Locked decision: CH1 only, window size = 10 samples.

        Parameters
        ----------
        ch1_voltage : np.ndarray
            CH1 voltage signal.

        Returns
        -------
        np.ndarray
            Feature array of shape (n_samples, 2):
                column 0: raw amplitude
                column 1: rolling variance (window=10)
        """
        n = len(ch1_voltage)
        raw = ch1_voltage.astype(np.float64)

        # ---- Vectorised rolling variance using cumsum ----
        # Use a strict forward-looking window of size window_size for each
        # position i: samples [i, i+window_size).  At the end, window
        # shrinks.
        w = self.window_size
        cumsum = np.empty(n + 1, dtype=np.float64)
        cumsum_sq = np.empty(n + 1, dtype=np.float64)
        cumsum[0] = 0.0
        cumsum_sq[0] = 0.0
        np.cumsum(ch1_voltage, out=cumsum[1:])
        np.cumsum(ch1_voltage**2, out=cumsum_sq[1:])

        # For each position i, the window covers [i, i+w) (clamped to n)
        # window_sum  = cumsum[i+w] - cumsum[i]
        # window_sum2 = cumsum_sq[i+w] - cumsum_sq[i]
        ends = np.minimum(np.arange(n) + w, n)
        win_n = ends - np.arange(n)
        win_n = np.maximum(win_n, 1)  # at least 1 sample

        # Compute sums and sums-of-squares for each window
        win_sum = cumsum[ends] - cumsum[np.arange(n)]
        win_sum2 = cumsum_sq[ends] - cumsum_sq[np.arange(n)]

        # Variance = E[X^2] - (E[X])^2
        win_mean = win_sum / win_n
        rolling_var = (win_sum2 / win_n) - win_mean**2
        # Clamp floating-point negatives to 0
        np.maximum(rolling_var, 0.0, out=rolling_var)

        feature_vector = np.column_stack((raw, rolling_var))
        self._last_feature_vector = feature_vector
        return feature_vector

    # ------------------------------------------------------------------
    # Deterministic rules
    # ------------------------------------------------------------------

    def _check_deterministic_rules(
        self,
        ch1_metrics: Dict[str, float],
        ch2_metrics: Dict[str, float],
        result: Dict[str, Any],
        params: Dict[str, Any],
    ) -> None:
        """Check deterministic rules and update result with fault reasons."""
        # Overshoot threshold
        if ch1_metrics["overshoot_ratio"] > self.overshoot_threshold:
            result["is_fault"] = True
            result["fault_reasons"].append("overshoot_threshold_exceeded")

        # Vpp threshold on CH2 (power supply instability)
        if ch2_metrics["vpp"] > self.vpp_threshold:
            result["is_fault"] = True
            result["fault_reasons"].append("power_supply_vpp_exceeded")

        # CH2 DC level error threshold
        if ch2_metrics["dc_level_error"] > self.ch2_dc_level_error_threshold:
            result["is_fault"] = True
            result["fault_reasons"].append("ch2_dc_level_error_exceeded")

        # Rise time threshold on CH1
        target_rise = float(params.get("rise_time_target", self._default_rise))
        limit = target_rise * self.rise_time_multiplier
        if ch1_metrics["rise_time"] > limit:
            result["is_fault"] = True
            result["fault_reasons"].append("rise_time_threshold_exceeded")
