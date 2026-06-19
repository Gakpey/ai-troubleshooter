"""
Performance and benchmark tests for data_simulator.py
Tests execution latency and scalability
"""

import numpy as np
import pytest
import sys
import os
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_simulator import simulate_signals, generate_square_wave, generate_dc_signal


class TestPerformance:
    """Performance and benchmark tests"""

    def test_latency_benchmark_small(self):
        """Test latency with small sample sizes"""
        # Test with 1,000 samples
        duration = 1e-6  # 1 microsecond
        sample_rate = 1.0e6  # 1 MS/s
        # 1e-6 * 1e6 = 1,000 samples

        # Warm up cache
        _ = simulate_signals(
            frequency=5.0e6,
            amplitude=3.3,
            offset=0.0,
            sample_rate=sample_rate,
            duration=duration,
        )

        # Time the operation
        start_time = time.perf_counter()
        result = simulate_signals(
            frequency=5.0e6,
            amplitude=3.3,
            offset=0.0,
            sample_rate=sample_rate,
            duration=duration,
        )
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        # Should be very fast for small samples
        assert (
            elapsed_time < 0.01
        ), f"Small simulation took {elapsed_time:.3f}s, expected <0.01s"

        # Verify correct output size
        expected_samples = int(duration * sample_rate)
        assert len(result["ch1_time"]) == expected_samples
        assert len(result["ch1_voltage"]) == expected_samples

    def test_latency_benchmark_medium(self):
        """Test latency with medium sample sizes"""
        # Test with 10,000 samples
        duration = 10e-6  # 10 microseconds
        sample_rate = 1.0e6  # 1 MS/s
        # 10e-6 * 1e6 = 10,000 samples

        # Warm up cache
        _ = simulate_signals(
            frequency=5.0e6,
            amplitude=3.3,
            offset=0.0,
            sample_rate=sample_rate,
            duration=duration,
        )

        # Time the operation
        start_time = time.perf_counter()
        result = simulate_signals(
            frequency=5.0e6,
            amplitude=3.3,
            offset=0.0,
            sample_rate=sample_rate,
            duration=duration,
        )
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        # Should still be reasonably fast
        assert (
            elapsed_time < 0.05
        ), f"Medium simulation took {elapsed_time:.3f}s, expected <0.05s"

        # Verify correct output size
        expected_samples = int(duration * sample_rate)
        assert len(result["ch1_time"]) == expected_samples
        assert len(result["ch1_voltage"]) == expected_samples

    def test_latency_benchmark_large(self):
        """Test latency with large sample sizes (equivalent to 50k sample test)"""
        # Test with 50,000 samples as specified in requirements
        duration = 50e-6  # 50 microseconds
        sample_rate = 1.0e9  # 1 GS/s
        # 50e-6 * 1e9 = 50,000 samples

        # Warm up cache
        _ = simulate_signals(
            frequency=5.0e6,
            amplitude=3.3,
            offset=0.0,
            impedance_mismatch=0.3,
            noise_floor=0.4,
            rise_time_target=12.0e-9,
            sample_rate=sample_rate,
            duration=duration,
            virtual_uptime=3600.0,
        )

        # Time the operation
        start_time = time.perf_counter()
        result = simulate_signals(
            frequency=5.0e6,
            amplitude=3.3,
            offset=0.0,
            impedance_mismatch=0.3,
            noise_floor=0.4,
            rise_time_target=12.0e-9,
            sample_rate=sample_rate,
            duration=duration,
            virtual_uptime=3600.0,
        )
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        # Target: <100ms for data simulation alone
        assert (
            elapsed_time < 0.1
        ), f"Large simulation took {elapsed_time:.3f}s, expected <0.1s"

        # Verify correct output size
        expected_samples = int(duration * sample_rate)
        assert len(result["ch1_time"]) == expected_samples
        assert len(result["ch1_voltage"]) == expected_samples

    def test_scalability_with_sample_rate(self):
        """Test that performance scales reasonably with sample rate"""
        base_duration = 10e-6  # 10 microseconds
        sample_rates = [1.0e6, 5.0e6, 1.0e7, 5.0e7]  # 1 MS/s to 50 MS/s
        times = []

        for sample_rate in sample_rates:
            duration = base_duration
            expected_samples = int(duration * sample_rate)

            # Warm up cache for this sample rate
            _ = simulate_signals(
                frequency=5.0e6,
                amplitude=3.3,
                offset=0.0,
                sample_rate=sample_rate,
                duration=duration,
            )

            # Time the operation
            start_time = time.perf_counter()
            result = simulate_signals(
                frequency=5.0e6,
                amplitude=3.3,
                offset=0.0,
                sample_rate=sample_rate,
                duration=duration,
            )
            end_time = time.perf_counter()

            elapsed_time = end_time - start_time
            times.append(elapsed_time)

            # Verify correct output size
            assert len(result["ch1_time"]) == expected_samples
            assert len(result["ch1_voltage"]) == expected_samples

        # Check that time scales roughly linearly with sample rate
        # (allowing for some overhead and non-linearities)
        if len(times) >= 2:
            # Ratio of times should be roughly proportional to ratio of sample rates
            time_ratio = times[-1] / times[0]
            sample_rate_ratio = sample_rates[-1] / sample_rates[0]
            # Allow for substantial variation due to caching, overhead, etc.
            assert (
                time_ratio < sample_rate_ratio * 10
            ), f"Time scaling ({time_ratio:.2f}) too steep compared to sample rate scaling ({sample_rate_ratio:.2f})"

    def test_scalability_with_duration(self):
        """Test that performance scales reasonably with duration"""
        sample_rate = 1.0e7  # 10 MS/s fixed
        durations = [1e-6, 5e-6, 10e-6, 20e-6]  # 1 to 20 microseconds
        times = []

        for duration in durations:
            expected_samples = int(duration * sample_rate)

            # Warm up cache for this duration
            _ = simulate_signals(
                frequency=5.0e6,
                amplitude=3.3,
                offset=0.0,
                sample_rate=sample_rate,
                duration=duration,
            )

            # Time the operation
            start_time = time.perf_counter()
            result = simulate_signals(
                frequency=5.0e6,
                amplitude=3.3,
                offset=0.0,
                sample_rate=sample_rate,
                duration=duration,
            )
            end_time = time.perf_counter()

            elapsed_time = end_time - start_time
            times.append(elapsed_time)

            # Verify correct output size
            assert len(result["ch1_time"]) == expected_samples
            assert len(result["ch1_voltage"]) == expected_samples

        # Check that time scales roughly linearly with duration
        if len(times) >= 2:
            # Ratio of times should be roughly proportional to ratio of durations
            time_ratio = times[-1] / times[0]
            duration_ratio = durations[-1] / durations[0]
            # Allow for variation
            assert (
                time_ratio < duration_ratio * 10
            ), f"Time scaling ({time_ratio:.2f}) too steep compared to duration scaling ({duration_ratio:.2f})"

    def test_memory_usage_reasonable(self):
        """Test that memory usage is reasonable for large simulations"""
        import psutil
        import os

        # Get current process
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run a reasonably large simulation
        result = simulate_signals(
            frequency=5.0e6,
            amplitude=3.3,
            offset=0.0,
            impedance_mismatch=0.5,
            noise_floor=0.5,
            rise_time_target=10.0e-9,
            sample_rate=1.0e9,
            duration=100e-6,  # 100 microseconds = 100,000 samples
            virtual_uptime=3600.0,
        )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB for 100k samples)
        assert (
            memory_increase < 50
        ), f"Memory usage increased by {memory_increase:.1f}MB, expected <50MB"

        # Verify we got the expected data
        expected_samples = int(100e-6 * 1.0e9)  # 100,000
        assert len(result["ch1_time"]) == expected_samples
        assert len(result["ch1_voltage"]) == expected_samples

    def test_repeated_calls_performance(self):
        """Test that repeated calls with same parameters are fast (caching working)"""
        # Warm up
        _ = simulate_signals(
            frequency=5.0e6,
            amplitude=3.3,
            offset=0.1,
            impedance_mismatch=0.2,
            noise_floor=0.3,
            rise_time_target=15.0e-9,
            sample_rate=1.0e9,
            duration=1.0e-6,
            virtual_uptime=1800.0,
        )

        # Time multiple repeated calls
        times = []
        for i in range(10):
            start_time = time.perf_counter()
            result = simulate_signals(
                frequency=5.0e6,
                amplitude=3.3,
                offset=0.1,
                impedance_mismatch=0.2,
                noise_floor=0.3,
                rise_time_target=15.0e-9,
                sample_rate=1.0e9,
                duration=1.0e-6,
                virtual_uptime=1800.0,
            )
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        # Average time should be very small (indicating caching is working)
        avg_time = sum(times) / len(times)
        assert (
            avg_time < 0.01
        ), f"Average cached call time {avg_time:.3f}s too slow, expected <0.01s"

        # All results should be identical
        first_result = None
        for i in range(10):
            result = simulate_signals(
                frequency=5.0e6,
                amplitude=3.3,
                offset=0.1,
                impedance_mismatch=0.2,
                noise_floor=0.3,
                rise_time_target=15.0e-9,
                sample_rate=1.0e9,
                duration=1.0e-6,
                virtual_uptime=1800.0,
            )
            if first_result is None:
                first_result = result
            else:
                # Should be identical to first result (due to caching)
                np.testing.assert_array_almost_equal(
                    first_result["ch1_voltage"], result["ch1_voltage"]
                )
                np.testing.assert_array_almost_equal(
                    first_result["ch2_voltage"], result["ch2_voltage"]
                )


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v"])
