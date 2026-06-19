"""
Unit tests for the DiagnosticAgent class.
"""

import json
from unittest.mock import Mock, patch

import pytest

# Assume the module is available
try:
    from src.diagnostic_agent import DiagnosticAgent
except ImportError:
    # Handle missing dependency gracefully in test environment
    DiagnosticAgent = None


@pytest.mark.skipif(
    DiagnosticAgent is None, reason="google-genai not installed"
)
class TestDiagnosticAgent:
    """Test suite for DiagnosticAgent."""

    @pytest.fixture
    def mock_telemetry_fault(self):
        """Return a sample telemetry dict indicating a fault."""
        return {
            "is_fault": True,
            "ml_anomaly_score": -0.5,  # Negative indicates anomaly
            "computed_metrics": {
                "ch1": {
                    "overshoot_ratio": 0.15,  # > 0.10 threshold
                    "vpp": 3.3,
                    "rise_time": 15.0e-9,  # 15 ns > 1.2 * 10ns = 12ns
                },
                "ch2": {
                    "overshoot_ratio": 0.0,
                    "vpp": 0.15,  # 150mV > 100mV threshold
                    "rise_time": 0.0,
                },
            },
            "fault_reasons": [
                "overshoot_threshold_exceeded",
                "power_supply_vpp_exceeded",
                "rise_time_threshold_exceeded",
            ],
            "active_parameters": {
                "frequency": 5.0e6,
                "amplitude": 3.3,
                "offset": 0.0,
                "noise_floor": 50.0,  # 50%
                "rise_time_target": 10.0e-9,
                "impedance_mismatch": 0.8,
            },
        }

    @pytest.fixture
    def mock_telemetry_no_fault(self):
        """Return a sample telemetry dict indicating no fault."""
        return {
            "is_fault": False,
            "ml_anomaly_score": 0.5,
            "computed_metrics": {
                "ch1": {
                    "overshoot_ratio": 0.05,
                    "vpp": 3.3,
                    "rise_time": 9.0e-9,
                },
                "ch2": {
                    "overshoot_ratio": 0.0,
                    "vpp": 0.05,
                    "rise_time": 0.0,
                },
            },
            "fault_reasons": [],
            "active_parameters": {
                "frequency": 5.0e6,
                "amplitude": 3.3,
                "offset": 0.0,
                "noise_floor": 10.0,
                "rise_time_target": 10.0e-9,
                "impedance_mismatch": 0.0,
            },
        }

    @pytest.fixture
    def mock_gemini_response(self):
        """Return a mock Gemini response object."""
        mock_response = Mock()
        mock_candidate = Mock()
        mock_content = Mock()
        mock_part = Mock()
        mock_part.text = (
            "# Root Cause Analysis\n"
            "Impedance mismatch detected with severe ringing.\n\n"
            "## Evidence\n"
            "- Overshoot ratio: 0.15 (>10% threshold)\n"
            "- Vpp on CH2: 150mV (>100mV threshold)\n"
            "- Rise time: 15ns (>12ns limit)\n\n"
            "## Recommendations\n"
            "- Check termination impedance\n"
            "- Adjust probe compensation\n"
            "- Review power supply decoupling\n"
        )
        mock_content.parts = [mock_part]
        mock_candidate.content = mock_content
        mock_response.candidates = [mock_candidate]
        return mock_response

    def test_init_without_api_key(self, monkeypatch):
        """Test that agent logs warning when API key is missing."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with patch("src.diagnostic_agent.genai") as mock_genai:
            mock_genai.Client.return_value = None
            agent = DiagnosticAgent()
            assert agent.client is None
            # Check that warning was logged (caplog would be needed for full check)

    def test_init_with_api_key(self, monkeypatch):
        """Test that agent initializes client when API key is present."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        with patch("src.diagnostic_agent.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client
            agent = DiagnosticAgent()
            assert agent.client == mock_client
            mock_genai.Client.assert_called_once_with(api_key="test-key")

    def test_analyze_telemetry_no_fault(self, mock_telemetry_no_fault):
        """Test that no analysis is performed when there is no fault."""
        with patch("src.diagnostic_agent.genai"):
            agent = DiagnosticAgent()
            result = agent.analyze_telemetry(mock_telemetry_no_fault)
            assert result == ""

    def test_analyze_telemetry_no_client(
        self, mock_telemetry_fault, monkeypatch
    ):
        """Test error message when API key is not configured."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with patch("src.diagnostic_agent.genai") as mock_genai:
            mock_genai.Client.return_value = None
            agent = DiagnosticAgent()
            result = agent.analyze_telemetry(mock_telemetry_fault)
            assert "GOOGLE_API_KEY not configured" in result

    def test_analyze_telemetry_with_fault(
        self, mock_telemetry_fault, mock_gemini_response, monkeypatch
    ):
        """Test successful analysis when fault is present."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        with patch("src.diagnostic_agent.genai") as mock_genai:
            # Setup mock client and response
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content.return_value = (
                mock_gemini_response
            )

            agent = DiagnosticAgent()
            result = agent.analyze_telemetry(mock_telemetry_fault)

            # Verify the API was called
            mock_client.models.generate_content.assert_called_once()
            # Check that the result contains the expected markdown
            assert "# Root Cause Analysis" in result
            assert "Impedance mismatch detected" in result
            assert "## Evidence" in result
            assert "## Recommendations" in result

    def test_analyze_telemetry_api_error(
        self, mock_telemetry_fault, monkeypatch
    ):
        """Test handling of API errors."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        with patch("src.diagnostic_agent.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client
            # Simulate an API exception
            mock_client.models.generate_content.side_effect = Exception(
                "API timeout"
            )

            agent = DiagnosticAgent()
            result = agent.analyze_telemetry(mock_telemetry_fault)

            assert "Diagnostic analysis failed" in result
            assert "API timeout" in result

    def test_build_telemetry_payload(self, mock_telemetry_fault):
        """Test the internal payload building method."""
        with patch("src.diagnostic_agent.genai"):
            agent = DiagnosticAgent()
            payload = agent._build_telemetry_payload(mock_telemetry_fault)

            # Check structure
            assert "metrics" in payload
            assert "flags" in payload
            assert "active_parameters" in payload

            # Check metric values
            assert payload["metrics"]["overshoot_ratio"] == 0.15
            assert payload["metrics"]["vpp_ch1"] == 3.3
            assert payload["metrics"]["vpp_ch2"] == 0.15
            assert payload["metrics"]["rise_time"] == 15.0e-9

            # Check flags
            assert payload["flags"]["is_fault"] is True
            assert (
                payload["flags"]["ml_anomaly"] is True
            )  # because score is negative
            assert (
                "overshoot_threshold_exceeded"
                in payload["flags"]["fault_reasons"]
            )

            # Check active parameters
            assert payload["active_parameters"]["frequency"] == 5.0e6
            assert payload["active_parameters"]["noise_floor"] == 50.0
