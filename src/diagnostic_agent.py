"""
Diagnostic Agent for PulseGuard AI

Implements the Inline Diagnostics Agent as specified in
@specs.md section 3.3 and @CLAUDE.md.

The DiagnosticAgent class uses the Google Gemini SDK to generate
root-cause analysis and corrective lab configurations based on
telemetry data from the anomaly detector.
"""

import os
import json
import logging
from typing import Dict, Any

# Import the Google Generative AI client
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

logger = logging.getLogger(__name__)

# Default timeout for Gemini API calls in seconds
DEFAULT_TIMEOUT_SECONDS = 10.0

# System prompt for the Gemini model to act as a
# Rohde & Schwarz Field Applications Engineer
SYSTEM_PROMPT = """
You are a senior Field Applications Engineer at Rohde & Schwarz with expertise in
oscilloscope signal analysis, electrical fault diagnosis, and test equipment troubleshooting.
Your task is to analyze the provided telemetry data from a digital twin system that
simulates R&S RTB2000 oscilloscope traces and identify the root cause of any detected faults.

Given the following telemetry JSON payload containing computed metrics, anomaly flags,
and active virtual hardware parameters, provide a structured markdown analysis that includes:

1. **Root Cause Analysis**: Identify the most likely electrical fault based on the
   telemetry (e.g., impedance mismatch, power supply instability, probe compensation
   error, or combination thereof).

2. **Evidence Summary**: Reference specific values from the telemetry that support
   your diagnosis (e.g., overshoot ratio > 10%, Vpp > 100mV on DC rail, rise time
   exceeding 1.2x target).

3. **Recommended Corrective Lab Configurations**: Suggest specific adjustments to
   the test setup or equipment configuration to mitigate or eliminate the fault,
   such as:
   - Adjusting probe compensation settings
   - Changing impedance matching or termination
   - Checking power supply stability and decoupling
   - Modifying signal generator settings (frequency, amplitude, rise time)
   - Verifying grounding and shielding

4. **Additional Notes**: Any other observations or recommendations for further
   investigation if needed.

Format your response in clear, concise markdown suitable for display in a Streamlit
application. Be direct and actionable in your recommendations.
"""


class DiagnosticAgent:
    """
    Diagnostic agent that uses Google Gemini SDK for fault analysis.

    Parameters
    ----------
    model_name : str, default="gemini-2.5-flash"
        Name of the Gemini model to use.
    timeout_seconds : float, default=10.0
        Timeout for API calls in seconds.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        if genai is None:
            raise ImportError(
                "google-genai is not installed. Please install it with: pip install google-genai>=0.3.0"
            )

        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

        # Initialize the Gemini client
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning(
                "GOOGLE_API_KEY environment variable not set. "
                "Diagnostic agent will not be able to call the Gemini API."
            )
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)

    def analyze_telemetry(self, telemetry: Dict[str, Any]) -> str:
        """
        Analyze telemetry data and return a markdown root-cause analysis.

        Parameters
        ----------
        telemetry : dict
            Telemetry data from the anomaly detector, expected to contain:
            - is_fault: bool
            - computed_metrics: dict with ch1 and ch2 metrics
            - fault_reasons: list of str
            - active_parameters: dict of simulation parameters

        Returns
        -------
        str
            Markdown-formatted analysis and recommendations, or empty string if no fault.
        """
        # Check if we have a fault to analyze
        if not telemetry.get("is_fault", False):
            logger.info(
                "No fault detected in telemetry; skipping diagnostic analysis."
            )
            return ""

        # Check if we have a valid Gemini client
        if self.client is None:
            error_msg = (
                "Diagnostic analysis unavailable: GOOGLE_API_KEY not configured. "
                "Please set the environment variable to enable AI-powered insights."
            )
            logger.error(error_msg)
            return error_msg

        try:
            # Build the minified JSON payload as per specs
            payload = self._build_telemetry_payload(telemetry)

            # Convert payload to JSON string (minified)
            payload_json = json.dumps(payload, separators=(",", ":"))

            # Construct the prompt for the Gemini model
            prompt = f"""
            Analyze the following telemetry data from a simulated R&S RTB2000 oscilloscope:

            {payload_json}

            Provide your expert diagnosis and recommendations as specified in the system instructions.
            """

            # Generate content with the Gemini model
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user", parts=[types.Part(text=SYSTEM_PROMPT)]
                    ),
                    types.Content(
                        role="user", parts=[types.Part(text=prompt)]
                    ),
                ],
                # Configure timeout via request_options
                request_options={"timeout": self.timeout_seconds},
            )

            # Extract the text response
            if (
                response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts
            ):
                # Join all parts (should be just one part for text)
                markdown_response = "".join(
                    part.text
                    for part in response.candidates[0].content.parts
                    if hasattr(part, "text")
                )
                return markdown_response.strip()
            else:
                logger.warning("Empty response from Gemini model")
                return "Diagnostic analysis completed but returned empty response."

        except Exception as e:
            error_msg = f"Diagnostic analysis failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Error during analysis: {error_msg}"

    def _build_telemetry_payload(
        self, telemetry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build the minified JSON telemetry payload from the detector output.

        Parameters
        ----------
        telemetry : dict
            Raw telemetry from anomaly_detector.detect_anomalies

        Returns
        -------
        dict
            Minified payload containing metrics, flags, and active parameters.
        """
        # Extract computed metrics
        computed_metrics = telemetry.get("computed_metrics", {})
        ch1_metrics = computed_metrics.get("ch1", {})
        ch2_metrics = computed_metrics.get("ch2", {})

        # Build the payload structure
        payload = {
            "metrics": {
                "overshoot_ratio": ch1_metrics.get("overshoot_ratio", 0.0),
                "vpp_ch1": ch1_metrics.get("vpp", 0.0),
                "vpp_ch2": ch2_metrics.get("vpp", 0.0),
                "rise_time": ch1_metrics.get("rise_time", 0.0),
            },
            "flags": {
                "is_fault": telemetry.get("is_fault", False),
                "ml_anomaly": telemetry.get("ml_anomaly_score", 0.0)
                < 0,  # Note: score_samples returns lower for anomalies
                "fault_reasons": telemetry.get("fault_reasons", []),
            },
            "active_parameters": telemetry.get("active_parameters", {}),
        }

        return payload
