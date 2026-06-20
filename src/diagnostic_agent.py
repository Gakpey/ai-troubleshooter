"""
Diagnostic Agent for PulseGuard AI

Implements the Inline Diagnostics Agent as specified in
@specs.md section 3.3 and @CLAUDE.md.

The DiagnosticAgent class uses NVIDIA NIM API as primary and
Google Gemini SDK as fallback for generating root-cause analysis
and corrective lab configurations based on telemetry data from
the anomaly detector.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

# Import the Google Generative AI client (fallback)
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GEMINI_AVAILABLE = False

# Import OpenAI client for NVIDIA NIM (primary)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default timeout for API calls in seconds
DEFAULT_TIMEOUT_SECONDS = 10.0

# System prompt for the model to act as a
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
    Diagnostic agent that uses NVIDIA NIM API as primary and Google Gemini SDK as fallback.

    Parameters
    ----------
    nim_model_name : str, default=None
        Name of the NVIDIA NIM model to use. If None, reads from NIM_MODEL env var.
    gemini_model_name : str, default=None
        Name of the Gemini model to use. If None, reads from GEMINI_MODEL env var.
    timeout_seconds : float, default=10.0
        Timeout for API calls in seconds.
    """

    def __init__(
        self,
        nim_model_name: Optional[str] = None,
        gemini_model_name: Optional[str] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.timeout_seconds = timeout_seconds

        # Determine NVIDIA NIM model to use
        if nim_model_name is None:
            nim_model_name = os.getenv("NIM_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
        self.nim_model_name = nim_model_name

        # Determine Gemini model to use
        if gemini_model_name is None:
            gemini_model_name = os.getenv("GEMINI_MODEL", os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"))
        self.gemini_model_name = gemini_model_name

        # Initialize NVIDIA NIM client (primary)
        self.nim_client = None
        if OPENAI_AVAILABLE:
            nim_api_key = os.getenv("NVIDIA_API_KEY")
            if nim_api_key:
                try:
                    self.nim_client = OpenAI(
                        base_url="https://integrate.api.nvidia.com/v1",
                        api_key=nim_api_key
                    )
                    logger.info(f"NVIDIA NIM client initialized successfully with model: {self.nim_model_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize NVIDIA NIM client: {e}")
            else:
                logger.warning("NVIDIA_API_KEY environment variable not set")
        else:
            logger.warning("OpenAI package not available for NVIDIA NIM support")

        # Initialize Gemini client (fallback)
        self.gemini_client = None
        if GEMINI_AVAILABLE:
            gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if gemini_api_key:
                try:
                    self.gemini_client = genai.Client(api_key=gemini_api_key)
                    logger.info(f"Gemini client initialized successfully with model: {self.gemini_model_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize Gemini client: {e}")
            else:
                logger.warning("GOOGLE_API_KEY/GEMINI_API_KEY environment variable not set")
        else:
            logger.warning("Google Generative AI package not available")

        # Log initialization status
        if self.nim_client is None and self.gemini_client is None:
            raise RuntimeError(
                "Failed to initialize any diagnostic API client. "
                "Please set NVIDIA_API_KEY for NVIDIA NIM or GOOGLE_API_KEY/GEMINI_API_KEY for Gemini."
            )
        elif self.nim_client is not None:
            logger.info("Using NVIDIA NIM as primary diagnostic API")
        else:
            logger.info("Using Gemini as diagnostic API (NVIDIA NIM not available)")

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
            logger.info("No fault detected in telemetry; skipping diagnostic analysis.")
            return ""

        # Try NVIDIA NIM first (primary), then fall back to Gemini
        if self.nim_client is not None:
            try:
                return self._analyze_with_nim(telemetry)
            except Exception as e:
                logger.warning(f"NVIDIA NIM analysis failed, falling back to Gemini: {e}")
                # Fall through to Gemini

        # Try Gemini as fallback
        if self.gemini_client is not None:
            try:
                return self._analyze_with_gemini(telemetry)
            except Exception as e:
                logger.error(f"Gemini analysis also failed: {e}")
                return f"Error during analysis: Primary and fallback APIs both failed. NIM: {str(e) if self.nim_client else 'Not configured'}, Gemini: {str(e)}"
        else:
            error_msg = (
                "Diagnostic analysis unavailable: Neither NVIDIA NIM nor Gemini API configured. "
                "Please set NVIDIA_API_KEY for NVIDIA NIM or GOOGLE_API_KEY/GEMINI_API_KEY for Gemini."
            )
            logger.error(error_msg)
            return error_msg

    def _analyze_with_nim(self, telemetry: Dict[str, Any]) -> str:
        """Analyze telemetry using NVIDIA NIM API."""
        # Build the minified JSON payload as per specs
        payload = self._build_telemetry_payload(telemetry)

        # Convert payload to JSON string (minified)
        payload_json = json.dumps(payload, separators=(",", ":"))

        # Construct the prompt for the NIM model
        prompt = f"""
        Analyze the following telemetry data from a simulated R&S RTB2000 oscilloscope:

        {payload_json}

        Provide your expert diagnosis and recommendations as specified in the system instructions.
        """

        # Generate content with the NVIDIA NIM model
        completion = self.nim_client.chat.completions.create(
            model=self.nim_model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            top_p=0.95,
            max_tokens=65536,
            extra_body={"chat_template_kwargs":{"enable_thinking":True},"reasoning_budget":16384},
            stream=False
        )

        # Extract the response
        reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
        if reasoning:
            logger.debug(f"NIM reasoning content: {reasoning}")

        markdown_response = completion.choices[0].message.content
        if markdown_response:
            return markdown_response.strip()
        else:
            logger.warning("Empty response from NVIDIA NIM model")
            return "Diagnostic analysis completed but returned empty response."

    def _analyze_with_gemini(self, telemetry: Dict[str, Any]) -> str:
        """Analyze telemetry using Gemini API (fallback)."""
        if self.gemini_client is None:
            raise RuntimeError("Gemini client not initialized")

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
        response = self.gemini_client.models.generate_content(
            model=self.gemini_model_name,
            contents=[
                types.Content(role="user", parts=[types.Part(text=SYSTEM_PROMPT)]),
                types.Content(role="user", parts=[types.Part(text=prompt)]),
            ],
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

    def _build_telemetry_payload(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
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