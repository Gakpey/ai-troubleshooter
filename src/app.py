"""
PulseGuard AI - Streamlit Dashboard
Phase 4: Presentation Dashboard

This module implements the unified dashboard interface as specified in
@specs.md section 3.4 and @CLAUDE.md.

Layout:
- Left Column: Virtual instrument front panel with interactive controls
- Right Column: Oscilloscope display with Plotly visualization, KPIs, HUD, and AI diagnostics
"""

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    pass  # dotenv not available, continue without it

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import time
import traceback
from typing import Dict, Any

# Import local modules
from data_simulator import simulate_signals
from anomaly_detector import AnomalyDetector
from diagnostic_agent import DiagnosticAgent
from config import (
    WIDGET_RANGES,
    WIDGET_DEFAULTS,
)

# Import validation function (with fallback for when running validation script directly)
try:
    from validate_full_pipeline import run_all_tests
    VALIDATION_AVAILABLE = True
except ImportError:
    # When running the validation script directly, avoid circular import
    VALIDATION_AVAILABLE = False
    run_all_tests = None

# Cache the anomaly detector since its parameters are fixed


@st.cache_resource
def get_anomaly_detector():
    """Create and cache the AnomalyDetector instance.

    The AnomalyDetector parameters are fixed constants, so we cache the instance
    to avoid refitting the Isolation Forest model on every UI update.
    """

    return AnomalyDetector()


# Page configuration
st.set_page_config(
    page_title="PulseGuard AI - R&S Waveform Anomaly & Diagnostics Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Initialize session state for widget parameters
def initialize_session_state():
    """Initialize session state with default values if not present."""
    for key, default in WIDGET_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Initialize additional state
    if "fault_detected" not in st.session_state:
        st.session_state.fault_detected = False
    if "ai_diagnostics" not in st.session_state:
        st.session_state.ai_diagnostics = "Nominal operation - no faults detected"
    if "last_update" not in st.session_state:
        st.session_state.last_update = time.time()

def reset_to_defaults():
    """Reset all widget parameters to their default values."""
    for key, default in WIDGET_DEFAULTS.items():
        st.session_state[key] = default
    # Reset fault-related session state
    st.session_state.fault_detected = False
    st.session_state.fault_reasons = []
    st.session_state.ai_diagnostics = "Nominal operation - no faults detected"


# Cached simulation function to prevent redundant computation
@st.cache_data
def get_simulated_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cached wrapper for data simulation.

    Args:
        params: Dictionary of simulation parameters

    Returns:
        Dictionary containing simulation results
    """
    return simulate_signals(**params)


def create_plotly_chart(sim_result: Dict[str, Any], fault_result: Dict[str, Any] = None) -> go.Figure:
    """
    Create Plotly visualization for CH1 and CH2 signals.

    Args:
        sim_result: Dictionary from simulate_signals containing time and voltage arrays
        fault_result: Dictionary from anomaly detector containing fault info (optional)

    Returns:
        Plotly figure object
    """
    # Create subplots: 2 rows, 1 column, shared x-axis
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("CH1: Digital Signal (Square Wave)", "CH2: DC Rail"),
        row_heights=[0.6, 0.4],
    )

    # CH1 trace (top)
    fig.add_trace(
        go.Scatter(
            x=sim_result["ch1_time"] * 1e9,  # Convert to ns for display
            y=sim_result["ch1_voltage"],
            mode="lines",
            name="CH1",
            line=dict(color="#00FF00", width=2),  # Green like oscilloscope
            hovertemplate="Time: %{x:.1f} ns<br>Voltage: %{y:.3f} V<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # CH2 trace (bottom)
    fig.add_trace(
        go.Scatter(
            x=sim_result["ch2_time"] * 1e9,  # Convert to ns for display
            y=sim_result["ch2_voltage"],
            mode="lines",
            name="CH2",
            line=dict(color="#FF00FF", width=2),  # Magenta for DC rail
            hovertemplate="Time: %{x:.1f} ns<br>Voltage: %{y:.3f} V<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Update layout for oscilloscope feel
    fig.update_layout(
        title={
            "text": "PulseGuard AI - Real-Time Waveform Analysis",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20, "color": "#FFFFFF"},
        },
        plot_bgcolor="#000000",  # Black background
        paper_bgcolor="#000000",
        font=dict(color="#FFFFFF", size=12),
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)"
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=600,
        dragmode='zoom',  # Enable box zoom
    )

    # Update axes
    fig.update_xaxes(
        title_text="Time (ns)",
        gridcolor="#333333",
        zerolinecolor="#555555",
        row=2,
        col=1,
    )
    # Set fixed y-axis ranges to keep zero voltage line stationary during zoom
    def _get_y_range(voltage_array, padding=0.15):
        v_min = np.min(voltage_array)
        v_max = np.max(voltage_array)
        # Ensure zero is included in the range
        v_min = min(0.0, v_min)
        v_max = max(0.0, v_max)
        range_ = v_max - v_min
        if range_ == 0:
            range_ = 1.0
        return v_min - padding * range_, v_max + padding * range_

    ch1_yrange = _get_y_range(sim_result["ch1_voltage"])
    ch2_yrange = _get_y_range(sim_result["ch2_voltage"])

    fig.update_yaxes(
        title_text="Voltage (V)",
        gridcolor="#333333",
        zerolinecolor="#555555",
        range=ch1_yrange,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Voltage (V)",
        gridcolor="#333333",
        zerolinecolor="#555555",
        range=ch2_yrange,
        row=2,
        col=1,
    )

    # Add fault annotations if fault is detected
    if fault_result and fault_result.get("is_fault", False):
        # Map fault reasons to colors and symbols
        fault_config = {
            "overshoot_threshold_exceeded": {"color": "#ff6b6b", "symbol": "⚡", "name": "Impedance Mismatch"},
            "power_supply_vpp_exceeded": {"color": "#ffd93d", "symbol": "🔋", "name": "Power Supply Instability"},
            "rise_time_threshold_exceeded": {"color": "#4ecdc4", "symbol": "📡", "name": "Probe Compensation"},
        }

        fault_reasons = fault_result.get("fault_reasons", [])

        # Add annotations for each fault type
        for i, reason in enumerate(fault_reasons):
            config = fault_config.get(reason, {"color": "#ffffff", "symbol": "⚠️", "name": reason.replace("_", " ").title()})

            # Add a shaded region background for the fault
            # We'll highlight the entire time range for simplicity
            fig.add_vrect(
                x0=sim_result["ch1_time"][0] * 1e9,  # Start time in ns
                x1=sim_result["ch1_time"][-1] * 1e9,  # End time in ns
                fillcolor=config["color"],
                opacity=0.1,
                layer="below",
                line_width=0,
            )

            # Add annotation text
            fig.add_annotation(
                x=sim_result["ch1_time"][0] * 1e9 + (sim_result["ch1_time"][-1] - sim_result["ch1_time"][0]) * 1e9 * 0.02,
                y=max(sim_result["ch1_voltage"]) * 0.9,
                text=f"{config['symbol']} {config['name']}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=config["color"],
                ax=0,
                ay=-40,
                bgcolor=config["color"],
                opacity=0.9,
                font=dict(size=10, color="white"),
                bordercolor=config["color"],
                borderwidth=1,
                borderpad=4,
            )

    return fig


def display_kpis(sim_result: Dict[str, Any], fault_result: Dict[str, Any]):
    """
    Display key performance indicators in a metric grid.

    Args:
        sim_result: Dictionary from simulation
        fault_result: Dictionary from anomaly detector
    """
    # Create a container for KPIs
    kpi_container = st.container()

    with kpi_container:
        # CH1 Signal Quality Metrics
        st.markdown("#### 📈 CH1 Signal Quality")
        ch1_col1, ch1_col2, ch1_col3 = st.columns(3)

        with ch1_col1:
            st.metric(
                label="Overshoot Ratio",
                value=f"{sim_result['ch1_metrics']['overshoot_ratio']*100:.1f}%",
                help="Percentage of peak voltage exceeding baseline (ideal: <10%)"
            )

        with ch1_col2:
            st.metric(
                label="Rise Time",
                value=f"{sim_result['ch1_metrics']['rise_time']*1e9:.1f} ns",
                help="Time for signal to transition from 10% to 90% amplitude"
            )

        with ch1_col3:
            st.metric(
                label="Peak-to-Peak Voltage",
                value=f"{sim_result['ch1_metrics']['vpp']:.3f} V",
                help="Difference between maximum and minimum voltage"
            )

        # CH2 Power Rail Metrics
        st.markdown("#### 🔋 CH2 Power Rail")
        ch2_col1, ch2_col2 = st.columns(2)

        with ch2_col1:
            st.metric(
                label="Peak-to-Peak Voltage",
                value=f"{sim_result['ch2_metrics']['vpp']*1000:.0f} mV",
                help="AC ripple on DC rail (ideal: <100mV)"
            )

        with ch2_col2:
            # System status with color coding
            fault_status = "FAULT" if fault_result.get("is_fault", False) else "OK"
            status_color = "red" if fault_result.get("is_fault", False) else "green"
            st.markdown(f"**System Status**: <span style='color:{status_color}'>{fault_status}</span>", unsafe_allow_html=True)

        # ML & System Metrics
        st.markdown("#### ⚙️ ML & System")
        ml_col1, ml_col2, ml_col3 = st.columns(3)

        with ml_col1:
            st.metric(
                label="ML Anomaly Score",
                value=f"{fault_result.get('ml_anomaly_score', 0.0):.3f}",
                help="Isolation Forest anomaly score (negative values indicate anomalies)"
            )

        with ml_col2:
            st.metric(
                label="Update Rate",
                value=f"{1.0/(time.time() - st.session_state.last_update):.1f} Hz",
                help="Rate at which the dashboard updates with new data"
            )

        with ml_col3:
            # Fault counter or additional info
            if fault_result.get("is_fault", False):
                fault_count = len(fault_result.get("fault_reasons", []))
                st.metric(
                    label="Active Faults",
                    value=f"{fault_count}",
                    help="Number of simultaneous fault conditions detected"
                )
            else:
                st.metric(
                    label="Fault Count",
                    value="0",
                    help="Number of simultaneous fault conditions detected"
                )


def display_hud():
    """Display fault HUD (Heads-Up Display) when fault is detected."""
    hud_placeholder = st.empty()

    if st.session_state.fault_detected:
        # Map fault reason keys to user-friendly labels
        reason_labels = {
            "overshoot_threshold_exceeded": "Impedance Mismatch Detected",
            "power_supply_vpp_exceeded": "Power Supply Ripple Detected",
            "rise_time_threshold_exceeded": "Probe Compensation Error",
        }
        # Get fault reasons for detailed message
        fault_reasons = st.session_state.get("fault_reasons", [])
        if fault_reasons:
            friendly_reasons = [reason_labels.get(r, r.replace("_", " ").title()) for r in fault_reasons]
            reasons_str = " + ".join(friendly_reasons)
            message = f"⚠️ FAULT DETECTED — {reasons_str}"
        else:
            message = "⚠️ FAULT DETECTED"
        # Display high-contrast notification
        hud_placeholder.error(message, icon="⚠️")
    else:
        # Clear the HUD when no fault
        hud_placeholder.empty()


def display_ai_diagnostics(fault_result: Dict[str, Any]):
    """
    Display AI diagnostics area with streaming or spinner.

    Args:
        fault_result: Dictionary from anomaly detector containing fault info
    """
    ai_container = st.container()

    with ai_container:
        if not fault_result.get("is_fault", False):
            st.info("🔍 Nominal operation - no faults detected")
            st.session_state.ai_diagnostics = "Nominal operation - no faults detected"
            st.session_state.ai_diagnostics_timestamp = None
        else:
            # Show spinner while analyzing
            with st.spinner("🤖 Analyzing fault with Gemini AI..."):
                try:
                    start_time = time.time()
                    # Initialize diagnostic agent
                    agent = DiagnosticAgent()
                    # Get analysis (this will call the Gemini API)
                    analysis = agent.analyze_telemetry(fault_result)
                    end_time = time.time()
                    analysis_time = end_time - start_time

                    # Add Gemini attribution and timing
                    st.session_state.ai_diagnostics = analysis
                    st.session_state.ai_diagnostics_timestamp = analysis_time
                except Exception as e:
                    error_msg = f"AI diagnostics unavailable: {str(e)}"
                    st.session_state.ai_diagnostics = error_msg
                    st.session_state.ai_diagnostics_timestamp = None
                    st.error(error_msg)

            # Display the analysis with attribution and timing
            if st.session_state.ai_diagnostics:
                analysis_header = "### 📋 AI Root-Cause Analysis"
                if st.session_state.ai_diagnostics_timestamp is not None:
                    analysis_header += f" *(Analysis completed in {st.session_state.ai_diagnostics_timestamp:.2f}s using Gemini)*"
                else:
                    analysis_header += " *(Analysis unavailable)*"

                st.markdown(analysis_header)
                st.markdown(st.session_state.ai_diagnostics)


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # App header
    st.title("📡 PulseGuard AI: R&S Waveform Anomaly & Diagnostics Engine")
    st.caption("Real-time reactive pipeline for structural electrical fault detection")
    # Display last updated timestamp
    last_update_time = time.strftime("%H:%M:%S", time.localtime(st.session_state.last_update))
    st.caption(f"🕒 Last updated: {last_update_time}")

    # Create two-column layout: left column slightly narrower to reduce empty space
    left_col, right_col = st.columns([0.7, 1.3])

    # === LEFT COLUMN: VIRTUAL FRONT PANEL ===
    with left_col:
        st.subheader("🔧 Virtual Front Panel")

        # Carrier Frequency Slider
        st.markdown("<span style='color:#4ecdc4'>●</span> Carrier Frequency (MHz)", unsafe_allow_html=True)
        st.session_state.carrier_frequency = (
            st.slider(
                label="Carrier Frequency (MHz)",
                label_visibility="collapsed",
                min_value=WIDGET_RANGES["carrier_frequency"][0] / 1e6,
                max_value=WIDGET_RANGES["carrier_frequency"][1] / 1e6,
                value=st.session_state.carrier_frequency / 1e6,
                step=WIDGET_RANGES["carrier_frequency"][2] / 1e6,
                help="Frequency of the square wave signal on CH1",
            )
            * 1e6
        )  # Convert back to Hz

        # Amplitude (Vpp) Slider
        st.markdown("<span style='color:#4ecdc4'>●</span> Peak-to-Peak Amplitude (V)", unsafe_allow_html=True)
        st.session_state.amplitude = st.slider(
            label="Peak-to-Peak Amplitude (V)",
            label_visibility="collapsed",
            min_value=WIDGET_RANGES["amplitude"][0],
            max_value=WIDGET_RANGES["amplitude"][1],
            value=st.session_state.amplitude,
            step=WIDGET_RANGES["amplitude"][2],
            help="Peak-to-peak voltage of the CH1 square wave",
        )

        # Offset Slider
        st.markdown("<span style='color:#4ecdc4'>●</span> DC Offset (V)", unsafe_allow_html=True)
        st.session_state.offset = st.number_input(
            label="DC Offset (V)",
            label_visibility="collapsed",
            min_value=WIDGET_RANGES["offset"][0],
            max_value=WIDGET_RANGES["offset"][1],
            value=st.session_state.offset,
            step=WIDGET_RANGES["offset"][2],
            help="DC offset voltage for CH1",
        )

        # Noise Floor Slider
        st.markdown("<span style='color:#ff6b6b'>●</span> Noise Floor (%)", unsafe_allow_html=True)
        st.session_state.noise_floor = st.slider(
            label="Noise Floor (%)",
            label_visibility="collapsed",
            min_value=WIDGET_RANGES["noise_floor"][0],
            max_value=WIDGET_RANGES["noise_floor"][1],
            value=st.session_state.noise_floor,
            step=WIDGET_RANGES["noise_floor"][2],
            help="Relative noise floor percentage (maps to 0.00V-0.50V σ)",
        )

        # Rise Time Slider
        st.markdown("<span style='color:#ff6b6b'>●</span> Rise Time (ns)", unsafe_allow_html=True)
        st.session_state.rise_time_target = (
            st.slider(
                label="Rise Time (ns)",
                label_visibility="collapsed",
                min_value=WIDGET_RANGES["rise_time_target"][0] / 1e-9,
                max_value=WIDGET_RANGES["rise_time_target"][1] / 1e-9,
                value=st.session_state.rise_time_target / 1e-9,
                step=WIDGET_RANGES["rise_time_target"][2] / 1e-9,
                help="Target rise time (10% to 90%) for CH1 square wave",
            )
            * 1e-9
        )  # Convert back to seconds

        # Impedance Mismatch Slider
        st.markdown("<span style='color:#ff6b6b'>●</span> Impedance Mismatch (V)", unsafe_allow_html=True)
        st.session_state.impedance_mismatch = st.slider(
            label="",
            min_value=WIDGET_RANGES["impedance_mismatch"][0],
            max_value=WIDGET_RANGES["impedance_mismatch"][1],
            value=st.session_state.impedance_mismatch,
            step=WIDGET_RANGES["impedance_mismatch"][2],
            help="Impedance mismatch amplitude (0.0V = perfect match, 1.0V = severe)",
        )

        st.button("Reset to Defaults", on_click=reset_to_defaults)

    # === RIGHT COLUMN: OSCILLOSCOPE DISPLAY ===
    with right_col:
        # Dynamic title based on fault status
        if st.session_state.fault_detected:
            st.subheader("📊 Oscilloscope Display - ⚠️ FAULT DETECTED")
        else:
            st.subheader("📊 Oscilloscope Display - ✅ NORMAL OPERATION")

        # Prepare parameters for simulation
        sim_params = {
            "frequency": st.session_state.carrier_frequency,
            "amplitude": st.session_state.amplitude,
            "offset": st.session_state.offset,
            "impedance_mismatch": st.session_state.impedance_mismatch,
            "noise_floor": st.session_state.noise_floor,
            "rise_time_target": st.session_state.rise_time_target,
            "sample_rate": 1.0e9,  # Fixed at 1 ns timestep
            "duration": 1.0e-6,  # Fixed at 1 µs duration
            "virtual_uptime": 0.0,  # Could be made dynamic if needed
        }

        # Get simulated data (cached)
        sim_result = get_simulated_data(sim_params)

        # Detect anomalies (using cached detector to avoid refitting model)
        detector = get_anomaly_detector()
        fault_result = detector.detect_anomalies(
            ch1_time=sim_result["ch1_time"],
            ch1_voltage=sim_result["ch1_voltage"],
            ch2_time=sim_result["ch2_time"],
            ch2_voltage=sim_result["ch2_voltage"],
            params=sim_params,
        )

        # Update session state for fault details
        st.session_state.fault_detected = fault_result.get("is_fault", False)
        st.session_state.fault_reasons = fault_result.get("fault_reasons", [])

        # Update last update time
        st.session_state.last_update = time.time()

        # Create and display Plotly chart
        fig = create_plotly_chart(sim_result, fault_result)
        plot_placeholder = st.empty()
        with plot_placeholder.container():
            st.plotly_chart(fig, width="stretch", config={'scrollZoom': True, 'doubleClick': 'reset'})

        # Display KPIs
        st.subheader("📈 Key Performance Indicators")

        # Display KPI metrics
        display_kpis(sim_result, fault_result)

        # Display HUD (fault notification)
        st.subheader("🚨 System Status")
        display_hud()

        # Display AI diagnostics
        st.subheader("🤖 AI Diagnostics")
        display_ai_diagnostics(fault_result)

        # Add validation button and panel
        st.subheader("🔬 System Validation")
        if VALIDATION_AVAILABLE:
            if st.button("Run Full System Validation", help="Run end-to-end validation tests"):
                with st.spinner("Running validation suite..."):
                    # Capture validation output
                    import io
                    import sys
                    from contextlib import redirect_stdout, redirect_stderr

                    # Create string buffers to capture output
                    stdout_buffer = io.StringIO()
                    stderr_buffer = io.StringIO()

                    try:
                        # Redirect stdout and stderr to capture validation output
                        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                            result_code = run_all_tests()

                        # Get captured output
                        stdout_output = stdout_buffer.getvalue()
                        stderr_output = stderr_buffer.getvalue()

                        # Display results in an expander
                        with st.expander("Validation Results", expanded=True):
                            if result_code == 0:
                                st.success("✅ All validation tests passed!")
                            else:
                                st.error("⚠️ Some validation tests failed")

                            # Show the captured output
                            if stdout_output:
                                st.text("Standard Output:")
                                st.code(stdout_output)

                            if stderr_output:
                                st.text("Standard Error:")
                                st.code(stderr_output)

                    except Exception as e:
                        st.error(f"Validation failed to run: {str(e)}")
                        st.text("Error details:")
                        st.code(traceback.format_exc())
        else:
            st.info("Validation module not available in this context")
            st.caption("Run `python validate_full_pipeline.py` from project root for terminal validation")


# Run the app
if __name__ == "__main__":
    main()
