"""
PulseGuard AI - Streamlit Dashboard
Phase 4: Presentation Dashboard

This module implements the unified dashboard interface as specified in
@specs.md section 3.4 and @CLAUDE.md.

Layout:
- Left Column: Virtual instrument front panel with interactive controls
- Right Column: Oscilloscope display with Plotly visualization, KPIs, HUD, and AI diagnostics
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from typing import Dict, Any

# Import local modules
from data_simulator import simulate_signals
from anomaly_detector import AnomalyDetector
from diagnostic_agent import DiagnosticAgent
from config import (
    WIDGET_RANGES,
    WIDGET_DEFAULTS,
    IMPEDANCE_MISMATCH_OVERSHOOT_THRESHOLD,
    POWER_SUPPLI_VPP_THRESHOLD,
    PROBE_COMPENSATION_RISE_TIME_MULTIPLIER,
    DEFAULT_FREQUENCY,
    DEFAULT_AMPLITUDE,
    DEFAULT_OFFSET,
    DEFAULT_NOISE_FLOOR,
    DEFAULT_RISE_TIME_TARGET,
    DEFAULT_IMPEDANCE_MISMATCH,
)

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
    if 'fault_detected' not in st.session_state:
        st.session_state.fault_detected = False
    if 'ai_diagnostics' not in st.session_state:
        st.session_state.ai_diagnostics = "Nominal operation - no faults detected"
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()

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

def create_plotly_chart(sim_result: Dict[str, Any]) -> go.Figure:
    """
    Create Plotly visualization for CH1 and CH2 signals.

    Args:
        sim_result: Dictionary from simulate_signals containing time and voltage arrays

    Returns:
        Plotly figure object
    """
    # Create subplots: 2 rows, 1 column, shared x-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("CH1: Digital Signal (Square Wave)", "CH2: DC Rail"),
        row_heights=[0.6, 0.4]
    )

    # CH1 trace (top)
    fig.add_trace(
        go.Scatter(
            x=sim_result["ch1_time"] * 1e9,  # Convert to ns for display
            y=sim_result["ch1_voltage"],
            mode='lines',
            name='CH1',
            line=dict(color='#00FF00', width=2),  # Green like oscilloscope
            hovertemplate='Time: %{x:.1f} ns<br>Voltage: %{y:.3f} V<extra></extra>'
        ),
        row=1, col=1
    )

    # CH2 trace (bottom)
    fig.add_trace(
        go.Scatter(
            x=sim_result["ch2_time"] * 1e9,  # Convert to ns for display
            y=sim_result["ch2_voltage"],
            mode='lines',
            name='CH2',
            line=dict(color='#FF00FF', width=2),  # Magenta for DC rail
            hovertemplate='Time: %{x:.1f} ns<br>Voltage: %{y:.3f} V<extra></extra>'
        ),
        row=2, col=1
    )

    # Update layout for oscilloscope feel
    fig.update_layout(
        title={
            'text': "PulseGuard AI - Real-Time Waveform Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#FFFFFF'}
        },
        plot_bgcolor='#000000',  # Black background
        paper_bgcolor='#000000',
        font=dict(color='#FFFFFF', size=12),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=600
    )

    # Update axes
    fig.update_xaxes(
        title_text="Time (ns)",
        gridcolor='#333333',
        zerolinecolor='#555555',
        row=2, col=1
    )
    fig.update_yaxes(
        title_text="Voltage (V)",
        gridcolor='#333333',
        zerolinecolor='#555555',
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Voltage (V)",
        gridcolor='#333333',
        zerolinecolor='#555555',
        row=2, col=1
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
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)

        # CH1 Metrics
        with col1:
            st.metric(
                label="CH1 Overshoot Ratio",
                value=f"{sim_result['ch1_metrics']['overshoot_ratio']*100:.1f}%",
                delta=None,
                delta_color="off"
            )

        with col2:
            st.metric(
                label="CH1 Rise Time",
                value=f"{sim_result['ch1_metrics']['rise_time']*1e9:.1f} ns",
                delta=None,
                delta_color="off"
            )

        with col3:
            st.metric(
                label="CH1 Vpp",
                value=f"{sim_result['ch1_metrics']['vpp']:.3f} V",
                delta=None,
                delta_color="off"
            )

        # CH2 & System Metrics
        with col4:
            st.metric(
                label="CH2 Vpp (PSI)",
                value=f"{sim_result['ch2_metrics']['vpp']*1000:.0f} mV",
                delta=None,
                delta_color="off"
            )

        # Second row for additional metrics
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric(
                label="ML Anomaly Score",
                value=f"{fault_result.get('ml_anomaly_score', 0.0):.3f}",
                delta=None,
                delta_color="off"
            )

        with col6:
            fault_status = "FAULT" if fault_result.get('is_fault', False) else "OK"
            st.metric(
                label="System Status",
                value=fault_status,
                delta=None,
                delta_color="off" if not fault_result.get('is_fault', False) else "inverse"
            )

        with col7:
            # Update fault detection state in session
            st.session_state.fault_detected = fault_result.get('is_fault', False)

        with col8:
            st.metric(
                label="Update Rate",
                value=f"{1.0/(time.time() - st.session_state.last_update):.1f} Hz",
                delta=None,
                delta_color="off"
            )

def display_hud():
    """Display fault HUD (Heads-Up Display) when fault is detected."""
    hud_placeholder = st.empty()

    if st.session_state.fault_detected:
        # Display high-contrast notification
        hud_placeholder.error(
            "⚠️ FAULT DETECTED - Check AI Diagnostics for Root Cause Analysis",
            icon="⚠️"
        )
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
        if not fault_result.get('is_fault', False):
            st.info("🔍 Nominal operation - no faults detected")
            st.session_state.ai_diagnostics = "Nominal operation - no faults detected"
        else:
            # Show spinner while analyzing
            with st.spinner("🤖 Analyzing fault with Gemini AI..."):
                try:
                    # Initialize diagnostic agent
                    agent = DiagnosticAgent()
                    # Get analysis (this will call the Gemini API)
                    analysis = agent.analyze_telemetry(fault_result)
                    st.session_state.ai_diagnostics = analysis
                except Exception as e:
                    error_msg = f"AI diagnostics unavailable: {str(e)}"
                    st.session_state.ai_diagnostics = error_msg
                    st.error(error_msg)

            # Display the analysis
            if st.session_state.ai_diagnostics:
                st.markdown("### 📋 AI Root-Cause Analysis")
                st.markdown(st.session_state.ai_diagnostics)

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # App header
    st.title("📡 PulseGuard AI: R&S Waveform Anomaly & Diagnostics Engine")
    st.caption("Real-time reactive pipeline for structural electrical fault detection")

    # Create two-column layout
    left_col, right_col = st.columns([1, 1])

    # === LEFT COLUMN: VIRTUAL FRONT PANEL ===
    with left_col:
        st.subheader("🔧 Virtual Front Panel")

        # Carrier Frequency Slider
        st.session_state.carrier_frequency = st.slider(
            label="Carrier Frequency (MHz)",
            min_value=WIDGET_RANGES["carrier_frequency"][0]/1e6,
            max_value=WIDGET_RANGES["carrier_frequency"][1]/1e6,
            value=st.session_state.carrier_frequency/1e6,
            step=WIDGET_RANGES["carrier_frequency"][2]/1e6,
            help="Frequency of the square wave signal on CH1"
        ) * 1e6  # Convert back to Hz

        # Amplitude (Vpp) Slider
        st.session_state.amplitude = st.slider(
            label="Peak-to-Peak Amplitude (V)",
            min_value=WIDGET_RANGES["amplitude"][0],
            max_value=WIDGET_RANGES["amplitude"][1],
            value=st.session_state.amplitude,
            step=WIDGET_RANGES["amplitude"][2],
            help="Peak-to-peak voltage of the CH1 square wave"
        )

        # Offset Slider
        st.session_state.offset = st.number_input(
            label="DC Offset (V)",
            min_value=WIDGET_RANGES["offset"][0],
            max_value=WIDGET_RANGES["offset"][1],
            value=st.session_state.offset,
            step=WIDGET_RANGES["offset"][2],
            help="DC offset voltage for CH1"
        )

        # Noise Floor Slider
        st.session_state.noise_floor = st.slider(
            label="Noise Floor (%)",
            min_value=WIDGET_RANGES["noise_floor"][0],
            max_value=WIDGET_RANGES["noise_floor"][1],
            value=st.session_state.noise_floor,
            step=WIDGET_RANGES["noise_floor"][2],
            help="Relative noise floor percentage (maps to 0.00V-0.50V σ)"
        )

        # Rise Time Slider
        st.session_state.rise_time_target = st.slider(
            label="Rise Time (ns)",
            min_value=WIDGET_RANGES["rise_time_target"][0]/1e-9,
            max_value=WIDGET_RANGES["rise_time_target"][1]/1e-9,
            value=st.session_state.rise_time_target/1e-9,
            step=WIDGET_RANGES["rise_time_target"][2]/1e-9,
            help="Target rise time (10% to 90%) for CH1 square wave"
        ) * 1e-9  # Convert back to seconds

        # Impedance Mismatch Slider
        st.session_state.impedance_mismatch = st.slider(
            label="Impedance Mismatch (V)",
            min_value=WIDGET_RANGES["impedance_mismatch"][0],
            max_value=WIDGET_RANGES["impedance_mismatch"][1],
            value=st.session_state.impedance_mismatch,
            step=WIDGET_RANGES["impedance_mismatch"][2],
            help="Impedance mismatch amplitude (0.0V = perfect match, 1.0V = severe)"
        )

    # === RIGHT COLUMN: OSCILLOSCOPE DISPLAY ===
    with right_col:
        st.subheader("📊 Oscilloscope Display")

        # Prepare parameters for simulation
        sim_params = {
            "frequency": st.session_state.carrier_frequency,
            "amplitude": st.session_state.amplitude,
            "offset": st.session_state.offset,
            "impedance_mismatch": st.session_state.impedance_mismatch,
            "noise_floor": st.session_state.noise_floor,
            "rise_time_target": st.session_state.rise_time_target,
            "sample_rate": 1.0e9,  # Fixed at 1 ns timestep
            "duration": 1.0e-6,    # Fixed at 1 µs duration
            "virtual_uptime": 0.0,  # Could be made dynamic if needed
        }

        # Get simulated data (cached)
        sim_result = get_simulated_data(sim_params)

        # Update last update time
        st.session_state.last_update = time.time()

        # Create and display Plotly chart
        fig = create_plotly_chart(sim_result)
        plot_placeholder = st.empty()
        with plot_placeholder.container():
            st.plotly_chart(fig, width='stretch')

        # Display KPIs
        st.subheader("📈 Key Performance Indicators")

        # Detect anomalies
        detector = AnomalyDetector()
        fault_result = detector.detect_anomalies(
            ch1_time=sim_result["ch1_time"],
            ch1_voltage=sim_result["ch1_voltage"],
            ch2_time=sim_result["ch2_time"],
            ch2_voltage=sim_result["ch2_voltage"],
            params=sim_params
        )

        # Display KPI metrics
        display_kpis(sim_result, fault_result)

        # Display HUD (fault notification)
        st.subheader("🚨 System Status")
        display_hud()

        # Display AI diagnostics
        st.subheader("🤖 AI Diagnostics")
        display_ai_diagnostics(fault_result)

# Run the app
if __name__ == "__main__":
    main()