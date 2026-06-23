# PulseGuard AI - R&S Waveform Anomaly & Diagnostics Engine

A real-time reactive pipeline that synthesizes high-frequency time-series signal data mimicking Rohde & Schwarz oscilloscope traces, detects structural electrical faults via a hybrid Isolation Forest and deterministic rule engine, and provides automated root-cause diagnostics through the Google Gemini SDK.

## 🎯 Project Overview

PulseGuard AI creates an interactive digital twin for R&S RTB2000 waveform analysis with:
- **Sub-250ms latency** for data simulation and fault detection
- **AI-powered insights** via Google Gemini SDK (gemini-2.5-flash)
- **Interactive dashboard** with real-time visualization and controls
- **Fault detection** for impedance mismatch, power supply instability, and probe compensation errors

## 🔧 Features

### Four-Phase Pipeline
1. **Phase 1 - Ingestion & Simulation** (`data_simulator.py`): 
   - Baseline signal synthesizers + fault injection
   - Impedance mismatch, power supply instability, probe compensation simulation

2. **Phase 2 - Analytics & Feature Engine** (`anomaly_detector.py`):
   - Isolation Forest + mathematical heuristics
   - Overshoot %, Vpp, rise time calculations

3. **Phase 3 - Diagnostics & Synthesis** (`diagnostic_agent.py`):
   - Google Gemini SDK for root-cause analysis
   - Structured markdown insights and recommendations

4. **Phase 4 - Presentation Dashboard** (`app.py`):
   - Streamlit UI with interactive controls
   - Plotly visualization with real-time updates
   - KPI display, fault HUD, AI diagnostics viewport

### Interactive Controls (Left Panel)
- Carrier Frequency: 1.0-25.0 MHz (slider, 0.1 MHz steps)
- Vpp (Amplitude): 20 mV-5.0 V (slider, 10 mV steps)
- Voffset (DC Offset): -2.5 to +2.5 V (number input, 0.1 V steps)
- Noise Floor: 0-100% (slider, 1% steps)
- Rise Time: 1-100 ns (slider, 1 ns steps)
- Impedance Mismatch: 0.0-1.0 V (slider, 0.01 V steps)

### Dashboard Components (Right Panel)
- Real-time Plotly visualization (CH1: green square wave, CH2: magenta DC)
- KPI display: overshoot ratio, rise time, Vpp, ML anomaly score
- Fault HUD: high-contrast notification when thresholds exceeded
- AI Diagnostics: streaming analysis from Gemini AI

## 📋 Implementation Status

### Milestone 1: ✅ Complete
- Standalone `data_simulator.py` with physics verification

### Milestone 2: ✅ Complete
- `anomaly_detector.py` math features + IsolationForest threshold validation

### Milestone 3: ✅ Complete
- Integrated `diagnostic_agent.py` via Gemini SDK

### Milestone 4: ✅ Complete
- **Interactive Streamlit dashboard** (`src/app.py`)
- Two-column layout: virtual front panel + oscilloscope display
- Full `st.session_state` reactivity
- `@st.cache_data` optimization for simulation functions
- Real-time Plotly visualization
- KPI panel with key metrics
- Fault HUD with visual notifications
- AI diagnostics using Gemini API (with spinner during analysis)
- Professional R&S RTB2000 aesthetic

## ⚡ Performance
- **Processing Latency**: <250ms for data simulation + Isolation Forest evaluation (50,000 samples)
- **Memory Optimization**: `@st.cache_data` prevents redundant computation
- **Asynchronous GenAI**: Non-blocking API calls with timeout handling
- **Overall Interactive Latency**: 5-10 seconds target (including Gemini API)

## 🧪 Testing
- **TDD Mandatory**: ≥80% coverage target
- **pytest Fixtures**: Baseline signals and fault profiles
- **Numerical Assertions**: Tolerance-based validation
- **Latency Tests**: End-to-end timing benchmarks
- **Streamlit Tests**: Session state mocking

## 🛠️ Tech Stack
- Python 3.11+
- Streamlit >=1.30.0
- scikit-learn >=1.4.0
- Plotly >=5.18.0
- google-genai >=0.3.0
- numpy >=1.26.0
- scipy >=1.11.0
- pytest >=8.0.0
- black >=24.0.0
- flake8 >=7.0.0

## 🚀 Usage

### Installation
```bash
pip install -e .[dev]
```

### Development Commands
- `dev`: Install development dependencies
- `test`: Run pytest with coverage
- `lint`: Run black and flake8
- `profile`: Profile performance with cProfile/snakeviz
- `run`: Launch Streamlit dashboard (port 8501)

### Running the Application
```bash
streamlit run src/app.py --server.port=8501
```
Then open http://localhost:8501 in your browser.

## 📁 Project Structure
```
src/
├── app.py             # Streamlit dashboard (Milestone 4)
├── data_simulator.py  # Signal generation & fault injection (Milestone 1)
├── anomaly_detector.py # Hybrid fault detection (Milestone 2)
├── diagnostic_agent.py # AI diagnostics (Milestone 3)
├── config.py          # Constants and configuration
└── __init__.py        # Package initializer

tests/                 # Unit tests
specs.md               # Detailed specifications
```

## 📈 Key Metrics & Fault Detection
- **Overshoot Ratio**: Localized peak search on logical '1' blocks (threshold: >10%)
- **Vpp**: Delta bounds on DC arrays (threshold: >100mV)
- **Rise Time**: 10%→90% amplitude crossing (threshold: >1.2× target)
- **Isolation Forest**: n_estimators=100, contamination=0.03, random_state=42

## 🏷️ Conventional Commits
- feat: new feature
- fix: bug fix
- docs: documentation changes
- style: formatting, linting
- refactor: code restructuring
- test: test additions/modifications
- chore: build process, tooling updates

## 📝 Changelog
See `docs/CHANGELOG.md` for detailed release notes.

## ⚖️ License
This project is proprietary software developed for the R&S Hackathon.

---
*Built with ❤️ for the R&S Hackathon by Joshua Gakpey-Lawson & Kwaku Baffour*
