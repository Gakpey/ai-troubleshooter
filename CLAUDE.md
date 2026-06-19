# PulseGuard AI – CLAUDE.md

## 1. Project Overview & Goals
PulseGuard AI is a real-time reactive pipeline that synthesizes high-frequency time-series signal data mimicking Rohde & Schwarz oscilloscope traces, detects structural electrical faults via a hybrid Isolation Forest and deterministic rule engine, and provides automated root-cause diagnostics through the Google Gemini SDK. The goal is to create an interactive digital twin for R&S RTB2000 waveform analysis with sub-250ms latency and AI-powered insights.

## 2. Tech Stack & Version Constraints
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
*(Pin exact versions in requirements.txt; avoid deprecated APIs)*

## 3. Mandatory Commands
- `dev`: `pip install -e .[dev]` (installs dev dependencies)
- `test`: `pytest -v --cov=src --cov-report=term-missing`
- `lint`: `black src tests && flake8 src tests`
- `profile`: `python -m cProfile -s time src/app.py` (or use `snakeviz`)
- `run`: `streamlit run src/app.py --server.port=8501`

## 4. Architecture & Phasing
**Four-Phase Pipeline** (exactly as in specs.md):
1. **Phase 1 – Ingestion & Simulation** (`data_simulator.py`): Baseline signal synthesizers + fault injection (impedance mismatch, power supply instability, probe compensation).
2. **Phase 2 – Analytics & Feature Engine** (`anomaly_detector.py`): IsolationForest + mathematical heuristics (overshoot %, Vpp, rise time).
3. **Phase 3 – Diagnostics & Synthesis** (`diagnostic_agent.py`): Google Gemini SDK (`gemini-2.5-flash`) for root-cause analysis.
4. **Phase 4 – Presentation Dashboard** (`app.py`): Streamlit UI with interactive controls, Plotly visualization, HUD, AI insights.

**Milestones** (sequential):
- M1: Standalone `data_simulator.py` with physics verification.
- M2: `anomaly_detector.py` math features + IsolationForest threshold validation.
- M3: Integrate `diagnostic_agent.py` via Gemini SDK.
- M4: Wire states in `app.py`, deploy Plotly + real-time HUD + streaming diagnostics.

## 5. Coding Standards
- Type hints on all public functions/methods.
- Google-style docstrings (Args, Returns, Raises, Examples).
- Use numpy/scipy for signal processing; avoid manual loops where vectorized.
- PEP 8 compliance enforced by black & flake8.
- Numerical tolerance: Use `numpy.isclose` with rtol=1e-5, atol=1e-8 for physics checks.
- Constants in UPPER_SNAKE_CASE; configure via `src/config.py`.

## 6. Critical Non-Functional Requirements
- **Memory Optimization:** Streamlit caching operations (`@st.cache_data`) must be wrapped around the simulation functions to block repetitive computational execution during slider changes. (Specs section 4)
- **Processing Latency:** Combined data simulation and Isolation Forest evaluation must complete execution in under $250\text{ms}$ on standard CPU profiles for datasets up to 50,000 sampling points. (Specs section 4)
- **Asynchronous GenAI Execution:** API calls to the Google Gemini SDK must handle timeouts elegantly and run non-blockingly so that the chart interactions remain fluid even while waiting for an AI analysis stream. (Specs section 4)
- **Overall Interactive Latency:** End-to-end latency (including Gemini API) target: 5-10 seconds for interactive use. (New problem statement)
- **Diagnostic Accuracy:** Target >80% precision and recall on key fault classes, evaluated on labeled datasets of waveforms/traces with annotated faults. (New problem statement)

## 7. Signal Processing & Math Fidelity Rules
Implement exact formulas from specs.md sections 3.1 & 3.6:
- **Baseline Signals**:
  - CH1: Square wave, f=5MHz, Δt=1ns, Vlow=0V, Vhigh=3.3V, tr=10ns.
  - CH2: Constant Vdc=3.3V or 5.0V.
- **Fault Injection**:
  1. Impedance Mismatch: Overshoot Ratio = (Vmax - Vhigh) / Vamplitude > 0.10; damped ringing: A·exp(-t/τ)·cos(2πf_ring t).
  2. Power Supply Instability: Vpp = Vmax - Vmin > 100mV; sawtooth/sine + Gaussian noise.
  3. Probe Compensation: Measured rise time > 1.2 × target rise time; modify charging coefficient τ_slow.
- **Metrics**:
  - Overshoot %: Localized peak search on logical '1' blocks.
  - Vpp: Delta bounds across DC arrays.
  - Rise time (tr): Δt between 10% and 90% amplitude thresholds on positive transition.
- **Digital Twin Effects** (3.6):
  - ADC Quantization: 10-bit steps based on Full-scale = Vdiv × 8.
  - Sampling Rate & Anti-Aliasing: If virt. sample rate < 2×f_carrier, reflect aliased frequencies.
  - Thermal Drift: ±2% voltage offset after virtual uncalibrated uptime.
  - VNA Errors: Directivity (D) and Source Match (Sm) error matrices; map Z=R+jX to VSWR, Return Loss, Phase.
  - SCPI Register: NOALigndata bit, overvoltage (>400V) logging, airflow errors.

## 8. Testing Strategy
- **TDD Mandatory**: Write tests before implementation; maintain ≥80% coverage.
- **pytest Fixtures**: Baseline signals (square wave, DC) and each fault profile.
- **Numerical Assertions**: Compare computed metrics against analytical expectations with tolerance.
- **Isolation Forest**: Validate contamination=0.03 yields expected outlier fraction.
- **Latency Tests**: End-to-end timing benchmarks using `time.perf_counter`.
- **Streamlit Tests**: Use `streamlit-runner` or `pytest` with `st.session_state` mocking.

## 9. UI/Streamlit Rules
- **Reactive State**: All parameters stored in `st.session_state`; widget changes trigger full pipeline rerun.
- **Plotly**: Real-time updating of traces; use `plotly.graph_objects` for performance.
- **Widget Mapping** (exact from 3.5):
  - Carrier Frequency: Slider 1.0–25.0 MHz, step 0.1 MHz, default 5.0 MHz.
  - Vpp: Slider 20 mV–5.0 V, step 10 mV, default 3.3 V.
  - Voffset: Number input -2.5 V to +2.5 V, step 0.1 V, default 0.0 V.
  - Noise Floor: Slider 0–100% → σ from 0.00V to 0.50V.
  - Rise Time: Slider 1–100 ns, step 1 ns, default 10 ns.
  - Impedance Mismatch: Slider 0.0V (perfect) to 1.0V (severe).
- **HUD**: Flash high-contrast notification when sliders cross fault threshold.
- **AI Viewport**: Use `st.write_stream` or async spinner for streaming Gemini markdown.

## 10. Gemini SDK Rules
- **Security**: API key only via `.env` (gitignored); load via `os.getenv`; never hardcode.
- **Payload**: Minified JSON with computed metrics, anomaly flags, virtual hardware params.
- **Persona**: System instruction to mimic Rohde & Schwarz Field Applications Engineer.
- **Output**: Structured markdown insights (root cause, corrective lab config).
- **Error Handling**: Timeout, retry with exponential backoff; display user-friendly message.

## 11. Git & Review Etiquette
- **Commits**: Conventional Commits (feat, fix, docs, style, refactor, test, chore).
- **Branches**: Short-lived feature branches off `main`; PRs required for Merge.
- **Secrets**: Never commit `.env`, API keys, or credentials; use `pre-commit` hooks to block.
- **Reviews**: Require at least one approval; self-merge discouraged.
- **Changelog**: Update `docs/CHANGELOG.md` on feature/releases.

## 12. Progressive Disclosure
This file holds essential conventions. For detailed specifications, refer to:
- `@specs.md` (authoritative SRS v3.0.0)
- Future `docs/` directory (design deep dives, API references, deployment guides).

Let this be the single source of truth so prompts remain focused and concise.