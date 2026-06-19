# SOFTWARE REQUIREMENT SPECIFICATION (SRS)
## Project: PulseGuard AI – R&S Waveform Anomaly & Diagnostics Engine

### 1. Document Control & Metadata
* **Author:** Joshua Gakpey-Lawson (Junior Engineer) & Gemini (Principal AI Architect)
* **Version:** 3.0.0
* **Status:** Approved for Code Generation
* **Target Stack:** Python 3.11+, Streamlit, Scikit-Learn, Plotly, Google Gemini SDK (`google-genai`)

---

### 2. System Architecture & High-Level Topology
The system employs a decoupled, real-time reactive pipeline architectural pattern to ingest synthesized high-frequency time-series signal data, isolate structural electrical faults via localized machine learning / deterministic rules, and stream automated root-cause diagnostics via the Google Gemini SDK.

   [ Phase 1: Ingestion & Simulation Layer ]
                       │
                       ▼
   ┌───────────────────────────────────────┐
   │     data_simulator.py (CLI/UI)        │ ──► (Generates Digital / DC Trace Arrays)
   └───────────────────────────────────────┘
                       │
                       ▼
   [ Phase 2: Analytics & Feature Engine ]
                       │
                       ▼
   ┌───────────────────────────────────────┐
   │  anomaly_detector.py (Hybrid Filter)  │ ──► (Applies IsolationForest + Math Heuristics)
   └───────────────────────────────────────┘
                       │
                       ▼
   [ Phase 3: Diagnostics & Synthesis Layer ]
                       │
                       ▼
   ┌───────────────────────────────────────┐
   │     diagnostic_agent.py (GenAI)       │ ──► (Invokes `google-genai` for Root-Cause)
   └───────────────────────────────────────┘
                       │
                       ▼
   [ Phase 4: Presentation Dashboard ]
                       │
                       ▼
   ┌───────────────────────────────────────┐
   │         app.py (Streamlit UI)         │ ──► (Renders Plotly Charts, HUD & AI Insights)
   └───────────────────────────────────────┘

---

### 3. Detailed Component Specifications

#### 3.1 Data Simulation Engine (`data_simulator.py`)
To bypass physical test instrument limitations, this module programmatically synthesizes raw time-series trace arrays that structurally mimic specific hardware failures derived from Rohde & Schwarz oscilloscope operational bounds.

* **Baseline Signal Synthesizers:**
    1.  **High-Speed Digital Clock Line (CH1):** A periodic square wave signal.
        * Parameters: Frequency $f = 5\text{ MHz}$, Time step $\Delta t = 1\text{ ns}$, Low State $V_{\text{low}} = 0.0\text{V}$, High State $V_{\text{high}} = 3.3\text{V}$, Target Rise Time $t_r = 10\text{ ns}$.
    2.  **DC Power Rail Line (CH2):** A flat-line constant voltage trace.
        * Parameters: Steady-state voltage $V_{\text{dc}} = 3.3\text{V}$ or $5.0\text{V}$.
* **Programmatic Fault Injection Mechanics:**
    * **Fault Profile 1: Impedance Mismatch & Signal Reflection**
        * *Implementation:* On the rising edge transitions of the CH1 square wave, superimpose a high-frequency, exponentially decaying cosine wave (damped ringing) where the primary overshoot peak violates the steady-state threshold:
            $$\text{Overshoot Ratio} = \frac{V_{\text{max}} - V_{\text{high}}}{V_{\text{amplitude}}} > 0.10$$
    * **Fault Profile 2: Power Supply Instability (Ripple/Noise)**
        * *Implementation:* On the CH2 DC Rail, superimpose a combined periodic AC component (sawtooth or sine representing switching regulator frequency noise) and high-variance Gaussian white noise:
            $$V_{\text{pp}} = V_{\text{max}} - V_{\text{min}} > 100\text{mV}$$
    * **Fault Profile 3: Improper Probe Compensation Error**
        * *Implementation:* Modify the rising edge charging coefficient of the CH1 square wave to simulate capacitive under-compensation, forcing the transition edge into a sluggish, rounded exponential curve where measured rise time exceeds target specification boundaries:
            $$t_{\text{measured\_rise}} > 1.2 \times t_{\text{target\_rise}}$$

#### 3.2 Anomaly Detection & Feature Isolation Layer (`anomaly_detector.py`)
Executes an automated hybrid processing engine. It couples an unsupervised ML classifier with explicit mathematical formulas to flag outliers without external network overhead.

* **ML Classifier Layer:**
    * **Model:** `sklearn.ensemble.IsolationForest`
    * **Hyperparameters:** `n_estimators=100`, `contamination=0.03`, `random_state=42`
    * **Feature Vectors:** Ingests raw amplitude and computes a rolling variance window (Window Size = 10 samples) to catch high-frequency noise transitions natively.
    * **Optimization:** To comply with tight latency budgets, the model uses an optimized inference pattern or caches its training state to avoid full refits on rapid slider movements.
* **Deterministic Mathematical Rules Engines:**
    * The module must programmatically extract the following localized electrical engineering metrics from the ingested arrays:
        1.  **Overshoot Percentage:** Localized peak search on logical '1' bit blocks.
        2.  **Peak-to-Peak Voltage ($V_{\text{pp}}$):** Delta bounds tracking across DC arrays.
        3.  **Rise Time ($t_r$):** The precise delta time array index spacing between the $10\%$ amplitude threshold point and the $90\%$ amplitude threshold point along positive transitions.

#### 3.3 Inline Diagnostics Agent (`diagnostic_agent.py`) [OVERHAULED]
Replaces manual file export patterns with an automated, structured LLM reasoning engine connected via the official Google Gemini SDK (`google-genai`).

* **Execution Trigger:** Invoked immediately whenever the mathematical heuristics or the Isolation Forest labels a telemetry sequence state as a `Fault`.
* **Payload Structure:** Compiles the telemetry snapshot into a minified JSON schema string containing:
    * Computed Metrics (Overshoot %, $V_{\text{pp}}$, $t_r$).
    * Anomaly flags (`true`/`false`).
    * Active virtual hardware parameters (Frequency, Noise floor scale).
* **Prompt Architecture & Guardrails:** Passes the telemetry bundle into `gemini-2.5-flash` with system instructions to mimic a Rohde & Schwarz Field Applications Engineer. The output must be structured as markdown insights explaining the root cause (e.g., severe impedance mismatch reflection) and corrective test lab configurations.

#### 3.4 Unified Dashboard Interface (`app.py`)
The presentation layer transitions into a high-rate reactive console that models a physical Rohde & Schwarz RTB2000 front panel. It couples UI-driven parameter states directly to the underlying signal math pipeline.

* **Layout Geometry:**
    * **Left Column (Virtual Instrument Front Panel):** Renders physical parameter knobs using Streamlit widgets grouped by hardware subsystem definitions.
    * **Right Column (Digital Oscilloscope Screen):** Renders the real-time interactive Plotly visualization, live KPI data frames, and telemetry alert streams.
* **State & Reactive Architecture:**
    * The UI must use Streamlit session state handles (`st.session_state`) to cache the active parameter matrix. 
    * Adjusting any slider or input must instantly re-trigger the ingestion-to-detection pipeline without requiring explicit page reload buttons.
* **Real-Time HUD & GenAI Insights Panel:**
    * **Threshold Discovery HUD:** When a user transitions the sliders past the point where the engine triggers a `Fault`, the UI flashes a high-contrast notification.
    * **AI Diagnostics Viewport:** A dedicated container using `st.write_stream` or an asynchronous spinner that renders the real-time markdown summary streaming back from `diagnostic_agent.py`.

---

### 3.5 Digital Twin & Signal Control Specification
This section defines the strict mapping of Streamlit interactive widgets to programmatic signal generation parameters, mirroring standard SCPI (Standard Commands for Programmable Instruments) parameters for the R&S RTB2000 Waveform Generator subsystem.

#### 3.5.1 Baseline Signal Foundation Controls
The frontend panel must expose controls mapping to standard `WGENerator` hardware specifications to shape the core time-series array:
1.  **Carrier Frequency ($f$):** 
    * *Widget:* Slider. Range: `1.0 MHz` to `25.0 MHz`. Step: `0.1 MHz`. Default: `5.0 MHz`.
    * *Hardware Alignment:* Maps to SCPI `:WGENerator:FREQuency <frequency>`
2.  **Peak-to-Peak Amplitude ($V_{\text{pp}}$):**
    * *Widget:* Slider. Range: `20 mV` to `5.0 V`. Step: `10 mV`. Default: `3.3 V`.
    * *Hardware Alignment:* Maps to SCPI `:WGENerator:VOLTage <amplitude>`
3.  **DC Offset ($V_{\text{offset}}$):**
    * *Widget:* Number Input. Range: `-2.5 V` to `+2.5 V`. Step: `0.1 V`. Default: `0.0 V`.
    * *Hardware Alignment:* Maps to SCPI `:WGENerator:VOLTage:OFFSet <offset>`

#### 3.5.2 Interactive Fault Injection Control Matrix
A dedicated "Fault Calibration Sandbox" must allow users to continuously dial in signal degradation metrics to dynamically trace the Isolation Forest's classification thresholds.

1.  **Broadband Component Noise Floor:**
    * *Widget:* Slider. Range: `0%` to `100%` relative scaling. Maps linearly to a Gaussian standard deviation ($\sigma$) filter from `0.00V` to `0.50V`.
    * *Hardware Alignment:* Maps to SCPI `:WGENerator:NOISe:RELative <percentage>`
2.  **Edge Transition Time / Sluggish Rise ($t_r$):**
    * *Widget:* Slider. Range: `1 ns` to `100 ns`. Step: `1 ns`. Default: `10 ns`. 
    * *Behavior:* Modifies the exponential rise time constant ($\tau_{\text{slow}}$).
    * *Hardware Alignment:* Maps to SCPI `:WGENerator:FUNCtion:PULSe:ETIME <time>`
3.  **Impedance Mismatch Damping Scale:**
    * *Widget:* Slider. Range: `0.0V` (Perfect match) to `1.0V` (Severe termination reflection).
    * *Behavior:* Controls the initial amplitude coefficient ($A_{\text{ringing}}$) of the exponentially decaying cosine wave added to logic transitions.

---

### 3.6 High-Fidelity Hardware & Instrument Topology Simulator
This module expands the ingestion layer into a true "Digital Twin" by modeling the underlying physical architecture, analog constraints, and systematic error vectors of an R&S RTB Oscilloscope and ZNL Vector Network Analyzer (VNA) without physical hardware connectivity.

#### 3.6.1 Oscilloscope Hardware Core Simulation (R&S RTB Profile)
The simulation pipeline must pass all ideal signal arrays through a physical constraint layer matching the instrument specifications:
1.  **Vertical Quantization (ADC Noise):** Ideal floating-point signal vectors must be digitized through simulated 10-bit Analog-to-Digital Converter quantization steps based on the user-selected Full-Scale vertical window ($V_{\text{div}} \times 8$).
2.  **Sampling Rate & Anti-Aliasing Filter:** Implement a dynamic hardware constraint where reducing the timebase configuration alters the virtual sampling rate. If the virtual sampling rate falls below the Nyquist limit ($< 2 \times f_{\text{carrier}}$), the output array must mathematically reflect aliased down-converted frequencies.
3.  **Environmental Thermal Drift:** Simulate a continuous calibration drift component driven by an internal machine temperature coefficient variable. If the machine's virtual uptime matches an uncalibrated thermal state, values must exhibit a voltage measurement offset deviation ($\pm 2\%$ drift).

#### 3.6.2 Vector Network Analyzer Core Simulation (R&S ZNL Profile)
To realistically model antenna and RF component measurements, the simulator must calculate and expose complex S-Parameters ($S_{11}$ reflection and $S_{21}$ transmission vectors):
1.  **Systemic Error Matrix Tracking:** Implement an internal mathematical matrix that applies fixed hardware error terms directly to the simulated device-under-test (DUT) characteristics:
    * **Directivity Error ($D$):** Simulates leakage through the internal directional couplers, adding a phase-shifted error vector.
    * **Source Match Error ($S_m$):** Simulates multi-path reflections bouncing back into the instrument port.
2.  **Complex Data Conversions:** The output array must map complex impedance reflections ($Z = R + jX$) to synthesize Voltage Standing Wave Ratio (VSWR), Return Loss (dB), and raw Phase angles.

#### 3.6.3 Simulated Machine Telemetry & SCPI Command Map
The data pipeline must maintain a simulated SCPI (Standard Commands for Programmable Instruments) string parser and register model to allow future drop-in physical connectivity:
-   **Register Status Emulation:** Implement an internal register tracking state bits such as the `NOALigndata` (Self-Alignment uncalibrated flag) and hardware overvoltage alerts.
-   **Instrument Log Simulator:** Populate an array mimicking the formatting of an R&S `RSError.log` file, logging chronological entries for overvoltage saturation ($>400\text{V}$ absolute clip boundaries) or internal cooling fan airflow restriction errors.

---

### 4. Non-Functional & Computational Constraints
* **Memory Optimization:** Streamlit caching operations (`@st.cache_data`) must be wrapped around the simulation functions to block repetitive computational execution during slider changes.
* **Processing Latency:** Combined data simulation and Isolation Forest evaluation must complete execution in under $250\text{ms}$ on standard CPU profiles for datasets up to 50,000 sampling points.
* **Asynchronous GenAI Execution:** API calls to the Google Gemini SDK must handle timeouts elegantly and run non-blockingly so that the chart interactions remain fluid even while waiting for an AI analysis stream.

---

### 5. Tactical Implementation Schedule
* **Milestone 1:** Build standalone `data_simulator.py` engine with physical instrument topologies and execute matrix verification testing.
* **Milestone 2:** Implement `anomaly_detector.py` math features, verify Isolation Forest outlier classification boundaries, and establish the hybrid evaluation state.
* **Milestone 3:** Core integration of `diagnostic_agent.py` via the `google-genai` SDK, mapping telemetry parameters cleanly to structured prompt frames.
* **Milestone 4:** Wire data states inside `app.py`, implement interactive Plotly trace rendering, and deploy the dynamic real-time HUD and streaming diagnostics panel.
