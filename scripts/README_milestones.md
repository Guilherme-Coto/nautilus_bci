# BCI g.Nautilus EEG Streaming & Real-Time Visualization

This project sets up a high-performance pipeline to stream 32 channels of EEG data from a **g.Nautilus BCI device** (`NP-2026.05.01`) to the **Lab Streaming Layer (LSL)**, perform real-time signal processing, and visualize brainwave activity.

---

## 🚀 Quick Start (How to Resume)

To run the full suite, open separate terminal windows and run the following commands:

### Step 1: Start the LSL Streamer
Resolves the g.Nautilus connection and broadcasts the 32 channels of EEG data.
```powershell
cd "C:\Users\VR\Desktop\BCI gtec\g.tec_Suite_2024_1.24.01_Win64\scripts"
uv run python gds_to_lsl.py
```

### Step 2: Choose a Visualizer

* **Option A: Real-Time 32-Channel Waveform Plot (Python)**
  Displays all 32 channels stacked vertically with a gain/spacing slider.
  ```powershell
  cd "C:\Users\VR\Desktop\BCI gtec\g.tec_Suite_2024_1.24.01_Win64\scripts"
  uv run python lsl_viewer.py
  ```

* **Option B: Brain Wave Band Feature Analyzer (Python)**
  Displays relative powers (Delta, Theta, Alpha, Beta) and a rolling concentration index.
  ```powershell
  cd "C:\Users\VR\Desktop\BCI gtec\g.tec_Suite_2024_1.24.01_Win64\scripts"
  uv run python eeg_features.py
  ```

* **Option D: Interactive Brain Rhythm Explorer & Simulator (Python)**
  An educational visualizer designed specifically for students to explore brain rhythms (Delta, Theta, Alpha, Beta, Gamma) in real time. Can stream live from g.Nautilus LSL, or automatically run in **Simulation Mode** (where students can trigger blinks, muscle clenches, and adjust alpha/beta levels using on-screen controls to see signal features update instantly). Features biofeedback challenges (Alpha relaxation, concentration, and stability games) with progress meters and state classification cards!
  ```powershell
  cd "C:\Users\VR\Desktop\BCI gtec\g.tec_Suite_2024_1.24.01_Win64\scripts"
  uv run python rhythm_visualizer.py
  ```

* **Option C: High-Performance C# Desktop Application**
  A WinForms app featuring hardware-accelerated rendering and a live FFT spectrum.
  ```powershell
  cd "C:\Users\VR\Desktop\BCI gtec\g.tec_Suite_2024_1.24.01_Win64\EegVisualizer32"
  dotnet run
  ```

---

## 🏆 Milestones Achieved

### 📦 1. Environment & Dependency Setup
* Initialized a python environment managed by **`uv`** inside the g.tec Suite directory.
* Configured dependencies: `pylsl`, `numpy`, `scipy`, `pyqtgraph`, and `PySide6`.
* Configured and compiled the C# WinForms project (`EegVisualizer32`) targeting `.NET 10` with `ScottPlot 5`, `FftSharp`, and `SharpLSL`.

### 📡 2. Resilient LSL Streamer (`gds_to_lsl.py`)
* Connects to the g.Nautilus device using non-exclusive locks (`open_exclusively=False`) so the stream recovers gracefully after crashes without locking up the GDS service.
* Dynamically unpacks g.Nautilus channel labels (e.g. `Fp1`, `Fp2`, `Cz`) and streams them to LSL.

### 📈 3. Real-Time 32-Ch Signal Plot (`lsl_viewer.py`)
* Renders 32 channels of live EEG in a vertically stacked layout with a custom channel spacing slider.
* Upgraded to a steep **4th-order Butterworth bandpass filter (2–45 Hz)** and a **50 Hz Notch filter** to keep signals flat, flat-lined, and clean of slow DC baseline drifts.

### 🧠 4. Vectorized Feature Analyzer (`eeg_features.py`)
* Computes live relative powers for Delta (0.5–4Hz), Theta (4–8Hz), Alpha (8–12Hz), and Beta (12–30Hz) bands.
* **Vectorized the entire pipeline:** Removed the channel-by-channel loop and replaced it with vectorized NumPy FFT operations (`np.fft.rfft(..., axis=0)`). Computations now run in `< 1ms`.
* Integrated the **4th-order Butterworth filter** to suppress DC/Delta drift dominance.

### 🎮 5. Interactive Rhythm Explorer & Simulator (`rhythm_visualizer.py`)
* Designed a beautiful dark-themed Qt6 education application to visualize relative power of brainwaves (Delta, Theta, Alpha, Beta, Gamma).
* Implemented a dual source design: automatically resolves network LSL EEG streams or falls back to an **Interactive Simulator Mode** so students can learn without device setup.
* Added interactive simulation injection features: controls for base noise, 50 Hz power line interference, Alpha waves amplitude, Beta waves amplitude, and instant action buttons to trigger frontal Eye Blinks (Delta) and temporal Muscle Clenches (Gamma).
* Created a **Biofeedback Challenge game** featuring 3 student tasks: "Alpha Relaxation", "Beta Concentration Focus", and "Perfect Signal Stability" with score tracking, hold duration timer, and visual win notifications.

### 💻 6. C# WinForms Visualizer (`EegVisualizer32`)
* Created a lightweight application using `ScottPlot 5` to show live waterfall signal plots and frequency spectrums.
* Resolved type conflicts (`ScottPlot` vs. `System.Drawing`) and off-by-one array bounds errors in the FFT rendering loop.

---

## 🔍 Core Files Summary

| File Path | Language | Purpose |
| :--- | :--- | :--- |
| [`scripts/gds_to_lsl.py`](file:///C:/Users/VR/Desktop/BCI%20gtec/g.tec_Suite_2024_1.24.01_Win64/scripts/gds_to_lsl.py) | Python | Stream BCI data to LSL |
| [`scripts/lsl_viewer.py`](file:///C:/Users/VR/Desktop/BCI%20gtec/g.tec_Suite_2024_1.24.01_Win64/scripts/lsl_viewer.py) | Python | 32-Channel Stacked waveform viewer (4th-order Butterworth) |
| [`scripts/eeg_features.py`](file:///C:/Users/VR/Desktop/BCI%20gtec/g.tec_Suite_2024_1.24.01_Win64/scripts/eeg_features.py) | Python | Vectorized Alpha/Beta band power analyzer |
| [`scripts/rhythm_visualizer.py`](file:///C:/Users/VR/Desktop/BCI%20gtec/g.tec_Suite_2024_1.24.01_Win64/scripts/rhythm_visualizer.py) | Python | Interactive brain rhythm visualizer & simulator dashboard |
| [`EegVisualizer32/Form1.cs`](file:///C:/Users/VR/Desktop/BCI%20gtec/g.tec_Suite_2024_1.24.01_Win64/EegVisualizer32/Form1.cs) | C# | High-performance multithreaded WinForms visualizer logic |
