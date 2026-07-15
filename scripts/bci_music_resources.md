# BCI Music & EEG Sonification Resources

A compiled list of open-source GitHub repositories, frameworks, and datasets exploring the intersection of Brain-Computer Interfaces (BCIs), real-time EEG sonification, neural entrainment, and auditory decoding.

---

## 🎵 Real-Time EEG Sonification & Music Synthesis
Tools and frameworks designed to translate live brainwave signals (via LSL or direct hardware connections) into sound, MIDI, or musical parameters.

### 1. [EEGsynth](https://github.com/eegsynth/eegsynth)
*   **Language:** Python
*   **Description:** A prominent, highly modular Python framework built to convert EEG, EMG, and ECG signals into sound, MIDI, music, and analog control voltages (CV) for hardware synthesizers in real time. Perfect for biofeedback and interactive BCI musical performances.

### 2. [brain2music](https://github.com/Tallivm/brain2music)
*   **Language:** Python
*   **Description:** An AI-powered pipeline that maps EEG wave features and uses Stable Diffusion / Riffusion models to generate high-fidelity music reflecting brain states in real-time.

### 3. [EEG-pd-sonification](https://github.com/jajcayn/EEG-pd-sonification)
*   **Language:** Python & Pure Data (Pd)
*   **Description:** Provides data processing scripts in Python coupled with Pure Data patches to generate rich acoustic feedback from multi-channel EEG signals.

### 4. [NeuroZine](https://github.com/Avidabits/NeuroZine)
*   **Language:** Python / Processing / Pure Data
*   **Description:** A dedicated educational project that visualizes multi-channel EEG data and sonifies brainwaves into musical structures.

---

## 🧠 Neural Entrainment & Music Research
Scientific codebases studying how neural oscillations phase-lock to acoustic beats, speech envelopes, and musical structures.

### 1. [entrainment_eeg](https://github.com/caligiu/entrainment_eeg)
*   **Language:** Python (MNE-Python) & MATLAB (EEGLAB)
*   **Description:** Contains scripts and experiment definitions evaluating how slow neural oscillations (Delta and Theta) entrain to periodic sentence structures and auditory rhythms. Excellent reference for custom filtering and frequency-locked analysis.

### 2. [eeg-music-reconstruction](https://github.com/nevcam/eeg-music-reconstruction)
*   **Language:** Python (PyTorch)
*   **Description:** Uses deep neural networks (CNNs, Transformers) to decode and reconstruct Mel-spectrograms of music tracks directly from the EEG signals of subjects listening to them.

### 3. [Live-EEG_MusicGEN](https://github.com/hippobo/Live-EEG_MusicGEN)
*   **Language:** Python
*   **Description:** Employs machine learning models to infer emotional states from live EEG streams (like Muse) and dynamically generates matching musical compositions.

---

## 📊 Public EEG Music Datasets
Datasets of real brainwave recordings from subjects listening to different auditory tempos and musical genres.

### 1. [Music-EEG Dataset](https://AdamosDA.github.io/Music-EEG/)
*   **Link:** [GitHub Pages / Repo](https://github.com/AdamosDA/Music-EEG)
*   **Description:** Open-access database containing multi-channel EEG recordings from subjects passively listening to various genres of music. Useful for testing classification and sonification algorithms without a headset.

### 2. [MUSIN-G (OpenNeuro)](https://openneuro.org/datasets/ds003774)
*   **Link:** [OpenNeuro Portal](https://github.com/OpenNeuroDatasets/ds003774)
*   **Description:** A comprehensive dataset focused on EEG brain responses to distinct genres and tempos of music.

---

## 🔍 Key GitHub Topics to Explore
To search for newly created or updated projects:
*   [eeg-signal-processing](https://github.com/topics/eeg-signal-processing)
*   [data-sonification](https://github.com/topics/data-sonification)
*   [brain-computer-interface](https://github.com/topics/brain-computer-interface)
