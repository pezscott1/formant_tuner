# Formant Tuner

**Formant Tuner** is a scientific and educational tool for real‑time vowel analysis and voice calibration.  
It provides a PyQt‑based interface for singers, educators, clinicians, and researchers to measure vowel formants (F1, F2, F0), visualize spectrograms, and generate personalized voice profiles.

The system is built on a modern, modular architecture with high test coverage, robust DSP routines, and a clean calibration workflow.

---

## Features

### 🎙️ Live Microphone Capture
- Real‑time audio streaming via `sounddevice`
- Rolling audio buffers for stable spectrogram updates
- Safe fallback paths for missing or short audio frames

### 🔬 Spectrogram Analysis
- Powered by `librosa` with a custom `safe_spectrogram` fallback
- Automatic downsampling of time bins for smooth UI performance
- Robust handling of edge cases (short signals, FFT failures)

### 📈 Formant Extraction
- LPC‑based formant estimation (`estimate_formants_lpc`)
- Median‑based smoothing and plausibility filtering
- Vowel‑specific heuristics for difficult vowels (/o/, /u/)

### 🖼️ Dual‑Panel Visualization
**Left:** Rolling spectrogram (0–4 kHz)  
**Right:** Vowel space (F2 vs F1) with durable scatter artists

### 🗂️ Profile Management
- Save calibration results to JSON
- Load and activate profiles at runtime
- Profiles include F1, F2, F0 per vowel + metadata

### 🎨 Durable Vowel Plotting
- Each vowel has a persistent scatter artist
- Consistent color mapping across sessions
- Automatic legend management

### 📋 Text Summary Panel
- Captured formants printed in vowel‑matched colors
- Clear feedback during calibration phases

### ✅ User‑Friendly Calibration Flow
- **Prepare → Sing → Capture → Analyze**
- Countdown timer with visual cues
- Retry logic for low‑confidence captures
- Automatic progression through /i e a o u/
- Popup confirmation when calibration completes

---

## Project Structure

```
formant_tuner/
│
├── analysis/
│   ├── engine.py          # Mic pipeline, raw frame processing
│   ├── lpc.py             # LPC formant estimation
│   ├── pitch.py           # F0 estimation
│   ├── smoothing.py       # Median + window smoothing
│   ├── scoring.py         # Plausibility checks
│   └── vowel.py           # Vowel utilities
│
├── calibration/
│   ├── window.py          # PyQt5 CalibrationWindow (UI + workflow)
│   ├── session.py         # CalibrationSession (state + results)
│   ├── state_machine.py   # Phase transitions (prep/sing/capture)
│   ├── plotter.py         # Spectrogram + vowel plotting
│   └── profiles/          # Saved JSON profiles
│
├── tuner/
│   ├── controller.py      # Real‑time tuner logic
│   ├── live_analyzer.py   # Streaming analysis for tuning mode
│   └── tuner_plotter.py   # Tuner visualization
│
├── tests/                 # 90%+ coverage test suite
├── requirements.txt
└── README.md
```

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python main.py
```

---

## Usage

### Starting Calibration
1. Open the app and select **New Profile**
2. Click **Calibrate**
3. Follow the on‑screen countdown:
   - “Prepare: Sing /i/ in 3…”
4. Sing the vowel during the capture window
5. Watch the spectrogram and vowel scatter update in real time

### During Calibration
- Each vowel is captured using a rolling buffer
- Formants are extracted and validated
- Accepted values appear in the summary panel
- Low‑confidence captures trigger a retry

### Completion
- A popup announces **Calibration Complete**
- Profile is saved automatically to:

```
calibration/profiles/<profile_name>.json
```

- The new profile becomes active immediately

---

## Profile Format

Profiles are saved as JSON:

```json
{
  "i": { "f1": 265.6, "f2": 3342.1, "f0": 148.5 },
  "e": { "f1": 295.0, "f2": 3181.4, "f0": 145.7 },
  "a": { "f1": 394.4, "f2": 3024.9, "f0": 145.0 },
  "o": { "f1": 517.9, "f2": 1609.8, "f0": 154.6 },
  "u": { "f1": 355.3, "f2": 1211.2, "f0": 214.7 },
  "voice_type": "bass"
}
```

Profiles can be reloaded and applied at any time.

---

## Development Notes

### Plotting
- Spectrogram mesh is recreated when dimensions change
- Vowel scatter artists persist across updates
- Draw calls are throttled for performance

### Calibration Workflow
- `_poll_audio()` handles streaming + spectrogram updates
- `_process_capture()` handles vowel‑specific logic
- `CalibrationSession` stores results and retry reasons
- `CalibrationStateMachine` manages phase transitions

### Robustness
- All DSP routines have safe fallbacks
- All UI updates are exception‑tolerant
- Tests cover >90% of the codebase

---

## License

MIT License — free to use, modify, and distribute.