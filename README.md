# Formant Tuner

Formant Tuner is a scientific and educational tool for real‑time vowel analysis, singer calibration, and acoustic feedback.
It provides a PyQt‑based interface for measuring vowel formants (F1, F2, F3), pitch (F0), and resonance alignment, with a modern DSP pipeline and a robust calibration workflow.

The system is designed for singers, educators, clinicians, and researchers who want accurate, reproducible, real‑time vowel diagnostics.

## ✨ Features

### 🎙️ Live Microphone Capture
- Real‑time audio streaming via sounddevice
- Rolling audio buffers for stable spectrograms
- Safe fallbacks for short or missing frames

### 🔬 DSP Pipeline
- LPC‑based formant estimation (analysis/lpc.py)
- Harmonic pitch estimation (analysis/pitch.py)
- Multi‑stage smoothing (analysis/smoothing.py)
- Robust vowel guessing (analysis/vowel.py)
- Live scoring for tuning (analysis/scoring.py)

### 📈 Visualization
- Rolling spectrogram (0–4 kHz)
- Real‑time vowel scatter plot (F2 vs F1)
- Durable scatter artists for each vowel
- Color‑coded feedback in calibration and tuning modes

### 🗂️ Calibration Workflow
- Prepare → Sing → Capture → Analyze
- Automatic retries for low‑confidence frames
- Median‑based capture logic
- Saves calibrated F1/F2/F0 per vowel
- Profiles stored as JSON and activated immediately

### 🎛️ Tuner Mode
- Continuous vowel tracking
- Real‑time resonance scoring
- Live feedback for singers and educators

### 🧪 High Test Coverage
- Pytest suite covering DSP, smoothing, plausibility, engine wiring, calibration logic, and UI state transitions
- No brittle pixel‑tests; structural tests for plotters
- CI‑friendly, headless‑safe

## 📁 Project Structure
```
formant_tuner/
│
├── analysis/
│   ├── engine.py              # unified formant analysis engine
│   ├── lpc.py                 # LPC + envelope + cepstral formants
│   ├── pitch.py               # pitch estimation (HPS + fallback)
│   ├── vowel.py               # vowel ranges, guessing, plausibility
│   ├── vowel_data.py          # reference formants + pitch ranges
│   ├── scoring.py             # plausibility + tuning + live scoring
│   ├── smoothing.py           # all smoothing utilities
│   └── utils.py               # helpers
│
├── calibration/
│   ├── session.py             # calibration logic (retry, capture)
│   ├── plotter.py             # spectrogram + vowel scatter
│   ├── state_machine.py       # prep/sing/capture phases
│   ├── dialog.py              # confirmation + error dialogs
│   └── window.py              # calibration UI
│
├── tuner/
│   ├── controller.py
│   ├── live_analyzer.py       # smoothing + plausibility + UI updates
│   ├── profile_controller.py  # profile loading/activation
│   ├── tuner_plotter.py       # tuner visualization
│   └── window.py              # thin PyQt wrapper
│
├── utils/
│   └── music_utils.py         # musical helpers (note names, etc.)
│
├── tests/                     # pytest suite (85–90% coverage)
│
├── main.py                    # application entry point
├── requirements.txt
├── pytest.ini
├── structure.txt
└── README.md
```
## 🚀 Installation
```
pip install -r requirements.txt
python main.py
```
## 🎯 Usage

### Starting Calibration
- Launch the app
- Choose New Profile
- Click Calibrate
- Follow the countdown prompts
- Sing each vowel during the capture window
- Accepted captures appear in the summary panel
- Low‑confidence captures trigger retries automatically

Profiles saved to:
calibration/profiles/<profile_name>.json

### Using the Tuner
- Switch to Tuner Mode
- Live vowel tracking begins immediately
- Scatter plot and scores update continuously

## 📄 Profile Format

``` 
{
  "i":  { "f1": 280.0, "f2": 2852.8, "f0": 145.0 },
  "ɛ":  { "f1": 595.6, "f2": 2794.9, "f0": 139.1 },
  "ɑ":  { "f1": 722.6, "f2": 2374.0, "f0": 117.1 },
  "ɔ":  { "f1": 642.4, "f2": 2680.9, "f0": 138.8 },
  "u":  { "f1": 653.7, "f2": 2823.9, "f0": 127.3 },
  "voice_type": "baritone"
}
```


## 🧠 Development Notes

DSP
- LPC order auto‑selected based on sample rate
- Median smoothing for F1/F2/F3
- Plausibility gating prevents wild outliers
- Back‑vowel heuristics for /ɔ/ and /u/

UI
- All PyQt updates are exception‑tolerant
- Plotting throttled for performance
- Durable artists prevent flicker

Testing
- Engine wiring tests
- Smoothing + plausibility tests
- Calibration state machine tests
- Structural plotter tests (no pixel diffs)
- High‑coverage CI‑friendly suite

## 📜 License

MIT License — free to use, modify, and distribute.