# Formant Tuner

Formant Tuner is a scientific and educational tool for real‑time vowel analysis, singer calibration, and acoustic feedback.
It provides a PyQt‑based interface for measuring vowel formants (F1, F2, F3), pitch (F0), and resonance alignment, backed by a modern DSP pipeline and a robust, test‑driven calibration workflow.

The system is designed for singers, educators, clinicians, and researchers who need accurate, reproducible, real‑time vowel diagnostics.

## ✨ Features

### 🎙️ Live Microphone Capture
- Real‑time audio streaming via sounddevice
- Rolling audio buffers for stable spectrograms
- Graceful fallbacks for short or missing frames

### 🔬 DSP Pipeline
- LPC‑based formant estimation (hybrid envelope + LPC)
- Harmonic pitch estimation with fallback strategies
- Multi‑stage smoothing for F0, F1, F2, F3
- Confidence‑aware vowel guessing
- Live scoring for tuning and resonance alignment

### 📈 Visualization
- Rolling spectrogram (0–4 kHz)
- Real‑time vowel scatter plot (F2 vs F1)
- Durable artists for stable rendering
- Color‑coded feedback in calibration and tuner modes

### 🗂️ Calibration Workflow
- Prepare → Sing → Capture → Analyze
- Automatic retries for low‑confidence frames
- Median‑based capture logic for stable vowel centers
- Expanded Mode option for advanced calibration
- Profiles saved as JSON and activated immediately

### 🎛️ Tuner Mode
- Continuous vowel tracking
- Real‑time resonance scoring
- Scatter plot + pitch + formant feedback

### 🧪 High Test Coverage
- ~85–90% coverage across DSP, calibration, UI logic, and controllers
- Structural plotter tests (no pixel diffs)
- CI‑friendly and headless‑safe
- Full PyQt6 compatibility

## 📁 Project Structure

```
formant_tuner/
├── analysis/
│   ├── engine.py
│   ├── hybrid_formants.py
│   ├── lpc.py
│   ├── pitch.py
│   ├── plausibility.py
│   ├── scoring.py
│   ├── smoothing.py
│   ├── true_envelope.py
│   ├── utils.py
│   ├── vowel_classifier.py
│   └── vowel_data.py
│
├── calibration/
│   ├── dialog.py
│   ├── plotter.py
│   ├── session.py
│   ├── state_machine.py
│   └── window.py
│
├── tuner/
│   ├── controller.py
│   ├── live_analyzer.py
│   ├── profile_controller.py
│   ├── spectrogram_view.py
│   ├── tuner_plotter.py
│   ├── window.py
│   └── window_toggle.py
│
├── profiles/
│   ├── active_profile.json
│   ├── Scott_baritone_profile.json
│   └── test_bass_profile.json
│
├── profile_viewer/
│   └── profile_viewer.py
│
├── scripts/
│   ├── pyqt5_to_pyqt6_migration.py
│   └── run_coverage.sh
│
├── tests/
│   ├── analysis/
│   ├── analyzer/
│   ├── calibration/
│   ├── engine/
│   ├── lpc/
│   ├── profile_viewer/
│   ├── profiles/
│   ├── scoring/
│   ├── smoothing/
│   ├── toggle_window/
│   └── tuner/
│
├── LEGACY/
│   └── utils/
│
├── logs/
│
├── main.py
├── conftest.py
├── requirements.txt
├── README.md
└── pytest.ini

```

## 🚀 Installation

```
pip install -r requirements.txt
python main.py
```

## 🎯 Usage

### Starting Calibration
- Launch the app
- Select New Profile or highlight an existing one
- Click Calibrate
- In the dialog:
  - Enter profile name
  - Choose voice type
  - (Optional) Enable Expanded Mode
- Follow the countdown prompts
- Sing each vowel during the capture window
- Accepted captures appear in the summary panel
- Low‑confidence captures retry automatically

Profiles are saved to:

```
calibration/profiles/<profile_name>.json
```

### Using the Tuner
- Switch to Tuner Mode
- Live vowel tracking begins immediately
- Scatter plot and scores update continuously

## 📄 Profile Format

```json
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

### DSP
- LPC order auto‑selected based on sample rate
- Median smoothing for F1/F2/F3
- Plausibility gating prevents wild outliers
- Back‑vowel heuristics for /ɔ/ and /u/

### UI
- Fully migrated to PyQt6
- All updates exception‑tolerant
- Plotting throttled for performance
- Durable artists prevent flicker
- Expanded‑mode selection now lives in the profile dialog

### Testing
- Engine wiring tests
- Smoothing + plausibility tests
- Calibration state machine tests
- Structural plotter tests
- Full PyQt6 compatibility
- 398 tests, all passing

## 📜 License

MIT License — free to use, modify, and distribute.
