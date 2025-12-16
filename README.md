# Formant Tuner

Formant Tuner is a scientific and educational tool for **audio analysis and vocal calibration**.  
It provides a graphical interface for singers, educators, and researchers to measure vowel formants (F1, F2, F0) in real time, visualize spectrograms, and save personalized voice profiles.

---

## Features

- 🎙️ Live microphone capture using [sounddevice](https://python-sounddevice.readthedocs.io/)  
- 🔬 Spectrogram analysis with `librosa` and custom `safe_spectrogram` routines  
- 📈 Formant extraction via LPC (`estimate_formants_lpc`)  
- 🖼️ Dual-panel visualization:
  - Left: spectrogram of captured audio
  - Right: vowel space scatter plot (F2 vs F1)
- 🗂️ Profile management:
  - Save calibration results to JSON
  - Reload and apply active profiles
- 🎨 Durable vowel plotting:
  - Each vowel plotted in a distinct color
  - Legend shows vowel labels
- 📋 Text summary:
  - Captured formants printed in matching colors above the countdown block
- ✅ User-friendly calibration flow:
  - Countdown → sing → capture → analysis
  - Retry logic for missing formants
  - Popup confirmation when calibration completes

---

## Project Structure
formant_wizard/ ├── calibration_py_qt.py   # Main PyQt5 calibration window ├── mic_analyzer.py        # MicAnalyzer class for audio capture and queuing ├── formant_utils.py       # Spectrogram, LPC, plausibility checks, profile helpers ├── PROFILES_DIR/          # Saved JSON profiles ├── requirements.txt       # Dependency list └── README.md              # This file

---

## Requirements

All dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

Usage
- Run the formant tuner window:
python formant_tuner.py
- Follow the prompts:
- If calibration is desired or needed (for a new profile), highlight New Profile, click Calibrate, and follow commands.
- Countdown appears: “Prepare: Sing /i/ in 3…”
- Sing the vowel during the capture window
- Spectrogram and vowel scatter update in real time
- Accepted formants are logged in the summary panel
- Completion:
- After all vowels are processed, “Calibration complete!” appears
- A popup prompts you to click OK to close
- Profile is saved to PROFILES_DIR/<name>_profile.json
- Active profile is set automatically

Profiles
Profiles are saved as JSON with formant values per vowel. Example:
{
  "i": [320.2, 1929.1, 2784.8],
  "e": [328.4, 1777.7, 2774.2],
  "a": [542.0, 1402.7, 2656.8],
  "o": [517.9, 1609.8, 2543.8],
  "u": [512.1, 1777.8, 2923.8]
}


Profiles can be reloaded and applied in the main app.

Development Notes
- Durable scatter: Each vowel has its own scatter artist (self._vowel_scatters) with a fixed color map.
- Summary text: Formants are appended to capture_panel in matching colors.
- Popup: QMessageBox is shown once at the end of calibration; finish() handles cleanup and closing.
- Guards: _finished prevents duplicate saves; _compute_in_flight prevents overlapping jobs.


License
MIT License — free to use, modify, and distribute.

---








