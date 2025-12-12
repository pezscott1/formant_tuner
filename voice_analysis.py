import json, logging, os
from collections import deque
import numpy as np
import sounddevice as sd
import librosa
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime, timezone

# Import all helpers from formant_utils
from formant_utils import (
    estimate_pitch,
    estimate_formants_lpc,
    guess_vowel,
    live_score_formants,
    resonance_tuning_score,
    robust_guess,
    plausibility_score,
    is_plausible_formants,
    normalize_profile_for_save,
    dump_live_profile,
)

from vowel_data import FORMANTS, VOWEL_MAP

logger = logging.getLogger(__name__)


class PitchSmoother:
    def __init__(self, size=5):
        self.values = deque(maxlen=size)

    def update(self, f0):
        if f0 is None:
            return None
        self.values.append(f0)
        return float(np.median(self.values))


class FormantSmoother:
    def __init__(self, size=5):
        self.buffer = deque(maxlen=size)

    def update(self, f1, f2):
        if f1 and f2:
            self.buffer.append((f1, f2))
        if self.buffer:
            f1s, f2s = zip(*self.buffer)
            return np.median(f1s), np.median(f2s)
        return f1, f2


class MedianSmoother:
    def __init__(self, size=5):
        self.f1 = deque(maxlen=size)
        self.f2 = deque(maxlen=size)
        self.f3 = deque(maxlen=size)

    def update(self, f1, f2, f3):
        self.f1.append(f1)
        self.f2.append(f2)
        self.f3.append(f3)
        return (
            float(np.median(self.f1)),
            float(np.median(self.f2)),
            float(np.median(self.f3)),
        )


class Analyzer:
    def __init__(self, voice_type="bass", smoothing=True, smooth_size=5):
        self.voice_type = voice_type
        self.pitch_smoother = PitchSmoother(size=5)
        self.formant_smoother = FormantSmoother(size=5)
        self.last_formants = (None, None)
        self.clf = None
        self.user_formants = FORMANTS.get(voice_type, VOWEL_MAP)
        self.smoothing = smoothing
        self.smoother = MedianSmoother(size=smooth_size) if smoothing else None
        self.progress_history = {v: [] for v in FORMANTS.keys()}

    def load_profile(self, profile_path, model_path=None):
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                self.user_formants = json.load(f)
            if model_path and os.path.exists(model_path):
                try:
                    self.clf = KNeighborsClassifier(n_neighbors=1)
                    self.clf = joblib.load(model_path)
                except Exception:
                    logger.exception("Failed to load model from %s", model_path)
            logger.info("Loaded profile %s", profile_path)
        except Exception:
            logger.exception("Failed to load profile %s", profile_path)

    def process_frame(self, audio_frame, sr, target_pitch_hz=None):
        f0 = estimate_pitch(audio_frame, sr)
        f0 = self.pitch_smoother.update(f0)
        if f0 is None or f0 < 50 or f0 > 600:
            f0 = target_pitch_hz or 220.0

        f1, f2 = estimate_formants_lpc(audio_frame, sr)
        if f1 and f2 and f1 > f2:
            f1, f2 = f2, f1
        f1, f2 = self.formant_smoother.update(f1, f2)

        if f1 is None: f1 = self.last_formants[0]
        if f2 is None: f2 = self.last_formants[1]
        if f1 and f2: self.last_formants = (f1, f2)

        vowel = self.clf.predict([[f1 or 0, f2 or 0, f0]])[0] if self.clf else guess_vowel(f1, f2, self.voice_type)

        target_formants = self.user_formants.get(vowel, (None, None, None))
        vowel_score = live_score_formants(target_formants, (f1, f2, None), tolerance=50)
        resonance_score = resonance_tuning_score((f1, f2, None), f0, tolerance=50)
        overall = int(0.5 * vowel_score + 0.5 * resonance_score)

        return {
            "status": "ok",
            "f0": f0,
            "formants": (f1, f2, None),
            "vowel": vowel,
            "vowel_score": vowel_score,
            "resonance_score": resonance_score,
            "overall": overall,
        }

    def calibrate_live(self, vowels=["i","e","a","o","u"], sr=44100, duration=2.0, profile_name="default"):
        data, labels = [], []
        self.user_formants = {}
        retries_map = {v: 0 for v in vowels}

        for v in vowels:
            try:
                recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
                sd.wait()
                y = recording[:, 0].astype(float)

                f1, f2 = estimate_formants_lpc(y, sr)
                try:
                    f0 = float(librosa.yin(y, fmin=50, fmax=500, sr=sr).mean())
                except Exception:
                    f0 = float("nan")

                self.user_formants[v] = (f1, f2, f0)
                data.append([f1 or 0.0, f2 or 0.0, f0 or 0.0])
                labels.append(v)
            except Exception:
                logger.exception("Failed to record/process vowel %s; skipping it", v)

        if len(data) >= 2:
            try:
                self.clf = KNeighborsClassifier(n_neighbors=1)
                self.clf.fit(data, labels)
            except Exception:
                logger.exception("Classifier training failed")

        profile_dict = normalize_profile_for_save(self.user_formants, retries_map=retries_map)
        dump_live_profile(profile_name, profile_dict)
        self.user_formants = profile_dict
        return True


    # --- Rendering helpers (from old analysis.py) ---
    def render_status_text(self, ax, status):
        ax.clear()
        if status["status"] != "ok":
            ax.text(0.02, 0.8, "Listening… (formants not stable)", color="orange", transform=ax.transAxes)
            ax.axis("off"); return
        f0 = status["f0"]
        f1, f2, _ = status["formants"]
        guess = status.get("vowel")
        pitch_str = f"{f0:.2f} Hz" if f0 else "—"
        ax.text(0.02, 0.85, f"You’re singing: Pitch={pitch_str}, Vowel=/{guess}/", 
                color="green", transform=ax.transAxes)
        ax.text(0.02, 0.70, f"Measured F1/F2: {int(f1)}/{int(f2)} Hz", color="gray", transform=ax.transAxes)
        ax.axis("off")


    def set_smoothing(self, enabled: bool, size: int = 5):
        self.smoothing = enabled
        self.smoother = MedianSmoother(size=size) if enabled else None

    def render_vowel_chart(self, ax, voice_type, measured_f1, measured_f2, ranked, show_breakdown=True):
        ax.clear()
        VOWEL_TARGETS = {
            vt: {v: (f1, f2) for v, (f1, f2, *_rest) in vowels.items()}
            for vt, vowels in FORMANTS.items()
        }
        vt_map = VOWEL_TARGETS.get(voice_type, {})
        for v, (t1, t2) in vt_map.items():
            ax.scatter(t2, t1, c="blue", s=30)
            ax.text(t2+15, t1+15, f"/{v}/", fontsize=8, color="blue")
        ax.scatter(measured_f2, measured_f1, c="red", s=40)

        if show_breakdown and ranked:
            for v, dist in ranked[:3]:
                if v in vt_map:
                    t1, t2 = vt_map[v]
                    ax.plot([measured_f2, t2], [measured_f1, t1], color="red", alpha=0.4)

        ax.set_xlabel("F2 (Hz)")
        ax.set_ylabel("F1 (Hz)")
        ax.set_title(f"Vowel chart ({voice_type})")
        ax.grid(True, alpha=0.2)


    def render_spectrum(self, ax, freqs, mags, formants, envelope=None):
        ax.clear()
        ax.plot(freqs, mags, color="black", lw=1)
        has_label = False
        if envelope is not None:
            ax.plot(freqs, envelope, color="red", lw=1, label="Filter envelope")
            has_label = True
        for f in formants[:3]:
            if f is None or np.isnan(f):
                continue
            ax.axvline(f, color="gray", ls="--", alpha=0.5)
        ax.set_xlim(0, min(5000, freqs[-1]))
        ax.set_title("Spectrum and formants")
        if has_label:
            ax.legend(loc="upper right")


    def render_diagnostics(self, ax, status, sr, frame_len_samples, voice_type="bass"):
        ax.clear()
        if status["status"] != "ok":
            ax.text(0.02, 0.9, "Diagnostics: awaiting stable frame", color="orange", transform=ax.transAxes)
            ax.axis("off")
            return

        f0 = status["f0"]
        f1, f2, f3 = status["formants"]
        guess, conf, next_best, resonance = (
            status["guess"], status["conf"], status["next"], status["resonance"]
        )
        overall = status["overall"]
        penalty = status["penalty"]
        dur = frame_len_samples / sr

        lines = [
            f"Voice type={voice_type}",
            f"Pitch={f0:.2f} Hz | F1={int(f1)} F2={int(f2)} F3={int(f3)}",
            f"Guessed=/{guess}/ (conf={conf:.2f}) Next=/{next_best}/",
            f"Resonance={int(resonance)} Penalty={penalty:.2f} Overall={overall}",
            f"Frame: SR={sr} len={frame_len_samples} dur={dur:.2f}s",
        ]
        y = 0.95
        for line in lines:
            ax.text(0.02, y, line, color="green", transform=ax.transAxes)
            y -= 0.1
        ax.axis("off")
