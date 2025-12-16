import json
import logging
import os
from collections import deque
import numpy as np
import sounddevice as sd
import librosa
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Import all helpers from formant_utils
from formant_utils import (
    estimate_pitch,
    estimate_formants_lpc,
    guess_vowel,
    live_score_formants,
    resonance_tuning_score,
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
        """
        Append only when at least one of f1/f2 is a valid numeric value.
        Return (f1_med, f2_med) where missing values are returned as None.
        """
        valid_f1 = (f1 is not None) and not (
            isinstance(f1, float) and np.isnan(f1)
        )
        valid_f2 = (f2 is not None) and not (
            isinstance(f2, float) and np.isnan(f2)
        )

        if valid_f1 or valid_f2:
            # store tuple with None for missing values so medians compute correctly
            self.buffer.append(
                (f1 if valid_f1 else None, f2 if valid_f2 else None)
            )

        if not self.buffer:
            return None, None

        f1s = [x[0] for x in self.buffer if x[0] is not None]
        f2s = [x[1] for x in self.buffer if x[1] is not None]

        f1_med = float(np.median(f1s)) if f1s else None
        f2_med = float(np.median(f2s)) if f2s else None
        return f1_med, f2_med


class MedianSmoother:
    def __init__(self, size=5):
        self.f1 = deque(maxlen=size)
        self.f2 = deque(maxlen=size)
        self.f3 = deque(maxlen=size)

    @staticmethod
    def _safe_median(values):
        arr = [
            v
            for v in values
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        ]
        if not arr:
            return None
        return float(np.median(arr))

    def update(self, f1, f2, f3):
        self.f1.append(f1)
        self.f2.append(f2)
        self.f3.append(f3)
        return (
            self._safe_median(self.f1),
            self._safe_median(self.f2),
            self._safe_median(self.f3),
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
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Failed to load model from %s", model_path
                    )
            logger.info("Loaded profile %s", profile_path)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load profile %s", profile_path)

    def process_frame(
        self, audio_frame, sr, target_pitch_hz=None, debug=False
    ):
        """
        Robust frame processing: pitch smoothing, guarded formant extraction,
        smoothing/fallback, vowel guess and scoring.
        """
        frame = np.asarray(audio_frame, dtype=float).flatten()
        if frame.size == 0:
            return {
                "status": "no_data",
                "f0": None,
                "formants": (None, None, None),
            }

        # Pitch
        f0 = estimate_pitch(frame, sr)
        f0 = self.pitch_smoother.update(f0)
        if f0 is None or f0 < 50 or f0 > 600:
            f0 = target_pitch_hz or None

        # Formants: only attempt if frame has energy
        energy = float(np.mean(frame**2))
        if energy < 1e-6:
            f1 = f2 = f3 = None
        else:
            res = estimate_formants_lpc(frame, sr, debug=debug)
            # res may be (f1,f2,f3) or (f1,f2,f3,candidates)
            if isinstance(res, (tuple, list)):
                if len(res) >= 3:
                    f1, f2, f3 = res[0], res[1], res[2]
                else:
                    f1 = f2 = f3 = None
            else:
                f1 = f2 = f3 = None

        # Ensure ordering and normalize NaN -> None
        if f1 is not None and f2 is not None:
            if np.isnan(f1) or np.isnan(f2):
                f1, f2 = None, None
            elif f1 > f2:
                f1, f2 = f2, f1

        # Smoothing
        if (
            hasattr(self, "formant_smoother")
            and self.formant_smoother is not None
        ):
            f1, f2 = self.formant_smoother.update(f1, f2)

        # Last-good fallback
        if f1 is None:
            f1 = self.last_formants[0]
        if f2 is None:
            f2 = self.last_formants[1]
        if f1 is not None and f2 is not None:
            self.last_formants = (f1, f2)

        # Vowel classification
        try:
            if self.clf:
                vowel = self.clf.predict([[f1 or 0.0, f2 or 0.0, f0 or 0.0]])[
                    0
                ]
            else:
                vowel = guess_vowel(f1, f2, self.voice_type)
        except Exception:  # noqa: BLE001
            vowel = guess_vowel(f1, f2, self.voice_type)

        # Scoring
        target_formants = self.user_formants.get(vowel, (None, None, None))
        vowel_score = live_score_formants(
            target_formants, (f1, f2, f3), tolerance=50
        )
        resonance_score = resonance_tuning_score(
            (f1, f2, f3), f0, tolerance=50
        )
        overall = int(0.5 * vowel_score + 0.5 * resonance_score)

        if debug:
            logger.debug(
                "process_frame: f0=%s f1=%s f2=%s f3=%s", f0, f1, f2, f3
            )

        return {
            "status": "ok",
            "f0": f0,
            "formants": (f1, f2, f3),
            "vowel": vowel,
            "vowel_score": vowel_score,
            "resonance_score": resonance_score,
            "overall": overall,
        }

    def calibrate_live(
        self, vowels=None, sr=44100, duration=2.0, profile_name="default"
    ):
        if vowels is None:
            vowels = ["i", "e", "a", "o", "u"]
        data, labels = [], []
        self.user_formants = {}
        retries_map = {v: 0 for v in vowels}

        for v in vowels:
            try:
                recording = sd.rec(
                    int(duration * sr),
                    samplerate=sr,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()
                y = recording[:, 0].astype(float)

                f1, f2, f0 = estimate_formants_lpc(y, sr)
                # f0 fallback using librosa.yin robustly
                try:
                    f0_arr = librosa.yin(y, fmin=50, fmax=500, sr=sr)
                    f0 = float(np.nanmedian(f0_arr)) if f0_arr.size else None
                except Exception:  # noqa: BLE001
                    f0 = None

                # Only accept if formants are plausible
                ok, reason = is_plausible_formants(f1, f2, self.voice_type)
                if ok:
                    self.user_formants[v] = (f1, f2, f0)
                    data.append([f1 or 0.0, f2 or 0.0, f0 or 0.0])
                    labels.append(v)
                else:
                    logger.info(
                        "Calibration: vowel %s rejected (%s)", v, reason
                    )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to record/process vowel %s; skipping it", v
                )

        if len(data) >= 2:
            try:
                self.clf = KNeighborsClassifier(n_neighbors=1)
                self.clf.fit(data, labels)
            except Exception:  # noqa: BLE001
                logger.exception("Classifier training failed")

        profile_dict = normalize_profile_for_save(
            self.user_formants, retries_map=retries_map
        )
        dump_live_profile(profile_name, profile_dict)
        self.user_formants = profile_dict
        return True

    # --- Rendering helpers (from old analysis.py) ---
    @staticmethod
    def render_status_text(ax, status):
        ax.clear()
        if status["status"] != "ok":
            ax.text(
                0.02,
                0.8,
                "Listening… (formants not stable)",
                color="orange",
                transform=ax.transAxes,
            )
            ax.axis("off")
            return
        f0 = status["f0"]
        f1, f2, _ = status["formants"]
        guess = status.get("vowel")
        pitch_str = f"{f0:.2f} Hz" if f0 else "—"
        ax.text(
            0.02,
            0.85,
            f"You’re singing: Pitch={pitch_str}, Vowel=/{guess}/",
            color="green",
            transform=ax.transAxes,
        )
        ax.text(
            0.02,
            0.70,
            f"Measured F1/F2: {int(f1)}/{int(f2)} Hz",
            color="gray",
            transform=ax.transAxes,
        )
        ax.axis("off")

    def set_smoothing(self, enabled: bool, size: int = 5):
        self.smoothing = enabled
        self.smoother = MedianSmoother(size=size) if enabled else None

    @staticmethod
    def render_vowel_chart(
        ax, voice_type, measured_f1, measured_f2, ranked, show_breakdown=True
    ):
        ax.clear()
        VOWEL_TARGETS = {
            vt: {v: (f1, f2) for v, (f1, f2, *_rest) in vowels.items()}
            for vt, vowels in FORMANTS.items()
        }
        vt_map = VOWEL_TARGETS.get(voice_type, {})
        for v, (t1, t2) in vt_map.items():
            ax.scatter(t2, t1, c="blue", s=30)
            ax.text(t2 + 15, t1 + 15, f"/{v}/", fontsize=8, color="blue")
        ax.scatter(measured_f2, measured_f1, c="red", s=40)

        if show_breakdown and ranked:
            for v, dist in ranked[:3]:
                if v in vt_map:
                    t1, t2 = vt_map[v]
                    ax.plot(
                        [measured_f2, t2],
                        [measured_f1, t1],
                        color="red",
                        alpha=0.4,
                    )

        ax.set_xlabel("F2 (Hz)")
        ax.set_ylabel("F1 (Hz)")
        ax.set_title(f"Vowel chart ({voice_type})")
        ax.grid(True, alpha=0.2)

    @staticmethod
    def render_spectrum(ax, freqs, mags, formants, envelope=None):
        ax.clear()
        ax.plot(freqs, mags, color="black", lw=1)
        has_label = False
        if envelope is not None:
            ax.plot(
                freqs, envelope, color="red", lw=1, label="Filter envelope"
            )
            has_label = True
        for f in formants[:3]:
            if f is None or np.isnan(f):
                continue
            ax.axvline(f, color="gray", ls="--", alpha=0.5)
        ax.set_xlim(0, min(5000, freqs[-1]))
        ax.set_title("Spectrum and formants")
        if has_label:
            ax.legend(loc="upper right")

    @staticmethod
    def render_diagnostics(
        ax, status, sr, frame_len_samples, voice_type="bass"
    ):
        ax.clear()
        if status.get("status") != "ok":
            ax.text(
                0.02,
                0.9,
                "Diagnostics: awaiting stable frame",
                color="orange",
                transform=ax.transAxes,
            )
            ax.axis("off")
            return

        f0 = status.get("f0")
        f1, f2, f3 = status.get("formants", (None, None, None))
        vowel = status.get("vowel")
        conf = status.get("vowel_score", 0)
        resonance = status.get("resonance_score", 0)
        overall = status.get("overall", 0)
        dur = frame_len_samples / float(sr)

        lines = [
            f"Voice type={voice_type}",
            f"Pitch={f0:.2f} Hz | F1={int(f1) if f1 else '—'} F2={int(f2) if f2 else '—'} F3={int(f3) if f3 else '—'}",
            f"Guessed=/{vowel}/ (score={conf})",
            f"Resonance={int(resonance)} Overall={overall}",
            f"Frame: SR={sr} len={frame_len_samples} dur={dur:.2f}s",
        ]
        y = 0.95
        for line in lines:
            ax.text(0.02, y, line, color="green", transform=ax.transAxes)
            y -= 0.1
        ax.axis("off")
