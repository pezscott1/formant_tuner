# tuner/live_analyzer.py
from analysis.vowel import is_plausible_formants
import numpy as np


class LiveAnalyzer:
    """
    Pass raw engine output through:
      - plausibility filtering
      - pitch smoothing
      - formant smoothing
      - vowel label smoothing
      - profile-based scoring (NEW)
    """

    def __init__(self, engine, pitch_smoother, formant_smoother, label_smoother):
        self.engine = engine
        self.pitch_smoother = pitch_smoother
        self.formant_smoother = formant_smoother
        self.label_smoother = label_smoother

    def process_raw(self, raw_dict):
        # ---------------- Extract raw values ----------------
        f0 = raw_dict.get("f0")
        f1, f2, f3 = raw_dict.get("formants", (None, None, None))

        # ---------------- Smooth pitch ----------------
        f0_s = self.pitch_smoother.update(f0)

        # ---------------- Smooth formants ----------------
        f1_s, f2_s = self.formant_smoother.update(f1, f2)
        f3_s = f3  # no smoothing for F3

        # ---------------- Label smoothing ----------------
        vowel_raw = raw_dict.get("vowel_guess")
        vowel_conf = float(raw_dict.get("vowel_confidence", 0.0))

        # First-frame initialization
        if (getattr(self.label_smoother, "current", None)
                is None and vowel_raw is not None):
            self.label_smoother.current = vowel_raw
            if hasattr(self.label_smoother, "last"):
                self.label_smoother.last = vowel_raw
            if hasattr(self.label_smoother, "counter"):
                self.label_smoother.counter = 0
            vowel_s = vowel_raw
        else:
            vowel_s = self.label_smoother.update(vowel_raw, vowel_conf)
        # ---------------- Profile-based scoring (NEW) ----------------
        profile = getattr(self.engine, "calibrated_profile", None)
        vowel_score = 0.0
        resonance_score = 0.0
        overall = 0.0

        if (
            profile
            and vowel_s in profile
            and f1_s is not None
            and f2_s is not None
        ):
            tf1, tf2, _ = profile[vowel_s]

            df1 = abs(f1_s - tf1)
            df2 = abs(f2_s - tf2)

            # Exponential scoring — smooth, intuitive, tunable
            vowel_score = float(np.exp(-(df1 + df2) / 500))
            resonance_score = float(np.exp(-df2 / 300))
            overall = (vowel_score + resonance_score) / 2

        # ---------------- Return processed dict ----------------
        return {
            "f0": f0_s,
            "formants": (f1_s, f2_s, f3_s),
            "vowel": vowel_s,
            "confidence": vowel_conf,
            "vowel_score": vowel_score,
            "resonance_score": resonance_score,
            "overall": overall,
            "stable": getattr(self.formant_smoother, "formants_stable", False),
            "fb_f1": raw_dict.get("fb_f1"),
            "fb_f2": raw_dict.get("fb_f2"),
        }

    def reset(self):
        """Reset all smoothing state, e.g., between calibrations or sessions."""
        if hasattr(self.pitch_smoother, "current"):
            self.pitch_smoother.current = None
        if hasattr(self.pitch_smoother, "audio_buffer"):
            self.pitch_smoother.audio_buffer.clear()

        if hasattr(self.formant_smoother, "buffer"):
            self.formant_smoother.buffer.clear()

        if hasattr(self.label_smoother, "current"):
            self.label_smoother.current = None
        if hasattr(self.label_smoother, "last"):
            self.label_smoother.last = None
        if hasattr(self.label_smoother, "counter"):
            self.label_smoother.counter = 0
