# tuner/live_analyzer.py
import numpy as np


class LiveAnalyzer:
    """
    Pass raw engine output through:
      - pitch smoothing (confidence-aware)
      - formant smoothing (confidence-aware)
      - vowel label smoothing (confidence-aware)
      - profile-based scoring
      - stability tracking
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

        # New fields from engine
        lpc_conf = float(raw_dict.get("confidence", 0.0))
        vowel_raw = raw_dict.get("vowel_guess")
        _vowel_conf = float(raw_dict.get("vowel_confidence", 0.0))

        # ---------------- Smooth pitch ----------------
        f0_s = self.pitch_smoother.update(
            f0,
            confidence=raw_dict.get("pitch_confidence", 1.0)
        )

        # ---------------- Smooth formants ----------------
        f1_s, f2_s = self.formant_smoother.update(
            f1,
            f2,
            confidence=lpc_conf
        )
        f3_s = f3  # no smoothing for F3

        # ---------------- Label smoothing ----------------
        if self.label_smoother.current is None and vowel_raw is not None:
            # First-frame initialization
            self.label_smoother.current = vowel_raw
            self.label_smoother.last = vowel_raw
            self.label_smoother.counter = 0
            vowel_s = vowel_raw
        else:
            # Use LPC confidence instead of vowel_conf
            vowel_s = self.label_smoother.update(
                vowel_raw,
                confidence=lpc_conf
            )

        # ---------------- Profile-based scoring ----------------
        profile = getattr(self.engine, "calibrated_profile", None)
        vowel_score = 0.0
        resonance_score = 0.0
        overall = 0.0

        # Fallback: if smoothing rejected formants but raw ones exist, use raw
        f1_for_score = f1_s if f1_s is not None else f1
        f2_for_score = f2_s if f2_s is not None else f2

        # ---------------- Profile-based scoring ----------------
        profile = getattr(self.engine, "calibrated_profile", None)
        vowel_score = 0.0
        resonance_score = 0.0
        overall = 0.0

        # Fallback: if smoothing rejected formants but raw ones exist, use raw
        f1_for_score = f1_s if f1_s is not None else f1
        f2_for_score = f2_s if f2_s is not None else f2

        if (
                profile
                and vowel_s in profile
                and f1_for_score is not None
                and f2_for_score is not None
        ):
            entry = profile[vowel_s]

            # Support multiple formats:
            #   - dict: {"f1":..., "f2":..., "f3":..., ...}
            #   - tuple/list: (f1, f2, f0, conf, stab) or (f1, f2, f3)
            tf1 = tf2 = tf3 = None

            if isinstance(entry, dict):
                tf1 = entry.get("f1")
                tf2 = entry.get("f2")
                tf3 = entry.get("f3")
            elif isinstance(entry, (tuple, list)):
                if len(entry) >= 3:
                    tf1, tf2, tf3 = entry[:3]
                elif len(entry) == 2:
                    tf1, tf2 = entry
                    tf3 = None

            # Only score if we actually got numeric targets
            if tf1 is not None and tf2 is not None:
                df1 = abs(f1_for_score - tf1)
                df2 = abs(f2_for_score - tf2)

                vowel_score = float(np.exp(-(df1 + df2) / 500.0))
                resonance_score = float(np.exp(-df2 / 300.0))
                overall = (vowel_score + resonance_score) / 2.0

        # ---------------- Stability ----------------
        stable = getattr(self.formant_smoother, "formants_stable", False)
        stability_score = getattr(self.formant_smoother, "_stability_score", float("inf"))

        # ---------------- Return processed dict ----------------
        return {
            "f0": f0_s,
            "formants": (f1_s, f2_s, f3_s),
            "vowel": vowel_s,
            "confidence": lpc_conf,
            "vowel_score": vowel_score,
            "resonance_score": resonance_score,
            "overall": overall,
            "stable": stable,
            "stability_score": stability_score,

            # Feedback copies
            "fb_f1": raw_dict.get("fb_f1"),
            "fb_f2": raw_dict.get("fb_f2"),

            # Pass-through debug info
            "method": raw_dict.get("method"),
            "roots": raw_dict.get("roots"),
            "peaks": raw_dict.get("peaks"),
            "lpc_order": raw_dict.get("lpc_order"),
            "lpc_debug": raw_dict.get("lpc_debug"),
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
