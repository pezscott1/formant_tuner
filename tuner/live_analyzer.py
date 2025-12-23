# tuner/live_analyzer.py
from analysis.vowel import is_plausible_formants


class LiveAnalyzer:
    """
    Pass raw engine output through:
      - plausibility filtering
      - pitch smoothing
      - formant smoothing
      - vowel label smoothing

    Input:  raw_dict from FormantAnalysisEngine.process_frame()
    Output: processed dict suitable for UI consumption.
    """

    def __init__(self, engine, pitch_smoother, formant_smoother, label_smoother):
        self.engine = engine
        self.pitch_smoother = pitch_smoother
        self.formant_smoother = formant_smoother
        self.label_smoother = label_smoother

    def process_raw(self, raw_dict):
        # Extract raw values
        f0 = raw_dict.get("f0")
        f1, f2, f3 = raw_dict.get("formants", (None, None, None))

        # Plausibility BEFORE smoothing
        ok, _ = is_plausible_formants(f1, f2, self.engine.voice_type)
        if not ok:
            # Reset formants completely
            f1 = None
            f2 = None
            # Clear smoother history (if present)
            if hasattr(self.formant_smoother, "buffer"):
                self.formant_smoother.buffer.clear()

        # Smooth pitch
        f0_s = self.pitch_smoother.update(f0)

        # Smooth formants (F1/F2)
        f1_s, f2_s = self.formant_smoother.update(f1, f2)

        # F3 is not smoothed
        f3_s = f3

        # Extract additional fields from engine output
        vowel_conf = float(raw_dict.get("vowel_confidence", 0.0))
        vowel_score = raw_dict.get("vowel_score", 0)
        resonance_score = raw_dict.get("resonance_score", 0)
        overall = raw_dict.get("overall", 0)

        # Stable vowel label from engine
        vowel = raw_dict.get("vowel_guess")

        # Initialize label smoother on first value
        if getattr(self.label_smoother, "current", None) is None and vowel is not None:
            self.label_smoother.current = vowel
            if hasattr(self.label_smoother, "last"):
                self.label_smoother.last = vowel
            if hasattr(self.label_smoother, "counter"):
                self.label_smoother.counter = 0

            return {
                "f0": f0_s,
                "formants": (f1_s, f2_s, f3_s),
                "vowel": vowel,
                "confidence": vowel_conf,
                "vowel_score": vowel_score,
                "resonance_score": resonance_score,
                "overall": overall,
                "fb_f1": raw_dict.get("fb_f1"),
                "fb_f2": raw_dict.get("fb_f2"),
            }

        # Otherwise smooth the label normally
        vowel_s = self.label_smoother.update(vowel, vowel_conf)

        return {
            "f0": f0_s,
            "formants": (f1_s, f2_s, f3_s),
            "vowel": vowel_s,
            "confidence": vowel_conf,
            "vowel_score": vowel_score,
            "resonance_score": resonance_score,
            "overall": overall,
            "fb_f1": raw_dict.get("fb_f1"),
            "fb_f2": raw_dict.get("fb_f2"),
        }

    def reset(self):
        """Reset all smoothing state, e.g., between calibrations or sessions."""
        # Pitch smoother
        if hasattr(self.pitch_smoother, "current"):
            self.pitch_smoother.current = None
        if hasattr(self.pitch_smoother, "audio_buffer"):
            self.pitch_smoother.audio_buffer.clear()

        # Formant smoother
        if hasattr(self.formant_smoother, "buffer"):
            self.formant_smoother.buffer.clear()

        # Label smoother
        if hasattr(self.label_smoother, "current"):
            self.label_smoother.current = None
        if hasattr(self.label_smoother, "last"):
            self.label_smoother.last = None
        if hasattr(self.label_smoother, "counter"):
            self.label_smoother.counter = 0
