from analysis.vowel import is_plausible_formants


class LiveAnalyzer:
    """Pass raw engine output through smoothing + plausibility + label smoothing."""
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
            # Clear smoother history (new API)
            self.formant_smoother.buffer.clear()

        # Smooth pitch
        f0 = self.pitch_smoother.update(f0)

        # Smooth formants (new API: only F1/F2)
        f1_s, f2_s = self.formant_smoother.update(f1, f2)

        # F3 is not smoothed by the new smoother
        f3_s = f3

        # Extract additional fields from engine output
        vowel_conf = raw_dict.get("vowel_confidence", 0.0)
        vowel_score = raw_dict.get("vowel_score", 0)
        resonance_score = raw_dict.get("resonance_score", 0)
        overall = raw_dict.get("overall", 0)

        # Stable vowel label
        vowel = raw_dict.get("vowel_guess")

        # Initialize smoother on first value
        if self.label_smoother.current is None and vowel is not None:
            self.label_smoother.current = vowel
            self.label_smoother.last = vowel
            return {
                "f0": f0,
                "formants": (f1_s, f2_s, f3_s),
                "vowel": vowel,
                "confidence": vowel_conf,
                "vowel_score": vowel_score,
                "resonance_score": resonance_score,
                "overall": overall,
                "fb_f1": raw_dict.get("fb_f1"),
                "fb_f2": raw_dict.get("fb_f2"),
            }

        # Otherwise smooth normally
        vowel = self.label_smoother.update(vowel)

        return {
            "f0": f0,
            "formants": (f1_s, f2_s, f3_s),
            "vowel": vowel,
            "confidence": vowel_conf,
            "vowel_score": vowel_score,
            "resonance_score": resonance_score,
            "overall": overall,
            "fb_f1": raw_dict.get("fb_f1"),
            "fb_f2": raw_dict.get("fb_f2"),
        }
