from analysis.engine import FormantAnalysisEngine
from analysis.smoothing import PitchSmoother, MedianSmoother, LabelSmoother


class FakeLiveAnalyzer:
    """
    Minimal, thread-free stand-in for LiveAnalyzer.
    Only implements process_raw and stores latest frames.
    """

    def __init__(self, engine, pitch_smoother, formant_smoother, label_smoother):
        self.engine = engine
        self.pitch_smoother = pitch_smoother
        self.formant_smoother = formant_smoother
        self.label_smoother = label_smoother
        self._latest_raw = None
        self._latest_processed = None

    def process_raw(self, raw_dict):
        # Store raw
        self._latest_raw = raw_dict

        # Extract raw values
        f0_raw = raw_dict.get("f0")
        f1_raw, f2_raw, f3_raw = raw_dict.get("formants", (None, None, None))
        vowel_raw = raw_dict.get("vowel") or raw_dict.get("vowel_guess")
        lpc_conf = float(raw_dict.get("confidence", 0.0))

        # Pitch smoothing
        f0_s = self.pitch_smoother.update(f0_raw, confidence=lpc_conf)

        # Formant smoothing
        f1_s, f2_s, f3_s = self.formant_smoother.update(
            f1=f1_raw, f2=f2_raw, f3=f3_raw, confidence=lpc_conf
        )

        # Vowel smoothing
        vowel_s = self.label_smoother.update(vowel_raw, confidence=lpc_conf)

        processed = {
            "f0": f0_s,
            "formants": (f1_s, f2_s, f3_s),
            "vowel": vowel_s,
            "vowel_guess": vowel_raw,
            "confidence": lpc_conf,
            "vowel_score": raw_dict.get("vowel_score"),
            "resonance_score": raw_dict.get("resonance_score"),
            "overall": raw_dict.get("overall"),
            "stable": getattr(self.formant_smoother, "formants_stable", False),
            "stability_score": getattr(
                self.formant_smoother, "_stability_score", float("inf")
            ),
        }

        self._latest_processed = processed
        return processed


def make_analyzer(user_formants=None):
    engine = FormantAnalysisEngine()
    if user_formants is not None:
        engine.set_user_formants(user_formants)

    return FakeLiveAnalyzer(
        engine,
        PitchSmoother(),
        MedianSmoother(),
        LabelSmoother(),
    )


def test_no_profile_means_zero_scores():
    la = make_analyzer(user_formants=None)
    out = la.process_raw({
        "f0": 100,
        "formants": (500, 1500, 200),
        "vowel_guess": "i",
        "vowel_confidence": 1.0,
        "overall": 0.0,
    })
    assert out["overall"] == 0.0


def test_missing_vowel_in_profile():
    la = make_analyzer(user_formants={"i": {"f1": 300, "f2": 2500, "f3": 100}})
    out = la.process_raw({
        "f0": 100,
        "formants": (500, 1500, 200),
        "vowel_guess": "ɛ",
        "vowel_confidence": 1.0,
        "overall": 0.0,
    })
    assert out["overall"] == 0.0


def test_missing_formants_zero_scores():
    la = make_analyzer(user_formants={"i": {"f1": 300, "f2": 2500, "f3": 100}})
    out = la.process_raw({
        "f0": 100,
        "formants": (None, None, None),
        "vowel_guess": "i",
        "vowel_confidence": 1.0,
        "overall": 0.0,
    })
    assert out["overall"] == 0.0


def test_valid_scoring_passes_through_engine():
    la = make_analyzer(user_formants={"i": {"f1": 300, "f2": 2500, "f3": 100}})
    out = la.process_raw({
        "f0": 100,
        "formants": (310, 2490, 100),
        "vowel_guess": "i",
        "vowel_confidence": 1.0,
        "overall": 42.0,
    })
    assert out["overall"] == 42.0


def test_scoring_fallback_zero():
    la = make_analyzer(user_formants=None)
    out = la.process_raw({
        "formants": (500, 1500, 2500),
        "vowel_guess": "i",
        "overall": 0.0,
    })
    assert out["overall"] == 0.0
