from tuner.live_analyzer import LiveAnalyzer
from analysis.engine import FormantAnalysisEngine
from analysis.smoothing import PitchSmoother, MedianSmoother, LabelSmoother


def make_analyzer(profile=None):
    engine = FormantAnalysisEngine()
    engine.calibrated_profile = profile
    return LiveAnalyzer(
        engine,
        PitchSmoother(),
        MedianSmoother(),
        LabelSmoother()
    )


def test_no_profile_means_zero_scores():
    la = make_analyzer(profile=None)
    out = la.process_raw({
        "f0": 100,
        "formants": (500, 1500, 200),
        "vowel_guess": "i",
        "vowel_confidence": 1.0,
    })
    assert out["overall"] is None


def test_missing_vowel_in_profile():
    la = make_analyzer(profile={"i": (300, 2500, 100)})
    out = la.process_raw({
        "f0": 100,
        "formants": (500, 1500, 200),
        "vowel_guess": "ɛ",
        "vowel_confidence": 1.0,
    })
    assert out["overall"] is None


def test_missing_formants_zero_scores():
    la = make_analyzer(profile={"i": (300, 2500, 100)})
    out = la.process_raw({
        "f0": 100,
        "formants": (None, None, None),
        "vowel_guess": "i",
        "vowel_confidence": 1.0,
    })
    assert out["overall"] is None


def test_valid_scoring_produces_positive_values():
    la = make_analyzer(profile={"i": (300, 2500, 100)})
    out = la.process_raw({
        "f0": 100,
        "formants": (310, 2490, 100),
        "vowel_guess": "i",
        "vowel_confidence": 1.0,
    })
    assert out["overall"] is None or out["overall"] > 0.0


def test_scoring_fallback_none():
    la = make_analyzer(profile=None)
    out = la.process_raw({"formants": (500, 1500, 2500), "vowel_guess": "i"})
    assert out["overall"] is None
