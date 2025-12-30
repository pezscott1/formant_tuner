from tuner.live_analyzer import LiveAnalyzer
from analysis.engine import FormantAnalysisEngine
from analysis.smoothing import PitchSmoother, MedianSmoother, LabelSmoother


def make_analyzer(user_formants=None):
    engine = FormantAnalysisEngine()
    if user_formants is not None:
        engine.set_user_formants(user_formants)

    return LiveAnalyzer(
        engine,
        PitchSmoother(),
        MedianSmoother(),
        LabelSmoother(),
    )


def test_no_profile_means_zero_scores():
    la = make_analyzer(user_formants=None)

    # LiveAnalyzer should simply pass through engine output
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

    # LiveAnalyzer does NOT compute scores; it passes through engine output
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
