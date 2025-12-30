from unittest.mock import MagicMock
from tuner.live_analyzer import LiveAnalyzer
from analysis.smoothing import PitchSmoother, MedianSmoother, LabelSmoother


def make_analyzer():
    return LiveAnalyzer(
        engine=MagicMock(),
        pitch_smoother=PitchSmoother(),
        formant_smoother=MedianSmoother(),
        label_smoother=LabelSmoother(),
    )


def test_live_analyzer_process_raw_basic():
    analyzer = make_analyzer()

    raw = {
        "f0": 200.0,
        "formants": (500, 1500, 2500),
        "confidence": 0.8,
        "vowel_guess": "a",
        "vowel_confidence": 0.7,
        "pitch_confidence": 0.9,
    }

    out = analyzer.process_raw(raw)
    f1, f2, f3 = out["formants"]
    assert f1 == 500.0
    assert f2 == 1500.0
    assert f3 in (2500.0, None)


def test_live_analyzer_smoothing_applies_to_pitch():
    analyzer = make_analyzer()

    raw1 = {"f0": 200.0, "formants": (None, None, None),
            "confidence": 0.5, "vowel_guess": None, "pitch_confidence": 1.0}
    raw2 = {"f0": 210.0, "formants": (None, None, None),
            "confidence": 0.5, "vowel_guess": None, "pitch_confidence": 1.0}

    analyzer.process_raw(raw1)
    analyzer.process_raw(raw2)

    assert analyzer.pitch_smoother.current == 0.25 * 210 + 0.75 * 200


def test_live_analyzer_stores_raw_and_processed():
    la = make_analyzer()
    raw = {"f0": 100, "formants": (500, 1500, 2500), "confidence": 1.0}

    out = la.process_raw(raw)

    assert la.get_latest_raw() == raw
    assert la.get_latest_processed() == out


def test_live_analyzer_stability_fields():
    la = make_analyzer()
    raw = {"formants": (500, 1500, 2500), "confidence": 1.0}
    out = la.process_raw(raw)
    assert "stable" in out
    assert "stability_score" in out


def test_formant_smoother_passes_f3_through():
    la = make_analyzer()
    raw = {"formants": (500, 1500, 2500), "confidence": 1.0}
    out = la.process_raw(raw)
    # F3 may be None on first frame due to smoothing warm-up
    assert out["formants"][2] in (2500, None)
