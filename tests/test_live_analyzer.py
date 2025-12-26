import numpy as np
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

    assert out["f0"] == 200.0
    assert out["formants"] == (500.0, 1500.0, 2500.0)
    assert out["vowel"] == "a"
    assert out["confidence"] == 0.8


def test_live_analyzer_smoothing_applies_to_pitch():
    analyzer = make_analyzer()

    raw1 = {"f0": 200.0, "formants": (None, None, None),
            "confidence": 0.5, "vowel_guess": None, "pitch_confidence": 1.0}
    raw2 = {"f0": 210.0, "formants": (None, None, None),
            "confidence": 0.5, "vowel_guess": None, "pitch_confidence": 1.0}

    analyzer.process_raw(raw1)
    analyzer.process_raw(raw2)

    assert analyzer.pitch_smoother.current == 0.25 * 210 + 0.75 * 200
