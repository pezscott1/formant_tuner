# tests/test_live_analyzer.py
import numpy as np
from unittest.mock import MagicMock

from tuner.live_analyzer import LiveAnalyzer
from analysis.smoothing import MedianSmoother, PitchSmoother, LabelSmoother


def make_analyzer():
    engine = MagicMock()
    engine.voice_type = "bass"
    return LiveAnalyzer(
        engine=engine,
        pitch_smoother=PitchSmoother(),
        formant_smoother=MedianSmoother(),
        label_smoother=LabelSmoother(),
    )


def test_process_raw_basic_fields_present():
    la = make_analyzer()

    raw = {
        "f0": 120,
        "formants": (500, 1500, 2500),
        "vowel_guess": "a",
        "vowel_confidence": 0.8,
        "fb_f1": 10,
        "fb_f2": -5,
    }

    out = la.process_raw(raw)

    assert out["f0"] is not None
    assert out["formants"] == (500, 1500, 2500)
    assert out["vowel"] == "a"
    assert out["confidence"] == 0.8
    assert out["fb_f1"] == 10
    assert out["fb_f2"] == -5


def test_process_raw_missing_formants():
    la = make_analyzer()

    raw = {
        "f0": 120,
        "formants": (None, None, None),
        "vowel_guess": None,
    }

    out = la.process_raw(raw)

    assert out["formants"] == (None, None, None)
    assert out["vowel"] is None


def test_process_raw_plausibility_filtering():
    la = make_analyzer()

    raw = {
        "f0": 120,
        "formants": (50, 80, 2500),
        "vowel_guess": "a",
    }

    out = la.process_raw(raw)

    f1, f2, f3 = out["formants"]
    assert f1 is None
    assert f2 is None
    assert f3 == 2500


def test_label_smoothing_applies():
    la = make_analyzer()

    raw1 = {"f0": 120, "formants": (500, 1500, 2500), "vowel_guess": "a"}
    raw2 = {"f0": 125, "formants": (520, 1480, 2500), "vowel_guess": "a"}
    raw3 = {"f0": 130, "formants": (510, 1490, 2500), "vowel_guess": "e"}

    out1 = la.process_raw(raw1)
    out2 = la.process_raw(raw2)
    out3 = la.process_raw(raw3)

    assert out1["vowel"] == "a"
    assert out2["vowel"] == "a"
    assert out3["vowel"] in ("a", "e")


def test_pitch_smoothing_changes_value():
    la = make_analyzer()

    raw1 = {"f0": 100, "formants": (500, 1500, 2500)}
    raw2 = {"f0": 200, "formants": (500, 1500, 2500)}

    out1 = la.process_raw(raw1)
    out2 = la.process_raw(raw2)

    assert out1["f0"] != 100 or out2["f0"] != 200