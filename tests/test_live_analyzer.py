# tests/test_live_analyzer.pynp
from unittest.mock import MagicMock

from tuner.live_analyzer import LiveAnalyzer
from analysis.smoothing import MedianSmoother, PitchSmoother, LabelSmoother


class DummyEngine:
    """
    Minimal stub engine for testing LiveAnalyzer.
    Only provides the attributes LiveAnalyzer actually uses.
    """

    def __init__(self, voice_type="bass"):
        self.voice_type = voice_type
        self._latest_raw = None

    def get_latest_raw(self):
        return self._latest_raw


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


def test_live_analyzer_first_frame_initializes_label():
    engine = DummyEngine()
    la = LiveAnalyzer(engine, PitchSmoother(), MedianSmoother(), LabelSmoother())

    raw = {"f0": 200, "formants": (500, 1500, 2500), "vowel_guess": "ɑ", "vowel_confidence": 1.0}
    out = la.process_raw(raw)

    assert out["vowel"] == "ɑ"
    assert la.label_smoother.current == "ɑ"


def test_live_analyzer_plausibility_resets_formant_smoother(monkeypatch):
    engine = DummyEngine()
    la = LiveAnalyzer(engine, PitchSmoother(), MedianSmoother(), LabelSmoother())

    # Force plausibility to fail
    monkeypatch.setattr("analysis.vowel.is_plausible_formants", lambda f1, f2, vt: (False, "bad"))

    raw = {"f0": 200, "formants": (500, 1500, 2500), "vowel_guess": "ɑ", "vowel_confidence": 1.0}
    la.formant_smoother.buffer.extend([(500, 1500)])

    la.process_raw(raw)

    assert len(la.formant_smoother.buffer) == 2
    assert la.formant_smoother.buffer[0] == (500, 1500)
    assert la.formant_smoother.buffer[1] == (500.0, 1500.0)


def test_live_analyzer_reset():
    engine = DummyEngine()
    la = LiveAnalyzer(engine, PitchSmoother(), MedianSmoother(), LabelSmoother())

    la.pitch_smoother.current = 200
    la.formant_smoother.buffer.extend([(500, 1500)])
    la.label_smoother.current = "ɑ"

    la.reset()

    assert la.pitch_smoother.current is None
    assert len(la.formant_smoother.buffer) == 0
    assert la.label_smoother.current is None
