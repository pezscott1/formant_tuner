# tests/test_live_analyzer_smoothing_reset.py
from tuner.live_analyzer import LiveAnalyzer
from analysis.engine import FormantAnalysisEngine
from analysis.smoothing import PitchSmoother, MedianSmoother, LabelSmoother


def test_plausibility_resets_formant_smoother():
    engine = FormantAnalysisEngine()
    la = LiveAnalyzer(engine, PitchSmoother(), MedianSmoother(), LabelSmoother())

    # First plausible frame
    la.process_raw({"f0": 100, "formants": (500, 1500, 200), "vowel_guess": "i"})

    # Now an implausible frame
    la.process_raw({"f0": 100, "formants": (99999, 99999, 200), "vowel_guess": "i"})

    assert len(la.formant_smoother.buffer) == 2
    assert la.formant_smoother.buffer[0] == (500.0, 1500.0)
    assert la.formant_smoother.buffer[1] == (99999.0, 99999.0)
