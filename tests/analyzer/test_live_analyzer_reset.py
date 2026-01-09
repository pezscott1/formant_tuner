# tests/live_analyzer/test_reset.py
from tuner.live_analyzer import LiveAnalyzer
from analysis.smoothing import PitchSmoother, MedianSmoother, LabelSmoother


class DummyEngine:
    def process_frame(self, segment, sr):
        return {
            "f0": 150,
            "formants": (500, 1500, 2500),
            "confidence": 1.0,
            "vowel": "a",
        }


def test_live_analyzer_reset_clears_smoothing_state():
    engine = DummyEngine()
    pa = PitchSmoother()
    fa = MedianSmoother()
    la = LabelSmoother()

    live = LiveAnalyzer(engine, pa, fa, la)

    # Feed a few frames
    for _ in range(5):
        live.process_raw(engine.process_frame(None, 48000))

    # Ensure buffers are populated
    assert len(fa.buf_f1) > 0
    assert len(fa.buf_f2) > 0
    assert len(fa.buf_f3) > 0
    assert fa.formants_stable in (True, False)

    live.reset()

    # All buffers cleared
    assert len(fa.buf_f1) == 0
    assert len(fa.buf_f2) == 0
    assert len(fa.buf_f3) == 0

    # Stability reset
    assert fa.formants_stable is False
    assert fa._stability_score == float("inf")

    # Pitch smoother reset
    assert pa.current is None

    # Label smoother reset
    assert la.current is None
    assert la.last is None
    assert la.counter == 0
