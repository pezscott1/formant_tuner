import numpy as np
from analysis.smoothing import MedianSmoother, PitchSmoother, LabelSmoother


def test_median_smoother_handles_none():
    sm = MedianSmoother()

    out = sm.update(None, None, confidence=1.0)
    # Stored as nan, smoothed result is None
    assert out == (None, None, None)


def test_median_smoother_stores_nan_for_none():
    sm = MedianSmoother()

    sm.update(None, None, confidence=1.0)
    assert np.isnan(sm.buf_f1[-1])
    assert np.isnan(sm.buf_f2[-1])
    assert np.isnan(sm.buf_f3[-1])


def test_median_smoother_all_nan_buffer():
    sm = MedianSmoother()

    sm.update(None, None, confidence=1.0)
    sm.update(None, None, confidence=1.0)
    sm.update(None, None, confidence=1.0)

    out = sm.update(None, None, confidence=1.0)
    assert out == (None, None, None)


def test_pitch_smoother_rejects_low_confidence():
    s = PitchSmoother(min_confidence=0.5)

    assert s.update(100, confidence=1.0) == 100.0
    assert s.update(200, confidence=0.1) == 100.0


def test_label_smoother_confidence_gate():
    sm = LabelSmoother(min_confidence=0.5)

    assert sm.update("a", confidence=0.1) is None
    assert sm.update("a", confidence=1.0) == "a"
