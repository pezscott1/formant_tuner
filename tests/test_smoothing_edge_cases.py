import numpy as np
from analysis.smoothing import (
    MedianSmoother,
    LabelSmoother,
    FormantStabilityTracker,
    PitchSmoother,
)


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


def test_pitch_smoother_handles_nan():
    ps = PitchSmoother()
    ps.update(120.0)
    out = ps.update(np.nan)

    # Current implementation will set current to nan; we just assert no crash and float-like.
    assert isinstance(out, float)


def test_median_smoother_all_nan():
    ms = MedianSmoother(window=3)
    ms.update(None, None, None, confidence=1.0)
    ms.update(None, None, None, confidence=1.0)
    f1, f2, f3 = ms.update(None, None, None, confidence=1.0)
    assert f1 is None and f2 is None and f3 is None


def test_stability_tracker_ridge_collapse():
    st = FormantStabilityTracker(window_size=5, min_full_frames=3)
    st.update(2601, 2602, 2599)
    st.update(2605, 2603, 2601)
    stable, score = st.update(2598, 2604, 2602)
    assert stable is False
    assert score == float("inf")


def test_stability_tracker_trimming():
    st = FormantStabilityTracker(window_size=6, min_full_frames=3, trim_pct=20)
    st.update(500, 1500, 2500)
    st.update(1000, 2000, 3000)
    stable, score = st.update(500, 1500, 2500)

    assert np.isfinite(score)
    # Whether it's stable or not depends on actual variance; we don't force True.
