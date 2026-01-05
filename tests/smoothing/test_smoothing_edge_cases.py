# ============================================================
# tests/smoothing/test_smoothing_edge_cases.py
# ============================================================
import numpy as np
import pytest
from analysis.smoothing import PitchSmoother, MedianSmoother, FormantStabilityTracker


def test_pitch_smoother_edge_low_confidence_resets():
    ps = PitchSmoother(min_confidence=0.6)
    ps.update(100, confidence=1.0)
    out = ps.update(200, confidence=0.1)
    assert out is None
    assert ps.current is None


def test_pitch_smoother_edge_large_jump():
    ps = PitchSmoother(jump_limit=1)
    ps.current = 100
    out = ps.update(500)
    assert out == 500


def test_pitch_smoother_edge_nan_input():
    ps = PitchSmoother()
    ps.update(120)
    out = ps.update(np.nan)
    assert isinstance(out, float)


def test_pitch_smoother_edge_ema_behavior():
    ps = PitchSmoother(alpha=0.5)
    ps.current = 100
    out = ps.update(200)
    assert out == pytest.approx(150)


def test_median_smoother_edge_all_nan():
    ms = MedianSmoother(window=3)
    ms.update(None, None, None, confidence=1.0)
    ms.update(None, None, None, confidence=1.0)
    f1, f2, f3 = ms.update(None, None, None, confidence=1.0)
    assert f1 is None and f2 is None and f3 is None


def test_stability_tracker_edge_ridge_collapse():
    st = FormantStabilityTracker(window_size=5, min_full_frames=3)
    st.update(2601, 2602, 2599)
    st.update(2605, 2603, 2601)
    stable, score = st.update(2598, 2604, 2602)
    assert stable is False
    assert score == float("inf")


def test_stability_tracker_edge_trim():
    st = FormantStabilityTracker(window_size=6, min_full_frames=3, trim_pct=20)
    st.update(500, 1500, 2500)
    st.update(1000, 2000, 3000)
    stable, score = st.update(500, 1500, 2500)
    assert np.isfinite(score)
