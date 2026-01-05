# ============================================================
# tests/smoothing/test_smoothing_branches.py
# ============================================================
import pytest
from analysis.smoothing import (
    PitchSmoother, MedianSmoother, LabelSmoother, FormantStabilityTracker)


def test_pitch_smoother_branch_low_confidence_resets():
    ps = PitchSmoother(min_confidence=0.7)
    ps.update(100, confidence=1.0)
    out = ps.update(150, confidence=0.1)
    assert out is None
    assert ps.current is None


def test_pitch_smoother_branch_large_jump_accepted():
    ps = PitchSmoother(jump_limit=5)
    ps.current = 100
    out = ps.update(200)
    assert out == 200
    assert ps.current == 200


def test_pitch_smoother_branch_unwrap_objects():
    class Obj:
        def __init__(self, f0):
            self.f0 = f0

    ps = PitchSmoother()
    out = ps.update(Obj(123))
    assert out == 123


def test_pitch_smoother_branch_bad_input_keeps_current():
    ps = PitchSmoother()
    ps.current = 100
    out = ps.update("bad")
    assert out == 100


def test_pitch_smoother_branch_octave_snap_up():
    ps = PitchSmoother()
    ps.current = 100
    assert ps._octave_correct(180) == 200


def test_pitch_smoother_branch_octave_snap_down():
    ps = PitchSmoother()
    ps.current = 200
    assert ps._octave_correct(90) == 100


def test_pitch_smoother_branch_ema():
    ps = PitchSmoother(alpha=0.25)
    ps.current = 100
    out = ps.update(200)
    assert out == pytest.approx(125)


# MedianSmoother branches remain unchanged
def test_median_smoother_branch_confidence_gate():
    ms = MedianSmoother(min_confidence=0.8)
    assert ms.update(500, 1500, 2500, confidence=0.2) == (None, None, None)


def test_median_smoother_branch_nan_propagation():
    ms = MedianSmoother()
    ms.update(None, None, None, confidence=1.0)
    f1, f2, f3 = ms.update(500, 1500, 2500, confidence=1.0)
    assert f1 == 500


def test_label_smoother_branch_low_confidence():
    ls = LabelSmoother(min_confidence=0.8)
    assert ls.update("a", confidence=0.1) is None


def test_label_smoother_branch_hysteresis():
    ls = LabelSmoother(hold_frames=2)
    ls.update("a", confidence=1.0)
    assert ls.update("b", confidence=1.0) == "a"
    assert ls.update("b", confidence=1.0) == "b"


def test_stability_tracker_branch_insufficient_frames():
    st = FormantStabilityTracker(min_full_frames=3)
    st.update(500, 1500, None)
    st.update(None, 1500, 2500)
    stable, score = st.update(500, None, 2500)
    assert stable is False
    assert score == float("inf")
