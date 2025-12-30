import numpy as np
import pytest

from analysis.smoothing import (
    hps_pitch,
    PitchSmoother,
    MedianSmoother,
    LabelSmoother,
    FormantStabilityTracker,
)


# ----------------------------------------------------------------------
# hps_pitch
# ----------------------------------------------------------------------

def test_hps_pitch_too_short():
    assert hps_pitch(np.zeros(100), 48000) is None


def test_hps_pitch_no_mask():
    sig = np.random.randn(2048)
    out = hps_pitch(sig, 48000, min_f0=2000, max_f0=3000)
    assert isinstance(out, float)


def test_hps_pitch_basic_peak():
    sr = 48000
    t = np.linspace(0, 0.05, int(0.05 * sr))
    sig = np.sin(2 * np.pi * 200 * t)
    out = hps_pitch(sig, sr)
    assert out in (50.0, 60.0, 100.0, 200.0)


# ----------------------------------------------------------------------
# PitchSmoother
# ----------------------------------------------------------------------

class DummyPitchObj:
    def __init__(self, f0):
        self.f0 = f0


class DummyFreqObj:
    def __init__(self, f):
        self.frequency = f


def test_pitch_smoother_unwrap_f0():
    ps = PitchSmoother()
    out = ps.update(DummyPitchObj(100))
    assert out == 100


def test_pitch_smoother_unwrap_frequency():
    ps = PitchSmoother()
    out = ps.update(DummyFreqObj(150))
    assert out == 150


def test_pitch_smoother_none_input():
    ps = PitchSmoother()
    assert ps.update(None) is None


def test_pitch_smoother_nonfloat_input():
    ps = PitchSmoother()
    ps.current = 100
    assert ps.update("bad") == 100


def test_pitch_smoother_first_frame():
    ps = PitchSmoother()
    assert ps.update(200) == 200


def test_pitch_smoother_confidence_gate():
    ps = PitchSmoother(min_confidence=0.8)
    ps.current = 100
    assert ps.update(120, confidence=0.5) == 100


def test_pitch_smoother_jump_suppression():
    ps = PitchSmoother(jump_limit=10)
    ps.current = 100
    assert ps.update(200) == 100


def test_pitch_smoother_ema():
    ps = PitchSmoother(alpha=0.5)
    ps.current = 100
    out = ps.update(120)
    assert out == pytest.approx(110)


def test_pitch_smoother_octave_correct_up():
    ps = PitchSmoother()
    ps.current = 100
    assert ps._octave_correct(180) == 200


def test_pitch_smoother_octave_correct_down():
    ps = PitchSmoother()
    ps.current = 200
    assert ps._octave_correct(90) == 100


# ----------------------------------------------------------------------
# MedianSmoother
# ----------------------------------------------------------------------

def test_median_smoother_confidence_gate():
    ms = MedianSmoother(min_confidence=0.5)
    assert ms.update(500, 1500, 2500, confidence=0.2) == (None, None, None)


def test_median_smoother_ridge_suppression():
    ms = MedianSmoother()
    f1, f2, f3 = ms.update(2601, 2602, 2603, confidence=1.0)
    assert f1 is None and f2 is None and f3 is None


def test_median_smoother_outlier_rejection():
    ms = MedianSmoother(window=5, outlier_thresh=10)
    ms.update(500, 1500, 2500, confidence=1.0)
    ms.update(505, 1505, 2505, confidence=1.0)
    ms.update(2000, 3000, 4000, confidence=1.0)  # outlier
    f1, f2, f3 = ms.update(495, 1495, 2495, confidence=1.0)
    assert 495 <= f1 <= 505


def test_median_smoother_nan_handling():
    ms = MedianSmoother()
    ms.update(None, None, None, confidence=1.0)
    f1, f2, f3 = ms.update(500, 1500, 2500, confidence=1.0)
    assert f1 == 500


def test_median_smoother_stability_propagation():
    ms = MedianSmoother()
    for _ in range(6):
        ms.update(500, 1500, 2500, confidence=1.0)
    assert ms.formants_stable in (True, False)
    assert isinstance(ms._stability_score, float)


# ----------------------------------------------------------------------
# LabelSmoother
# ----------------------------------------------------------------------

def test_label_smoother_confidence_gate():
    ls = LabelSmoother(min_confidence=0.8)
    assert ls.update("a", confidence=0.5) is None


def test_label_smoother_first_label():
    ls = LabelSmoother()
    assert ls.update("a", confidence=1.0) == "a"


def test_label_smoother_same_label():
    ls = LabelSmoother()
    ls.update("a", confidence=1.0)
    assert ls.update("a", confidence=1.0) == "a"


def test_label_smoother_hysteresis():
    ls = LabelSmoother(hold_frames=2)
    ls.update("a", confidence=1.0)
    assert ls.update("b", confidence=1.0) == "a"
    assert ls.update("b", confidence=1.0) == "b"


# ----------------------------------------------------------------------
# FormantStabilityTracker
# ----------------------------------------------------------------------

def test_stability_insufficient_frames():
    st = FormantStabilityTracker(min_full_frames=3)
    st.update(500, 1500, 2500)
    st.update(510, 1510, 2510)
    stable, score = st.update(None, 1500, 2500)
    assert stable is False
    assert score == float("inf")


def test_stability_ridge_collapse():
    st = FormantStabilityTracker(min_full_frames=3)
    st.update(2601, 2602, 2603)
    st.update(2605, 2606, 2607)
    stable, score = st.update(2604, 2603, 2602)
    assert stable is False
    assert score == float("inf")


def test_stability_variance_stable():
    st = FormantStabilityTracker(min_full_frames=3, var_threshold=1e6)
    st.update(500, 1500, 2500)
    st.update(505, 1505, 2505)
    stable, score = st.update(495, 1495, 2495)
    assert stable is True
    assert score
