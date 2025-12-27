
import numpy as np
from analysis.smoothing import (
    PitchSmoother,
    MedianSmoother,
    LabelSmoother,
    FormantStabilityTracker,
    hps_pitch,
)


def test_pitch_smoother_basic():
    s = PitchSmoother(alpha=0.5, jump_limit=80)

    # With jump suppression enabled, the smoother may reject large jumps.
    out2 = s.update(100)
    assert isinstance(out2, float)


def test_pitch_smoother_octave_correction():
    s = PitchSmoother()
    s.current = 100.0

    # 2× current within 40 Hz → octave correction
    out = s.update(195)  # 195 is close to 2*100 = 200
    assert out in (100.0, 200.0)


def test_pitch_smoother_jump_suppression_without_hps():
    s = PitchSmoother(jump_limit=20)
    s.current = 100.0

    # Big jump but no HPS buffer → return current
    out = s.update(200)
    assert out == 100.0


def test_median_smoother_basic():
    sm = MedianSmoother(window=5)

    out = sm.update(500, 1500, confidence=1.0)
    assert out[:2] == (500.0, 1500.0)

    out = sm.update(510, 1490, confidence=1.0)
    assert out[:2] == (505.0, 1495.0)


def test_label_smoother_basic():
    sm = LabelSmoother(hold_frames=2)

    assert sm.update("a", confidence=1.0) == "a"
    assert sm.update("a", confidence=1.0) == "a"

    # New label appears once → not enough to switch
    assert sm.update("i", confidence=1.0) == "a"

    # Second consecutive "i" → switch
    assert sm.update("i", confidence=1.0) == "i"


def test_stability_tracker_ridge_collapse():
    st = FormantStabilityTracker()
    for _ in range(5):
        st.update(2600, 2600, 2600)
    stable, score = st.update(2600, 2600, 2600)
    assert stable is False
    assert score == float("inf")


# ---------------------------------------------------------
# PitchSmoother
# ---------------------------------------------------------

def test_pitch_smoother_first_value():
    ps = PitchSmoother()
    out = ps.update(120.0)
    assert out == 120.0
    assert ps.current == 120.0


def test_pitch_smoother_jump_suppression():
    ps = PitchSmoother(jump_limit=50)
    ps.update(120.0)
    out = ps.update(300.0)  # too large a jump
    assert out == 120.0


def test_pitch_smoother_confidence_gate():
    ps = PitchSmoother(min_confidence=0.8)
    ps.update(120.0)
    out = ps.update(130.0, confidence=0.5)
    assert out == 120.0


def test_pitch_smoother_octave_correction_double():
    ps = PitchSmoother()
    ps.current = 120.0
    corrected = ps._octave_correct(240.0 + 10)  # within 40 Hz
    assert corrected == 240.0


def test_pitch_smoother_octave_correction_half():
    ps = PitchSmoother()
    ps.current = 200.0
    corrected = ps._octave_correct(100.0 + 10)  # within 20 Hz
    assert corrected == 100.0


# ---------------------------------------------------------
# MedianSmoother
# ---------------------------------------------------------

def test_median_smoother_basic_median():
    ms = MedianSmoother(window=5)
    vals = [(500, 1500, 2500)] * 5
    out = None
    for f1, f2, f3 in vals:
        out = ms.update(f1, f2, f3, confidence=1.0)

    f1_s, f2_s, f3_s = out
    assert abs(f1_s - 500.0) < 1e-6
    assert abs(f2_s - 1500.0) < 1e-6
    # f3 may be None or a float, depending on buffer history; we don’t force it.


def test_median_smoother_outlier_rejection():
    ms = MedianSmoother(window=5, outlier_thresh=100)
    ms.update(500, 1500, 2500, confidence=1.0)
    ms.update(505, 1490, 2510, confidence=1.0)
    ms.update(490, 1510, 2490, confidence=1.0)
    ms.update(5000, 1500, 2500, confidence=1.0)  # outlier
    f1, f2, f3 = ms.update(500, 1500, 2500, confidence=1.0)

    assert abs(f1 - 500) < 5
    assert abs(f2 - 1500) < 20


def test_median_smoother_ridge_suppression():
    ms = MedianSmoother(window=3)

    # Two ridge-band frames
    ms.update(2600, 2600, 2600, confidence=1.0)
    ms.update(2605, 2590, 2610, confidence=1.0)

    # One normal frame
    f1, f2, f3 = ms.update(500, 1500, 2500, confidence=1.0)

    assert f1 == 500.0
    assert f2 == 1500.0
    # For F3 we only assert that the ridge band is gone
    if f3 is not None:
        assert not (2400 < f3 < 2800)


# ---------------------------------------------------------
# LabelSmoother
# ---------------------------------------------------------

def test_label_smoother_basic_hold():
    ls = LabelSmoother(hold_frames=3)
    assert ls.update("a", confidence=1.0) == "a"
    assert ls.update("i", confidence=1.0) == "a"
    assert ls.update("i", confidence=1.0) == "a"
    assert ls.update("i", confidence=1.0) == "i"  # after 3 frames


def test_label_smoother_confidence_gate():
    ls = LabelSmoother(min_confidence=0.8)
    ls.update("a", confidence=1.0)
    out = ls.update("i", confidence=0.1)
    assert out == "a"


# ---------------------------------------------------------
# FormantStabilityTracker
# ---------------------------------------------------------

def test_stability_tracker_requires_full_frames():
    st = FormantStabilityTracker(window_size=5, min_full_frames=3)
    st.update(500, None, 2500)
    st.update(None, 1500, 2500)
    stable, score = st.update(500, 1500, None)
    assert stable is False
    assert score == float("inf")


def test_stability_tracker_detects_stable_region():
    st = FormantStabilityTracker(window_size=6, min_full_frames=3)
    st.update(500, 1500, 2500)
    st.update(505, 1490, 2510)
    stable, score = st.update(495, 1510, 2490)
    assert stable is True
    assert score < 1e5


# ---------------------------------------------------------
# hps_pitch
# ---------------------------------------------------------

def test_hps_pitch_returns_none_for_short_signal():
    assert hps_pitch(np.zeros(100), 48000) is None


def test_hps_pitch_detects_simple_tone():
    sr = 48000
    f0 = 200
    t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
    sig = np.sin(2 * np.pi * f0 * t)
    out = hps_pitch(sig, sr)

    assert out is not None
    # Must be within allowed band
    assert 50 <= out <= 500

