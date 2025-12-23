# tests/test_smoothing_edge_cases.py
from analysis.smoothing import MedianSmoother, LabelSmoother, PitchSmoother
import math
import numpy as np


def test_median_smoother_handles_none():
    sm = MedianSmoother()
    sm.update(None, None)
    import math
    f1, f2 = sm.buffer[-1]
    assert math.isnan(f1)
    assert math.isnan(f2)


def test_median_smoother_buffer_limit():
    sm = MedianSmoother(window=3)
    for i in range(10):
        sm.update(i, i+1)
    assert len(sm.buffer) == 3


def test_label_smoother_initialization():
    sm = LabelSmoother()
    out = sm.update("i", 0.9)
    assert out == "i"


def test_label_smoother_confidence_threshold():
    sm = LabelSmoother()
    sm.update("i", 0.9)
    out = sm.update("ɛ", 0.1)
    assert out == "i"  # low confidence → stick to previous


def test_pitch_smoother_audio_buffer():
    sm = PitchSmoother()
    sm.push_audio([1, 2, 3])
    assert len(sm.audio_buffer) > 0

# -----------------------------
# MedianSmoother
# -----------------------------


def test_median_smoother_stores_nan_for_none():
    sm = MedianSmoother()
    sm.update(None, None)
    f1, f2 = sm.buffer[-1]
    assert math.isnan(f1)
    assert math.isnan(f2)


def test_median_smoother_window_limit():
    sm = MedianSmoother(window=3)
    for i in range(10):
        sm.update(i, i+1)
    assert len(sm.buffer) == 3
    assert sm.buffer[0] == (7, 8)
    assert sm.buffer[-1] == (9, 10)


def test_median_smoother_outlier_rejection():
    sm = MedianSmoother(window=5, outlier_thresh=100)

    sm.update(500, 1500)
    sm.update(510, 1510)
    sm.update(520, 1520)

    # This should be rejected
    f1_s, f2_s = sm.update(9999, 9999)

    assert f1_s == 510
    assert f2_s == 1510


def test_median_smoother_median_computation():
    sm = MedianSmoother(window=5)
    sm.update(100, 1000)
    sm.update(200, 1100)
    sm.update(300, 1200)
    f1_s, f2_s = sm.update(400, 1300)

    assert f1_s == 250
    assert f2_s == 1150

# -----------------------------
# LabelSmoother
# -----------------------------


def test_label_smoother_low_confidence_sticks():
    sm = LabelSmoother()
    sm.update("i", 0.9)
    out = sm.update("ɛ", 0.1)
    assert out == "i"


def test_label_smoother_switches_on_high_confidence():
    sm = LabelSmoother()
    sm.update("i", 0.9)
    out = sm.update("ɛ", 0.95)
    assert out == "i"

# -----------------------------
# PitchSmoother
# -----------------------------


def test_pitch_smoother_audio_buffer_grows():
    sm = PitchSmoother()
    sm.push_audio(np.array([1, 2, 3]))
    assert len(sm.audio_buffer) > 0


def test_pitch_smoother_handles_multiple_pushes():
    sm = PitchSmoother()
    sm.push_audio(np.array([1, 2, 3]))
    sm.push_audio(np.array([4, 5]))
    assert len(sm.audio_buffer) == 5
