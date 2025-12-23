# tests/test_smoothing.py
import numpy as np
from analysis.smoothing import MedianSmoother, PitchSmoother, LabelSmoother


# ---------------------------------------------------------
# PitchSmoother (exponential smoothing)
# ---------------------------------------------------------

def test_pitch_smoother_basic():
    s = PitchSmoother(alpha=0.5, jump_limit=None)

    # First value initializes the smoother
    assert s.update(100) == 100

    # Exponential smoothing:
    # new = 0.5 * 200 + 0.5 * 100 = 150
    assert s.update(200) == 150

    # new = 0.5 * 300 + 0.5 * 150 = 225
    assert s.update(300) == 225


def test_pitch_smoother_ignores_none_and_nan():
    s = PitchSmoother(alpha=0.5)

    s.update(100)
    assert s.update(None) == 100
    assert s.update(np.nan) == 100

    # After ignoring invalid values, smoothing resumes normally
    assert s.update(120) == 110  # 0.5 * 120 + 0.5 * 100


# ---------------------------------------------------------
# MedianSmoother (F1/F2 only)
# ---------------------------------------------------------

def test_median_smoother_basic():
    s = MedianSmoother(window=3)

    # First value
    assert s.update(100, 200) == (100, 200)

    # Median of [100,110] = 105
    assert s.update(110, 210) == (105, 205)

    # Median of [100,110,90] = 100
    assert s.update(90, 190) == (100, 200)


def test_median_smoother_handles_none_and_nan():
    s = MedianSmoother(window=5)

    s.update(100, 200)
    s.update(None, None)
    s.update(np.nan, np.nan)

    # Should ignore None/NaN and keep previous valid median
    out = s.update(120, 220)
    assert out == (110, 210)


def test_median_smoother_window_rollover():
    s = MedianSmoother(window=3)

    s.update(10, 20)
    s.update(100, 200)
    s.update(50, 150)

    # Window now contains [(10,20), (100,200), (50,150)]
    # Median = (50,150)
    assert s.update(5, 15) == (50, 150)


# ---------------------------------------------------------
# LabelSmoother (hysteresis)
# ---------------------------------------------------------

def test_label_smoother_initialization():
    s = LabelSmoother(hold_frames=2)

    assert s.update("a") == "a"
    assert s.current == "a"
    assert s.last == "a"


def test_label_smoother_dwell_requirement():
    s = LabelSmoother(hold_frames=2)

    s.update("a")          # current = a
    assert s.update("e") == "a"  # not enough dwell
    assert s.update("e") == "e"  # now stable


def test_label_smoother_ignores_none():
    s = LabelSmoother(hold_frames=2)

    s.update("a")
    assert s.update(None) == "a"
    assert s.current == "a"


def test_label_smoother_window_rollover():
    s = LabelSmoother(hold_frames=2)

    s.update("a")
    s.update("b")          # buffer sees b once
    assert s.update("b") == "b"  # now stable

    s.update("c")          # new label
    assert s.update("c") == "c"  # stable after dwell


def test_pitch_smoother_hps_fallback(monkeypatch):
    ps = PitchSmoother(jump_limit=50, sr=44100)

    # Seed with stable pitch
    ps.current = 200.0

    # Fill audio buffer so HPS can run
    ps.audio_buffer.extend([0.1] * ps.hps_window)

    # Fake HPS result
    monkeypatch.setattr("analysis.smoothing.hps_pitch", lambda sig, sr: 210.0)

    # Incoming pitch is a huge jump → fallback should override it
    out = ps.update(500.0)
    assert abs(out - 202.5) < 1e-6


def test_pitch_smoother_no_hps_when_buffer_small(monkeypatch):
    ps = PitchSmoother(jump_limit=50, sr=44100)
    ps.current = 200.0

    # Buffer too small → fallback not used
    ps.audio_buffer.extend([0.1] * 100)

    out = ps.update(500.0)
    assert out == 200.0  # clamped to previous


def test_pitch_smoother_doubling_suppression():
    ps = PitchSmoother()
    ps.current = 200.0

    out = ps.update(400.0)  # doubling
    assert out == 200.0


def test_formant_smoother_outlier_rejection():
    ms = MedianSmoother(window=5, outlier_thresh=100)

    # Add stable values
    for _ in range(4):
        ms.update(500, 1500)

    # Add a huge outlier
    f1_s, f2_s = ms.update(2000, 5000)

    # Should ignore outlier and return median of stable values
    assert f1_s == 500
    assert f2_s == 1500


def test_formant_smoother_nan_handling():
    ms = MedianSmoother(window=3)

    ms.update(500, 1500)
    ms.update(None, None)
    f1_s, f2_s = ms.update(520, 1480)

    assert abs(f1_s - 510.0) < 1e-6
    assert abs(f2_s - 1490.0) < 1e-6


def test_label_smoother_confidence_threshold():
    ls = LabelSmoother(min_confidence=0.5)
    ls.update("ɑ", 1.0)

    out = ls.update("i", 0.2)  # too low confidence
    assert out == "ɑ"


def test_label_smoother_hysteresis():
    ls = LabelSmoother(hold_frames=3)
    ls.update("ɑ", 1.0)

    # Not enough persistence
    ls.update("i", 1.0)
    ls.update("i", 1.0)
    out = ls.update("i", 1.0)

    assert out == "i"
