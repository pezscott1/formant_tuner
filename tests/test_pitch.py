import numpy as np
from analysis.pitch import estimate_pitch


def test_empty_frame_returns_none():
    assert estimate_pitch([], 44100) is None
    assert estimate_pitch(np.array([]), 44100) is None


def test_constant_frame_returns_none():
    frame = np.ones(1024)
    assert estimate_pitch(frame, 44100) is None


def test_very_short_frame_returns_high_pitch():
    frame = np.array([0.1, -0.2, 0.3])
    f0 = estimate_pitch(frame, 44100)
    assert f0 is not None
    assert f0 > 10000  # extremely short frames → extremely high pitch


def test_pure_sine_wave_pitch_estimation():
    sr = 48000
    f0 = 200
    t = np.arange(0, 0.03, 1 / sr)
    frame = np.sin(2 * np.pi * f0 * t)

    est = estimate_pitch(frame, sr)
    assert est is not None
    assert abs(est - f0) < 5   # allow small tolerance


def test_noisy_sine_wave_pitch_estimation():
    sr = 48000
    f0 = 150
    t = np.arange(0, 0.03, 1 / sr)
    clean = np.sin(2 * np.pi * f0 * t)
    noise = 0.1 * np.random.randn(len(t))
    frame = clean + noise

    est = estimate_pitch(frame, sr)
    assert est is not None
    assert est > 0


def test_autocorrelation_peak_at_zero_returns_none():
    # Construct a pathological frame where autocorrelation peak is at lag 0
    frame = np.zeros(1024)
    frame[0] = 1.0  # spike at start → autocorr peak at 0
    assert estimate_pitch(frame, 44100) is None


def test_frame_with_nan_returns_none():
    frame = np.array([0.1, np.nan, 0.2])
    # After mean subtraction, NaN propagates → autocorr becomes NaN → no positive slope
    assert estimate_pitch(frame, 44100) is None
