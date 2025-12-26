import numpy as np
from analysis.pitch import estimate_pitch


def test_pure_sine_wave_pitch_estimation():
    sr = 48000
    f0 = 200
    t = np.arange(0, 0.03, 1 / sr)
    frame = np.sin(2 * np.pi * f0 * t)

    res = estimate_pitch(frame, sr)
    assert res is not None
    assert res.f0 is not None
    assert abs(res.f0 - f0) < 5   # small tolerance


def test_noisy_sine_wave_pitch_estimation():
    sr = 48000
    f0 = 150
    t = np.arange(0, 0.03, 1 / sr)
    clean = np.sin(2 * np.pi * f0 * t)
    noise = 0.1 * np.random.randn(len(t))
    frame = clean + noise

    res = estimate_pitch(frame, sr)
    assert res is not None
    assert res.f0 is not None
    assert res.f0 > 0


def test_autocorrelation_peak_at_zero_returns_none():
    # Pathological frame: autocorr peak at lag 0 → no valid pitch
    frame = np.zeros(1024)
    frame[0] = 1.0

    res = estimate_pitch(frame, 44100)
    assert res.f0 is None

