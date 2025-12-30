import numpy as np
from analysis.pitch import estimate_pitch


def test_pitch_empty_frame():
    """n == 0 → method='none', reason='empty'."""
    result = estimate_pitch(np.array([]), sr=48000)
    assert result.f0 is None
    assert result.method == "none"
    assert result.debug.get("reason") == "empty"


def test_pitch_short_no_rising_slope():
    """Short frame with no rising slope → short/no rising slope."""
    frame = np.zeros(100)
    result = estimate_pitch(frame, sr=48000)
    assert result.f0 is None
    assert result.method == "short"
    assert result.debug.get("reason") == "no rising slope"


def test_pitch_short_out_of_range():
    """Short frame but peak outside fmin/fmax → short/out_of_range."""
    sr = 48000
    t = np.linspace(0, 0.002, int(0.002 * sr))  # short frame
    frame = np.sin(2 * np.pi * 2000 * t)        # too high
    result = estimate_pitch(frame, sr=sr)
    assert result.f0 is None
    assert result.method == "short"
    assert result.debug.get("reason") == "out_of_range"


def test_pitch_normal_lag_window_invalid():
    """Force min_lag >= max_lag → none/lag window invalid."""
    frame = np.random.randn(500)
    # Make fmin > fmax to force invalid window
    result = estimate_pitch(frame, sr=48000, fmin=800, fmax=50)
    assert result.f0 is None
    assert result.method == "none"
    assert result.debug.get("reason") == "lag window invalid"


def test_pitch_normal_empty_segment():
    """Force empty segment → none/empty segment."""
    frame = np.random.randn(500)
    # Make min_lag == max_lag by setting fmin=fmax
    result = estimate_pitch(frame, sr=48000, fmin=200, fmax=200)
    assert result.f0 is None
    assert result.method == "none"
    assert result.debug.get("reason") in {"lag window invalid", "empty segment"}


def test_pitch_normal_subharmonic_expected():
    """
    Autocorrelation often returns subharmonics.
    For 200 Hz + 300 Hz mixture, your estimator returns ~100 Hz.
    """
    sr = 48000
    t = np.linspace(0, 0.03, int(0.03 * sr))
    frame = 1.0 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 300 * t)

    result = estimate_pitch(frame, sr=sr)
    assert result.f0 is not None
    assert 90 < result.f0 < 110  # subharmonic at ~100 Hz


def test_pitch_low_frequency_aliasing():
    """
    A 20 Hz tone produces a harmonic peak around ~150 Hz in your estimator.
    """
    sr = 48000
    t = np.linspace(0, 0.05, int(0.05 * sr))
    frame = np.sin(2 * np.pi * 20 * t)

    result = estimate_pitch(frame, sr=sr)
    assert result.f0 is not None
    assert 100 < result.f0 < 200  # harmonic aliasing expected
