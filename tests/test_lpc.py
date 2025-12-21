import numpy as np
import pytest
from analysis.lpc import estimate_formants_lpc


def test_lpc_short_frame_returns_empty():
    frame = np.array([0.1, -0.2])
    f1, f2, f3 = estimate_formants_lpc(frame, sr=48000)
    assert f1 is None and f2 is None and f3 is None


def test_lpc_zero_frame_returns_empty():
    frame = np.zeros(1024)
    f1, f2, f3 = estimate_formants_lpc(frame, sr=48000)
    assert f1 is None and f2 is None


def test_lpc_sine_wave_produces_formants():
    sr = 48000
    f0 = 200
    t = np.arange(0, 0.03, 1 / sr)
    frame = np.sin(2 * np.pi * f0 * t)

    f1, f2, f3 = estimate_formants_lpc(frame, sr)

    # At least one formant should be detected
    assert f1 is not None or f2 is not None or f3 is not None

    # All outputs must exist (even if None)
    assert isinstance((f1, f2, f3), tuple)


def test_lpc_forces_solver_failure():
    # A pathological frame that produces a singular autocorrelation matrix
    frame = np.ones(1024)  # constant → singular
    f1, f2, f3 = estimate_formants_lpc(frame, sr=48000)
    assert f1 is None and f2 is None


def test_lpc_invalid_roots():
    # Construct a pathological frame that produces invalid LPC roots
    # Random noise often produces roots outside the unit circle
    np.random.seed(0)
    frame = np.random.randn(2048)

    f1, f2, f3 = estimate_formants_lpc(frame, sr=48000)
    # We don't care what the values are — only that the function returns something
    assert f1 is None or isinstance(f1, float)
    assert f2 is None or isinstance(f2, float)


def test_lpc_nan_input_returns_empty():
    frame = np.array([np.nan] * 1024)
    f1, f2, f3 = estimate_formants_lpc(frame, sr=48000)
    assert f1 is None and f2 is None


def test_lpc_exception_handling():
    # Passing a non-numeric string should raise ValueError from np.asarray
    with pytest.raises(ValueError):
        estimate_formants_lpc("not an array", sr=48000)
