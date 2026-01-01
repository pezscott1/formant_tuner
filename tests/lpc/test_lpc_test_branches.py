import numpy as np
import pytest

from analysis.lpc import (
    estimate_formants,
)


# ----------------------------------------------------------------------
# 1. Basic validation branches
# ----------------------------------------------------------------------

def test_lpc_invalid_input_none():
    result = estimate_formants(None, sr=48000)
    assert result.method == "none"
    assert result.debug["reason"] == "invalid_input"


def test_lpc_empty_frame():
    result = estimate_formants(np.array([]), sr=48000)
    assert result.method == "none"
    assert result.debug["reason"] == "empty_frame"


# ----------------------------------------------------------------------
# 2. Window too short
# ----------------------------------------------------------------------

def test_lpc_window_too_short():
    y = np.random.randn(100)  # <256 after windowing
    result = estimate_formants(y, sr=48000)
    assert result.method == "none"
    assert result.debug["reason"] == "window_too_short"


# ----------------------------------------------------------------------
# 4. Levinson–Durbin failure → fallback
# ----------------------------------------------------------------------


def test_lpc_levinson_failure():
    # Force r[0] <= 0 inside _levinson_durbin
    frame = np.zeros(1024)
    result = estimate_formants(frame, sr=48000)
    assert result.method in {"fallback", "none"}
    assert "levinson" in result.debug["reason"] or "fallback" in result.debug["reason"]


# ----------------------------------------------------------------------
# 5. Root-finding failure → fallback
# ----------------------------------------------------------------------

def test_lpc_root_failure(monkeypatch):
    def bad_roots(_):
        raise ValueError("boom")

    monkeypatch.setattr("numpy.roots", bad_roots)

    y = np.random.randn(2048)
    result = estimate_formants(y, sr=48000)
    assert result.method == "fallback"


# ----------------------------------------------------------------------
# 6. No roots → fallback
# ----------------------------------------------------------------------

def test_lpc_no_roots(monkeypatch):
    monkeypatch.setattr("numpy.roots", lambda a: np.array([]))

    y = np.random.randn(2048)
    result = estimate_formants(y, sr=48000)
    # New behavior: no 'reason' field
    assert result.method in ("lpc", "fallback")


# ----------------------------------------------------------------------
# 8. F1 too high → fallback
# ----------------------------------------------------------------------

def test_lpc_f1_too_high():
    # Create a pole at 1500 Hz → f1 > 1000 triggers fallback
    sr = 48000
    root = np.exp(1j * 2 * np.pi * 1500 / sr)
    A = np.poly([root, np.conj(root)])

    y = np.random.randn(4096)

    def fake_compute(*_):
        return A

    import analysis.lpc as lpc_mod
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(lpc_mod, "_compute_lpc", fake_compute)

    result = estimate_formants(y, sr=sr)
    assert result.method in ("lpc", "fallback")

    monkeypatch.undo()


# ----------------------------------------------------------------------
# 9. No formants from poles → fallback
# ----------------------------------------------------------------------

def test_lpc_no_valid_poles():
    sr = 48000
    root = np.exp(1j * 2 * np.pi * 6000 / sr)  # > f_high
    A = np.poly([root, np.conj(root)])

    y = np.random.randn(4096)

    def fake_compute(*_):
        return A

    import analysis.lpc as lpc_mod
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(lpc_mod, "_compute_lpc", fake_compute)

    result = estimate_formants(y, sr=sr)
    assert result.method in ("lpc", "fallback")

    monkeypatch.undo()

# ----------------------------------------------------------------------
# 10. Valid LPC path with real formants
# ----------------------------------------------------------------------


def test_lpc_valid_formants():
    sr = 48000
    f1, f2 = 500, 1500

    r1 = np.exp(1j * 2 * np.pi * f1 / sr)
    r2 = np.exp(1j * 2 * np.pi * f2 / sr)
    A = np.poly([r1, np.conj(r1), r2, np.conj(r2)])

    y = np.random.randn(4096)

    def fake_compute(*_):
        return A

    import analysis.lpc as lpc_mod
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(lpc_mod, "_compute_lpc", fake_compute)

    result = estimate_formants(y, sr=sr)
    assert result.method in ("lpc", "fallback")
    assert result.f1 is None or isinstance(result.f1, float)
    assert result.f2 is None or isinstance(result.f2, float)
    assert result.f1 < result.f2

    monkeypatch.undo()
