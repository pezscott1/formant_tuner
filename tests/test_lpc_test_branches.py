import numpy as np
import pytest

from analysis.lpc import (
    estimate_formants,
    LPCConfig,
    _compute_lpc,
    _levinson_durbin,
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
# 3. Insufficient samples for chosen LPC order
# ----------------------------------------------------------------------

def test_lpc_insufficient_samples_for_order():
    cfg = LPCConfig(min_order=14, max_order=14)
    y = np.random.randn(20)
    result = estimate_formants(y, sr=48000, config=cfg)
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
    assert "root_fail" in result.debug["reason"]


# ----------------------------------------------------------------------
# 6. No roots → fallback
# ----------------------------------------------------------------------

def test_lpc_no_roots(monkeypatch):
    monkeypatch.setattr("numpy.roots", lambda A: np.array([]))

    y = np.random.randn(2048)
    result = estimate_formants(y, sr=48000)
    assert result.method == "fallback"
    assert "no_roots" in result.debug["reason"]


# ----------------------------------------------------------------------
# 7. No valid poles after filtering → fallback
# ----------------------------------------------------------------------

def test_lpc_no_valid_poles():
    # Construct LPC coeffs that produce poles outside f_low/f_high
    roots = np.array([np.exp(1j * 2 * np.pi * 6000 / 48000)])  # > f_high
    A = np.poly(roots)

    cfg = LPCConfig()
    y = np.random.randn(4096)

    # Monkeypatch _compute_lpc to return our custom A
    def fake_compute(*_):
        return A

    import analysis.lpc as lpc_mod
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(lpc_mod, "_compute_lpc", fake_compute)

    result = estimate_formants(y, sr=48000, config=cfg)
    assert result.method == "fallback"
    assert "no_valid_poles" in result.debug["reason"]

    monkeypatch.undo()


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
    assert result.method == "fallback"
    assert "f1_too_high_lpc" in result.debug["reason"]

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
    assert result.method == "fallback"
    assert "no_valid_poles" in result.debug["reason"]

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
    assert result.method == "lpc"
    assert result.f1 is not None
    assert result.f2 is not None
    assert result.f1 < result.f2

    monkeypatch.undo()
