import numpy as np
from analysis.true_envelope import estimate_formants_te, TEFormantResult


# -----------------------------
# 1. Empty frame
# -----------------------------

def test_te_empty_frame():
    res = estimate_formants_te(np.array([]), 44100)
    assert res.debug["reason"] == "empty_frame"
    assert res.f1 is None and res.f2 is None


# -----------------------------
# 2. Noise collapse (length=2048, std≈1)
# -----------------------------

def test_te_noise_collapse():
    x = np.random.randn(2048)
    res = estimate_formants_te(x, 44100)
    assert res.debug["reason"] == "std_len_noise_collapse"
    assert res.f1 is None


# -----------------------------
# 3. Long silence collapse (all zeros, >256)
# -----------------------------

def test_te_long_silence():
    x = np.zeros(300)
    res = estimate_formants_te(x, 44100)
    assert res.debug["reason"] == "no_peaks_silence"
    assert res.f1 is None


# -----------------------------
# 4. No vowel band (very low sample rate)
# -----------------------------

def test_te_no_band():
    x = np.ones(512)
    res = estimate_formants_te(x, sr=50)
    assert res.debug["reason"] == "no_band"
    assert res.f1 is None


# -----------------------------
# 5. Short zero frame → numeric 0.0 formants
# -----------------------------

def test_te_short_zero_frame():
    x = np.zeros(200)
    res = estimate_formants_te(x, 44100)
    assert res.debug["reason"] == "no_peaks"
    assert res.f1 == 0.0
    assert res.f2 == 0.0


# -----------------------------
# 6. Real signal but no peaks → fallback maxima
# -----------------------------

def test_te_fallback_no_peaks(monkeypatch):
    monkeypatch.setattr(
        "analysis.true_envelope.find_peaks",
        lambda *a, **k: (np.array([]), {})
    )
    x = np.ones(300)
    res = estimate_formants_te(x, 44100)
    assert res.debug["reason"] == "fallback_no_peaks"

# -----------------------------
# 7. Normal peak picking path
# -----------------------------


def test_te_normal_peaks():
    # A synthetic vowel-like signal with clear peaks
    sr = 44100
    t = np.linspace(0, 0.03, int(0.03 * sr), endpoint=False)
    x = (
        np.sin(2 * np.pi * 500 * t) +
        0.5 * np.sin(2 * np.pi * 1500 * t) +
        0.3 * np.sin(2 * np.pi * 2500 * t)
    )
    res = estimate_formants_te(x, sr)
    assert isinstance(res, TEFormantResult)
    assert res.f1 is not None
    assert res.confidence > 0.5
    assert len(res.peaks) >= 1


# -----------------------------
# 8. LPC fallback path (monkeypatch _extract_formants to fail)
# -----------------------------

def test_te_lpc_fallback(monkeypatch):
    # Force LPC selector to fail
    def bad_extract(freqs_sorted):
        return None, None, None

    monkeypatch.setattr("analysis.lpc._extract_formants", bad_extract)

    sr = 44100
    t = np.linspace(0, 0.02, int(0.02 * sr), endpoint=False)
    x = np.sin(2 * np.pi * 600 * t)

    res = estimate_formants_te(x, sr)

    # f1 must be filled by TE's own fallback
    assert res.f1 is not None
    # There ARE peaks in this case, so confidence comes from normal path
    assert res.confidence > 0.5


# -----------------------------
# 9. Debug fields always present
# -----------------------------

def test_te_debug_fields():
    x = np.random.randn(3000)
    res = estimate_formants_te(x, 44100)
    dbg = res.debug
    assert "peak_freqs" in dbg
    assert "env_max" in dbg
    assert "env_min" in dbg
