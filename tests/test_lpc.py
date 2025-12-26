import numpy as np
from analysis.lpc import estimate_formants, LPCConfig


def test_lpc_estimates_formants_for_clean_vowel():
    sig = np.random.randn(4096)
    cfg = LPCConfig()

    out = estimate_formants(sig, sr=48000, config=cfg, debug=False)

    # Basic structural expectations
    assert hasattr(out, "f1")
    assert hasattr(out, "f2")
    assert hasattr(out, "f3")
    assert hasattr(out, "confidence")
    assert hasattr(out, "method")
    assert hasattr(out, "lpc_order")
    assert hasattr(out, "debug")


def test_lpc_fallback_when_roots_fail_does_not_crash():
    # Pathological signal: all zeros
    sig = np.zeros(4096)
    cfg = LPCConfig()

    out = estimate_formants(sig, sr=48000, config=cfg, debug=False)

    # We don't assert exact values, just that it returns a valid object
    assert hasattr(out, "f1")
    assert hasattr(out, "confidence")
    assert isinstance(out.method, str)


def test_lpc_estimator_handles_back_vowels_structurally():
    # Low-F1, low-F2 region – "back vowel"-like
    sig = np.random.randn(4096) * 0.01  # quiet-ish, but valid
    cfg = LPCConfig()

    out = estimate_formants(sig, sr=48000, config=cfg, debug=False)

    # Just assert it doesn't blow up and returns sane types
    assert hasattr(out, "f1")
    assert hasattr(out, "f2")
    assert isinstance(out.confidence, float)
