# tests/test_hybrid_selector.py

from types import SimpleNamespace
import pytest
from analysis.hybrid_formants import (
    choose_formants_hybrid,
)


def _ns(f1, f2, f3=None, confidence=1.0, method="lpc"):
    return SimpleNamespace(f1=f1, f2=f2, f3=f3, confidence=confidence, method=method)


def test_front_i_prefers_lpc_when_te_collapses():
    # LPC: good /i/ for your baritone (approx)
    lpc_res = _ns(f1=260.0, f2=2200.0, confidence=0.9, method="lpc")

    # TE: harmonic confusion (F2 too low for /i/)
    te_res = _ns(f1=230.0, f2=950.0, method="te")

    out = choose_formants_hybrid(lpc_res, te_res, vowel_hint="i")

    # Method now encodes the hybrid case, but values must match LPC
    assert out.f1 == pytest.approx(lpc_res.f1)
    assert out.f2 == pytest.approx(lpc_res.f2)
    assert out.method in ("lpc", "hybrid_front")


def test_back_u_prefers_te_when_lpc_f2_too_high():
    # LPC: mis-tracks, f2 way too high for /u/
    lpc_res = _ns(f1=320.0, f2=1900.0, confidence=0.6, method="lpc")

    # TE: plausible /u/ cluster
    te_res = _ns(f1=300.0, f2=500.0, method="te")

    out = choose_formants_hybrid(lpc_res, te_res, vowel_hint="u")

    assert out.method == "te"
    assert out.f1 == te_res.f1
    assert out.f2 == te_res.f2
    assert out.confidence == 0.7  # TE baseline
    assert "back_high_f2" in out.debug["lpc_vetoes"]
    assert out.debug["selection_case"] in {"only_te_ok", "both_ok"}


def test_both_bad_falls_back_to_lpc_low_confidence():
    lpc_res = _ns(f1=None, f2=None, confidence=0.5, method="lpc")
    te_res = _ns(f1=None, f2=None, method="te")

    out = choose_formants_hybrid(lpc_res, te_res, vowel_hint=None)

    assert out.method == "lpc"  # fallback path
    assert out.confidence <= 0.2
    assert out.debug["selection_case"] == "both_bad"


def test_unknown_vowel_defaults_to_lpc_when_both_plausible():
    lpc_res = _ns(f1=400.0, f2=1800.0, confidence=0.7, method="lpc")
    te_res = _ns(f1=380.0, f2=1700.0, method="te")

    out = choose_formants_hybrid(lpc_res, te_res, vowel_hint=None)

    assert out.method == "lpc"
    assert out.debug["primary"] == "lpc"
    assert out.debug["selection_case"] == "both_ok"


def test_back_vowel_prefers_te_when_both_plausible():
    lpc_res = _ns(f1=300.0, f2=1100.0, confidence=0.9, method="lpc")
    te_res = _ns(f1=290.0, f2=900.0, method="te")

    out = choose_formants_hybrid(lpc_res, te_res, vowel_hint="u")

    assert out.debug["primary"] == "te"
    assert out.method == "te"
    assert out.debug["selection_case"] == "both_ok"
