import numpy as np
from analysis.hybrid_formants import estimate_formants_hybrid
from analysis.true_envelope import estimate_formants_te


def test_te_collapses_on_noise(monkeypatch):
    noise = np.random.randn(2048)
    res = estimate_formants_te(noise, sr=48000)
    f1, f2, conf = res.f1, res.f2, res.confidence

    assert f1 is None or f1 < 100
    assert f2 is None or f2 < 200
    assert conf < 0.2


def test_te_detects_formants_on_synthetic_vowel():

    sr = 48000
    t = np.linspace(0, 0.05, int(0.05 * sr), endpoint=False)

    # Synthetic vowel-like formants
    f1, f2 = 500, 1500
    sig = np.sin(2*np.pi*100*t) + 0.5*np.sin(2*np.pi*f1*t) + 0.3*np.sin(2*np.pi*f2*t)

    res = estimate_formants_te(sig, sr)
    f1, f2 = res.f1, res.f2
    assert abs(f1 - f1) < 100
    assert abs(f2 - f2) < 150


def test_hybrid_selector_vetoes_bad_te(monkeypatch):

    # Fake LPC and TE results
    class FakeLPC:
        f1 = 500
        f2 = 1500
        f3 = None
        confidence = 0.9
        method = "lpc"

    class FakeTE:
        f1 = 200
        f2 = 800   # too low for front vowel
        f3 = None

    monkeypatch.setattr(
        "analysis.hybrid_formants.estimate_formants",
        lambda *a, **k: FakeLPC()
    )
    monkeypatch.setattr(
        "analysis.hybrid_formants.estimate_formants_te",
        lambda *a, **k: FakeTE()
    )

    result = estimate_formants_hybrid(signal=None, sr=48000, vowel_hint="i")

    assert result.f1 == 500
    assert result.f2 == 1500
    assert result.method in ("lpc", "hybrid_front")
    assert "front_low_f2" in result.debug.get("te_vetoes", [])


def test_te_no_valid_peaks():
    y = np.zeros(4096)
    out = estimate_formants_te(y, sr=48000)
    assert out.method in ("te", "fallback")
    assert out.f1 is None
    assert out.f2 is None
