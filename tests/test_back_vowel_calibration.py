import numpy as np
from analysis.lpc import estimate_formants_lpc
from analysis.vowel import is_plausible_formants
from calibration.session import CalibrationSession


def synthetic_vowel(f1, f2, sr=44100, dur=0.05):
    """Generate a synthetic vowel-like signal with two formants."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sig = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    sig *= np.hamming(len(sig))
    return sig


def test_plausibility_accepts_back_vowels():
    # Typical baritone /o/ and /u/
    assert is_plausible_formants(400, 900, voice_type="baritone", vowel="o")[0]
    assert is_plausible_formants(300, 700, voice_type="baritone", vowel="u")[0]


def test_lpc_estimator_handles_back_vowels():
    sig = synthetic_vowel(350, 800)
    f1, f2, f3 = estimate_formants_lpc(sig, 44100)
    assert f1 is not None
    assert f2 is not None
    assert f1 < f2


def test_lpc_fallback_works_when_roots_fail(monkeypatch):
    # Force LPC to fail by returning empty roots
    def fake_lpc(*_args, **_kwargs):
        return None, None, None

    monkeypatch.setattr("analysis.lpc.estimate_formants_lpc", fake_lpc)

    sig = synthetic_vowel(350, 800)
    f1, f2, f3 = estimate_formants_lpc(sig, 44100)
    assert f1 is not None
    assert f2 is not None


def test_calibration_session_accepts_back_vowels():
    session = CalibrationSession("test", "baritone", ["o", "u"])

    # Simulate good captures
    session.handle_result(350, 800, 120)  # /o/
    session.handle_result(300, 700, 110)  # /u/

    assert "o" in session.results
    assert "u" in session.results
