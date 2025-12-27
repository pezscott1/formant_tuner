import numpy as np
import pytest

from analysis.engine import FormantAnalysisEngine


# ----------------------------------------------------------------------
# Dummy components to isolate engine behavior
# ----------------------------------------------------------------------

class DummyPitchResult:
    def __init__(self, f0):
        self.f0 = f0
        self.confidence = 0.0
        self.method = "dummy"
        self.debug = {}


class DummyPitch:
    def __init__(self, f0):
        self.f0 = f0

    def __call__(self, signal, sr):
        return DummyPitchResult(self.f0)


class DummyFormantResult:
    def __init__(self, f1, f2, f3, conf=1.0, method="lpc"):
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.confidence = conf
        self.method = method
        self.lpc_order = 10
        self.peaks = [100, 200]
        self.roots = [1, 2]
        self.bandwidths = [50, 60]
        self.debug = {"x": 1}


class DummyLPC:
    def __init__(self, f1, f2, f3, conf=1.0):
        self.result = DummyFormantResult(f1, f2, f3, conf)

    def __call__(self, signal, sr, config=None, debug=False):
        return self.result


class DummyScoring:
    def __init__(self, vowel_score=0.5, resonance_score=0.7):
        self.vowel_score = vowel_score
        self.resonance_score = resonance_score

    def live_score_formants(self, target, measured, tolerance):
        return self.vowel_score

    def resonance_tuning_score(self, measured, f0, tolerance):
        return self.resonance_score


# ----------------------------------------------------------------------
# Monkeypatch helpers
# ----------------------------------------------------------------------

@pytest.fixture
def patch_engine(monkeypatch):
    """Patch estimate_pitch, estimate_formants, robust_guess, guess_vowel, scoring."""
    def _apply(
        monkeypatch,
        pitch_f0=None,
        lpc_f1=None,
        lpc_f2=None,
        lpc_f3=None,
        lpc_conf=1.0,
        robust=None,
        guess=None,
        vowel_score=0.5,
        resonance_score=0.7,
    ):
        # Pitch
        monkeypatch.setattr(
            "analysis.engine.estimate_pitch",
            lambda signal, sr: DummyPitchResult(pitch_f0),
        )

        # LPC
        monkeypatch.setattr(
            "analysis.engine.estimate_formants",
            lambda signal, sr, config=None, debug=False: DummyFormantResult(
                lpc_f1, lpc_f2, lpc_f3, conf=lpc_conf
            ),
        )

        # Vowel guessers
        if robust is not None:
            monkeypatch.setattr("analysis.engine.robust_guess", robust)
        if guess is not None:
            monkeypatch.setattr("analysis.engine.guess_vowel", guess)

        # Scoring
        monkeypatch.setattr(
            "analysis.engine.live_score_formants",
            lambda target, measured, tolerance: vowel_score,
        )
        monkeypatch.setattr(
            "analysis.engine.resonance_tuning_score",
            lambda measured, f0, tolerance: resonance_score,
        )

    return _apply


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_empty_signal():
    eng = FormantAnalysisEngine()
    out = eng.process_frame(np.array([]), 48000)
    assert out["f0"] is None
    assert out["formants"] == (None, None, None)
    assert out["vowel"] is None
    assert out["method"] == "none"
    assert eng.get_latest_raw() == out


def test_pitch_nonfinite_dropped(patch_engine, monkeypatch):
    patch_engine(
        monkeypatch,
        pitch_f0=float("nan"),
        lpc_f1=500,
        lpc_f2=1500,
        lpc_f3=2500,
    )
    eng = FormantAnalysisEngine()
    out = eng.process_frame(np.ones(100), 48000)
    assert out["f0"] is None  # dropped


def test_vowel_guess_robust_path(patch_engine, monkeypatch):
    patch_engine(
        monkeypatch,
        pitch_f0=100,
        lpc_f1=500,
        lpc_f2=1500,
        lpc_f3=2500,
        lpc_conf=1.0,
        robust=lambda formants, voice_type: ("a", 0.9, "e"),
    )
    eng = FormantAnalysisEngine()
    out = eng.process_frame(np.ones(100), 48000)
    assert out["vowel"] == "a"
    assert out["vowel_confidence"] == 0.9


def test_vowel_guess_fallback_when_robust_raises(patch_engine, monkeypatch):
    def bad_robust(formants, voice_type):
        raise RuntimeError("boom")

    patch_engine(
        monkeypatch,
        pitch_f0=100,
        lpc_f1=500,
        lpc_f2=1500,
        lpc_f3=2500,
        lpc_conf=1.0,
        robust=bad_robust,
        guess=lambda f1, f2, vt: "i",
    )
    eng = FormantAnalysisEngine()
    out = eng.process_frame(np.ones(100), 48000)
    assert out["vowel"] == "i"
    assert out["vowel_confidence"] == 0.0


def test_vowel_guess_none_when_invalid_scalars(patch_engine, monkeypatch):
    patch_engine(
        monkeypatch,
        pitch_f0=100,
        lpc_f1=None,
        lpc_f2=1500,
        lpc_f3=2500,
        lpc_conf=1.0,
    )
    eng = FormantAnalysisEngine()
    out = eng.process_frame(np.ones(100), 48000)
    assert out["vowel"] is None
    assert out["vowel_confidence"] == 0.0


def test_scoring_no_user_formants(patch_engine, monkeypatch):
    patch_engine(
        monkeypatch,
        pitch_f0=100,
        lpc_f1=500,
        lpc_f2=1500,
        lpc_f3=2500,
    )
    eng = FormantAnalysisEngine()
    out = eng.process_frame(np.ones(100), 48000)
    assert out["vowel_score"] == 0.0
    assert out["overall"] == 0.0


def test_scoring_with_user_formants(patch_engine, monkeypatch):
    patch_engine(
        monkeypatch,
        pitch_f0=100,
        lpc_f1=500,
        lpc_f2=1500,
        lpc_f3=2500,
        vowel_score=0.8,
        resonance_score=0.6,
    )
    eng = FormantAnalysisEngine()
    eng.set_user_formants({"a": (500, 1500, 2500)})
    out = eng.process_frame(np.ones(100), 48000)
    assert out["vowel_score"] == 0.8
    assert out["resonance_score"] == 0.6
    assert out["overall"] == pytest.approx(0.7)


def test_resonance_score_fallback_on_exception(patch_engine, monkeypatch):
    def bad_resonance(measured, f0, tolerance):
        raise RuntimeError("boom")

    patch_engine(
        monkeypatch,
        pitch_f0=100,
        lpc_f1=500,
        lpc_f2=1500,
        lpc_f3=2500,
        vowel_score=0.8,
        resonance_score=0.6,
    )
    monkeypatch.setattr(
        "analysis.engine.resonance_tuning_score",
        bad_resonance,
    )

    eng = FormantAnalysisEngine()
    eng.set_user_formants({"a": (500, 1500, 2500)})
    out = eng.process_frame(np.ones(100), 48000)
    assert out["resonance_score"] == 0.0


def test_latest_raw_stored(patch_engine, monkeypatch):
    patch_engine(
        monkeypatch,
        pitch_f0=100,
        lpc_f1=500,
        lpc_f2=1500,
        lpc_f3=2500,
    )
    eng = FormantAnalysisEngine()
    out = eng.process_frame(np.ones(100), 48000)
    assert eng.get_latest_raw() == out
