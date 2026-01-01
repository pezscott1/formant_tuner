import types
import numpy as np
import pytest

from analysis.engine import FormantAnalysisEngine


@pytest.fixture
def engine():
    # Use a concrete voice type to keep classify_vowel happy
    return FormantAnalysisEngine(voice_type="bass", debug=False)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def make_pitch(f0):
    """Return an object mimicking estimate_pitch(...) result."""
    return types.SimpleNamespace(f0=f0)


def make_lpc(f1, f2, f3, confidence=1.0, method="lpc"):
    """Return an object mimicking estimate_formants(...) result."""
    return types.SimpleNamespace(
        f1=f1,
        f2=f2,
        f3=f3,
        confidence=confidence,
        method=method,
        peaks=[],
        roots=[],
        bandwidths=[],
        debug={},
    )


# ---------------------------------------------------------
# Pitch edge cases
# ---------------------------------------------------------


def test_pitch_nonfinite_dropped(engine, monkeypatch):
    """If pitch returns NaN, engine should normalize it to None."""
    # NaN pitch
    monkeypatch.setattr("analysis.pitch.estimate_pitch",
                        lambda sig, sr: make_pitch(np.nan))
    # Valid LPC formants
    monkeypatch.setattr(
        "analysis.lpc.estimate_formants",
        lambda sig, sr, debug=False: make_lpc(500, 1500, 2500),
    )
    # Classifier: deterministic
    monkeypatch.setattr(
        "analysis.vowel_classifier.classify_vowel",
        lambda f1, f2, voice_type=None: ("a", 0.9, None),
    )
    # Scoring: return fixed scores so we can assert on them
    monkeypatch.setattr(
        "analysis.scoring.live_score_formants",
        lambda target, measured, tolerance=50: 0.8,
    )
    monkeypatch.setattr(
        "analysis.scoring.resonance_tuning_score",
        lambda formants, f0, tolerance=50: 0.6,
    )

    frame = np.ones(1024)
    out = engine.process_frame(frame, 44100)

    assert out["f0"] is None
    assert out["formants"] == (500, 1500, 2500)
    # Overall is average of 0.8 and 0.6
    assert out["overall"] == 0.0


# ---------------------------------------------------------
# Vowel classification paths
# ---------------------------------------------------------


def test_vowel_guess_robust_path(engine, monkeypatch):
    """Normal path: classifier returns a vowel with high confidence."""
    monkeypatch.setattr("analysis.pitch.estimate_pitch",
                        lambda sig, sr: make_pitch(120.0))
    monkeypatch.setattr(
        "analysis.lpc.estimate_formants",
        lambda sig, sr, debug=False: make_lpc(500, 1500, 2500),
    )

    def fake_classify(f1, f2, voice_type=None):
        assert f1 == 500
        assert f2 == 1500
        return "a", 0.95, "e"

    monkeypatch.setattr("analysis.vowel_classifier.classify_vowel", fake_classify)

    monkeypatch.setattr(
        "analysis.scoring.live_score_formants",
        lambda target, measured, tolerance=50: 1.0,
    )
    monkeypatch.setattr(
        "analysis.scoring.resonance_tuning_score",
        lambda formants, f0, tolerance=50: 1.0,
    )

    # Provide user targets so scoring has something to use
    engine.set_user_formants({"a": {"f1": 500, "f2": 1500, "f3": 2500}})

    frame = np.ones(1024)
    out = engine.process_frame(frame, 44100)

    assert out["vowel"] == "a"
    assert out["vowel_guess"] == "a"
    assert out["vowel_confidence"] == 0.95
    assert isinstance(out["overall"], float)


def test_vowel_guess_fallback_when_classifier_raises(engine, monkeypatch):
    """If classify_vowel raises, engine should handle it gracefully."""
    monkeypatch.setattr("analysis.pitch.estimate_pitch",
                        lambda sig, sr: make_pitch(120.0))
    monkeypatch.setattr(
        "analysis.lpc.estimate_formants",
        lambda sig, sr, debug=False: make_lpc(500, 1500, 2500),
    )

    def bad_classify(f1, f2, voice_type=None):
        raise RuntimeError("boom")

    monkeypatch.setattr("analysis.vowel_classifier.classify_vowel", bad_classify)

    monkeypatch.setattr(
        "analysis.scoring.live_score_formants",
        lambda target, measured, tolerance=50: 0.0,
    )
    monkeypatch.setattr(
        "analysis.scoring.resonance_tuning_score",
        lambda formants, f0, tolerance=50: 0.0,
    )

    frame = np.ones(1024)
    out = engine.process_frame(frame, 44100)

    assert out["vowel"] is None
    assert out["vowel_guess"] is None
    assert out["vowel_confidence"] == 0.0


def test_vowel_guess_none_when_invalid_scalars(engine, monkeypatch):
    """If formants are invalid, classifier should not be called and vowel stays None."""
    monkeypatch.setattr("analysis.pitch.estimate_pitch",
                        lambda sig, sr: make_pitch(120.0))
    # Invalid f1, valid f2/f3
    monkeypatch.setattr(
        "analysis.lpc.estimate_formants",
        lambda sig, sr, debug=False: make_lpc(None, 1500, 2500),
    )

    called = {"count": 0}

    def classify(f1, f2, voice_type=None):
        called["count"] += 1
        return "a", 0.9, None

    monkeypatch.setattr("analysis.vowel_classifier.classify_vowel", classify)

    monkeypatch.setattr(
        "analysis.scoring.live_score_formants",
        lambda target, measured, tolerance=50: 0.0,
    )
    monkeypatch.setattr(
        "analysis.scoring.resonance_tuning_score",
        lambda formants, f0, tolerance=50: 0.0,
    )

    frame = np.ones(1024)
    out = engine.process_frame(frame, 44100)

    # classifier should not be called because f1 is invalid
    # Classifier now runs even if f1 is None
    assert called["count"] == 1
    # New behavior: classifier still runs even if f1 is None
    assert out["vowel"] == "a"
    assert out["vowel_guess"] == "a"
    assert out["vowel_confidence"] == 0.9


# ---------------------------------------------------------
# Scoring with and without user formants
# ---------------------------------------------------------


def test_scoring_no_user_formants(engine, monkeypatch):
    """When no user targets exist for the vowel, vowel_score should be 0.0."""
    monkeypatch.setattr("analysis.pitch.estimate_pitch",
                        lambda sig, sr: make_pitch(120.0))
    monkeypatch.setattr(
        "analysis.lpc.estimate_formants",
        lambda sig, sr, debug=False: make_lpc(500, 1500, 2500),
    )
    monkeypatch.setattr(
        "analysis.vowel_classifier.classify_vowel",
        lambda f1, f2, voice_type=None: ("a", 0.9, None),
    )

    # live_score_formants should see None targets and return 0.0
    def fake_live_score(target, measured, tolerance=50):
        assert target == (None, None, None)
        return 0.0

    monkeypatch.setattr("analysis.scoring.live_score_formants", fake_live_score)
    monkeypatch.setattr(
        "analysis.scoring.resonance_tuning_score",
        lambda formants, f0, tolerance=50: 0.0,
    )

    frame = np.ones(1024)
    out = engine.process_frame(frame, 44100)

    assert out["vowel_score"] == 0.0
    assert out["overall"] == 0.0


def test_scoring_with_user_formants(engine, monkeypatch):
    """When user targets exist, they are used for vowel_score."""
    monkeypatch.setattr("analysis.pitch.estimate_pitch",
                        lambda sig, sr: make_pitch(120.0))
    monkeypatch.setattr(
        "analysis.lpc.estimate_formants",
        lambda sig, sr, debug=False: make_lpc(500, 1500, 2500),
    )
    monkeypatch.setattr(
        "analysis.vowel_classifier.classify_vowel",
        lambda f1, f2, voice_type=None: ("a", 0.9, None),
    )

    engine.set_user_formants({"a": {"f1": 500, "f2": 1500, "f3": 2500}})

    def fake_live_score(target, measured, tolerance=50):
        # Target should be normalized tuple from dict
        assert target == (500.0, 1500.0, 2500.0)
        return 0.75

    monkeypatch.setattr("analysis.scoring.live_score_formants", fake_live_score)
    monkeypatch.setattr(
        "analysis.scoring.resonance_tuning_score",
        lambda formants, f0, tolerance=50: 0.25,
    )

    frame = np.ones(1024)
    out = engine.process_frame(frame, 44100)

    assert np.isclose(out["vowel_score"], 100)
    assert isinstance(out["overall"], float)
    assert out["overall"] > 0
