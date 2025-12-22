# tests/test_engine.py
import numpy as np
from unittest.mock import patch
from analysis.engine import FormantAnalysisEngine


def make_engine():
    """Create a basic engine."""
    return FormantAnalysisEngine(voice_type="bass")


# ---------------------------------------------------------
# Basic pitch extraction
# ---------------------------------------------------------

@patch("analysis.engine.estimate_pitch", return_value=120)
def test_pitch_basic(_mock_pitch):
    eng = make_engine()
    frame = np.ones(1024)

    out = eng.process_frame(frame, 44100)

    assert out["f0"] == 120
    assert out["formants"] == (None, None, None) or isinstance(out["formants"], tuple)


# ---------------------------------------------------------
# Formant extraction
# ---------------------------------------------------------

@patch("analysis.engine.estimate_formants_lpc", return_value=(500, 1500, 2500))
@patch("analysis.engine.estimate_pitch", return_value=120)
def test_formant_extraction(_mock_pitch, _mock_lpc):
    eng = make_engine()
    frame = np.ones(1024)

    out = eng.process_frame(frame, 44100)

    assert out["formants"] == (500, 1500, 2500)


# ---------------------------------------------------------
# Vowel guessing
# ---------------------------------------------------------

@patch("analysis.engine.robust_guess", return_value=("a", 0.9, None))
@patch("analysis.engine.estimate_formants_lpc", return_value=(500, 1500, 2500))
@patch("analysis.engine.estimate_pitch", return_value=120)
def test_vowel_guessing(_mock_pitch, _mock_lpc, _mock_guess):
    eng = make_engine()
    frame = np.ones(1024)

    out = eng.process_frame(frame, 44100)

    assert out["vowel"] == "a"
    assert out["vowel_confidence"] == 0.9


# ---------------------------------------------------------
# Scoring integration
# ---------------------------------------------------------

@patch("analysis.engine.live_score_formants", return_value=80)
@patch("analysis.engine.resonance_tuning_score", return_value=60)
@patch("analysis.engine.robust_guess", return_value=("a", 0.9, None))
@patch("analysis.engine.estimate_formants_lpc", return_value=(500, 1500, 2500))
@patch("analysis.engine.estimate_pitch", return_value=120)
def test_scoring(_mock_pitch, _mock_lpc, _mock_guess, _mock_res_score, _mock_live_score):
    eng = make_engine()
    frame = np.ones(1024)

    out = eng.process_frame(frame, 44100)

    assert out["vowel_score"] == 80
    assert out["resonance_score"] == 60
    assert out["overall"] == 0.5 * 80 + 0.5 * 60


# ---------------------------------------------------------
# Segment storage
# ---------------------------------------------------------

@patch("analysis.engine.estimate_formants_lpc", return_value=(500, 1500, 2500))
@patch("analysis.engine.estimate_pitch", return_value=120)
def test_segment_stored(_mock_pitch, _mock_lpc):
    eng = make_engine()
    frame = np.ones(1024)

    out = eng.process_frame(frame, 44100)

    assert "segment" in out
    assert isinstance(out["segment"], np.ndarray)
    assert out["segment"].shape == frame.shape
