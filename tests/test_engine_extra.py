import numpy as np
from unittest.mock import patch
from analysis.engine import FormantAnalysisEngine


def make_engine():
    return FormantAnalysisEngine(voice_type="bass")


# ---------------------------------------------------------
# Missing pitch
# ---------------------------------------------------------

@patch("analysis.engine.estimate_pitch", return_value=None)
@patch("analysis.engine.estimate_formants_lpc", return_value=(500, 1500, 2500))
def test_missing_pitch(_mock_lpc, _mock_pitch):
    eng = make_engine()
    frame = np.ones(1024)

    out = eng.process_frame(frame, 44100)

    assert out["f0"] is None
    assert "vowel" in out
    assert "vowel_confidence" in out


# ---------------------------------------------------------
# Missing formants
# ---------------------------------------------------------

@patch("analysis.engine.estimate_pitch", return_value=120)
@patch("analysis.engine.estimate_formants_lpc", return_value=(None, None, None))
def test_missing_formants(_mock_lpc, _mock_pitch):
    eng = make_engine()
    frame = np.ones(1024)

    out = eng.process_frame(frame, 44100)

    assert out["formants"] == (None, None, None)
    assert out["vowel"] is None or isinstance(out["vowel"], str)


# ---------------------------------------------------------
# Invalid formants (f1 > f2)
# ---------------------------------------------------------

@patch("analysis.engine.estimate_pitch", return_value=120)
@patch("analysis.engine.estimate_formants_lpc", return_value=(2000, 500, 2500))
def test_invalid_formants_swapped(_mock_lpc, _mock_pitch):
    eng = make_engine()
    frame = np.ones(1024)

    out = eng.process_frame(frame, 44100)

    # Engine should still return a tuple
    assert isinstance(out["formants"], tuple)

    # Vowel guess should still produce *something*
    assert isinstance(out["vowel"], str) or out["vowel"] is None


# ---------------------------------------------------------
# Invalid pitch (NaN)
# ---------------------------------------------------------

@patch("analysis.engine.estimate_pitch", return_value=np.nan)
@patch("analysis.engine.estimate_formants_lpc", return_value=(500, 1500, 2500))
def test_invalid_pitch_nan(_mock_lpc, _mock_pitch):
    eng = make_engine()
    frame = np.ones(1024)

    out = eng.process_frame(frame, 44100)

    # Engine passes NaN through unchanged
    assert np.isnan(out["f0"])

    # Vowel guess still runs and returns something
    assert "vowel" in out
    assert "vowel_confidence" in out


# ---------------------------------------------------------
# Vowel guess returns None (fallback)
# ---------------------------------------------------------

@patch("analysis.engine.robust_guess", return_value=(None, 0.0, None))
@patch("analysis.engine.estimate_formants_lpc", return_value=(500, 1500, 2500))
@patch("analysis.engine.estimate_pitch", return_value=120)
def test_vowel_guess_fallback(_mock_pitch, _mock_lpc, _mock_guess):
    eng = make_engine()
    frame = np.ones(1024)

    out = eng.process_frame(frame, 44100)

    assert out["vowel"] is None
    assert out["vowel_confidence"] == 0.0


# ---------------------------------------------------------
# Scoring fallback when vowel not recognized
# ---------------------------------------------------------

@patch("analysis.engine.live_score_formants", return_value=0)
@patch("analysis.engine.resonance_tuning_score", return_value=0)
@patch("analysis.engine.robust_guess", return_value=("zzz", 0.5, None))
@patch("analysis.engine.estimate_formants_lpc", return_value=(500, 1500, 2500))
@patch("analysis.engine.estimate_pitch", return_value=120)
def test_scoring_fallback_unknown_vowel(_mock_pitch, _mock_lpc, _mock_guess, _mock_res, _mock_live):
    eng = make_engine()
    frame = np.ones(1024)

    out = eng.process_frame(frame, 44100)

    assert out["vowel"] == "zzz"
    assert out["vowel_score"] == 0
    assert out["resonance_score"] == 0
    assert out["overall"] == 0


# ---------------------------------------------------------
# No formants AND no pitch
# ---------------------------------------------------------

@patch("analysis.engine.estimate_pitch", return_value=None)
@patch("analysis.engine.estimate_formants_lpc", return_value=(None, None, None))
def test_no_formants_no_pitch(_mock_lpc, _mock_pitch):
    eng = make_engine()
    frame = np.ones(1024)

    out = eng.process_frame(frame, 44100)

    assert out["formants"] == (None, None, None)
    assert out["f0"] is None
    assert out["vowel"] is None
