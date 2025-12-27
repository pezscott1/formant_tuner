from unittest.mock import patch, MagicMock
import numpy as np
from analysis.engine import FormantAnalysisEngine


@patch("analysis.engine.estimate_pitch")
@patch("analysis.engine.estimate_formants")
def test_engine_basic_processing(mock_lpc, mock_pitch):
    mock_pitch.return_value = MagicMock(f0=200.0)
    mock_lpc.return_value = MagicMock(
        f1=500.0,
        f2=1500.0,
        f3=2500.0,
        confidence=0.9,
        method="lpc",
        lpc_order=12,
        peaks=[1, 2, 3],
        roots=[0.1, 0.2],
        bandwidths=[100, 120],
        debug={"ok": True},
    )

    eng = FormantAnalysisEngine()
    frame = np.random.randn(2048)

    out = eng.process_frame(frame, sr=48000)

    assert out["f0"] == 200.0
    assert out["formants"] == (500.0, 1500.0, 2500.0)
    assert isinstance(out["segment"], np.ndarray)


@patch("analysis.engine.estimate_pitch")
@patch("analysis.engine.estimate_formants")
def test_engine_vowel_guessing(mock_lpc, mock_pitch):
    mock_pitch.return_value = MagicMock(f0=150.0)
    mock_lpc.return_value = MagicMock(
        f1=300.0,
        f2=2500.0,
        f3=3000.0,
        confidence=0.8,
        method="lpc",
        lpc_order=12,
        peaks=[],
        roots=[],
        bandwidths=[],
        debug={},
    )

    eng = FormantAnalysisEngine(voice_type="bass")
    frame = np.random.randn(2048)

    out = eng.process_frame(frame, sr=48000)

    # We don’t assert the exact vowel, just that we get a guess + confidence
    assert "vowel_guess" in out
    assert "vowel_confidence" in out


@patch("analysis.engine.estimate_pitch")
@patch("analysis.engine.estimate_formants")
def test_engine_scoring_with_user_formants(mock_lpc, mock_pitch):
    mock_pitch.return_value = MagicMock(f0=150.0)
    mock_lpc.return_value = MagicMock(
        f1=500.0,
        f2=1500.0,
        f3=2500.0,
        confidence=0.9,
        method="lpc",
        lpc_order=12,
        peaks=[],
        roots=[],
        bandwidths=[],
        debug={},
    )

    eng = FormantAnalysisEngine()
    eng.set_user_formants({"a": (500, 1500, 2500)})

    frame = np.random.randn(2048)
    out = eng.process_frame(frame, sr=48000)

    assert "vowel_score" in out
    assert "overall" in out
