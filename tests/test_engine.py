import numpy as np
from unittest.mock import patch, MagicMock
from analysis.engine import FormantAnalysisEngine


def test_engine_handles_empty_frame():
    eng = FormantAnalysisEngine()

    out = eng.process_frame(np.array([]), sr=48000)

    assert out["f0"] is None
    assert out["formants"] == (None, None, None)
    assert out["vowel"] is None
    assert out["vowel_confidence"] == 0.0
    assert out["overall"] == 0.0
    assert out["method"] == "none"
    assert eng.get_latest_raw() == out


@patch("analysis.engine.estimate_pitch", return_value=200.0)
@patch("analysis.engine.estimate_formants")
def test_engine_basic_processing(mock_lpc, mock_pitch):
    # Fake LPC result object
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
    assert out["confidence"] == 0.9
    assert out["method"] == "lpc"
    assert out["lpc_order"] == 12
    assert out["peaks"] == [1, 2, 3]
    assert out["roots"] == [0.1, 0.2]
    assert out["bandwidths"] == [100, 120]
    assert out["lpc_debug"] == {"ok": True}


@patch("analysis.engine.estimate_pitch", return_value=150.0)
@patch("analysis.engine.estimate_formants")
def test_engine_vowel_guessing(mock_lpc, mock_pitch):
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

    # Should guess a vowel (robust_guess or fallback)
    assert out["vowel"] is not None
    assert out["vowel_confidence"] >= 0.0


@patch("analysis.engine.estimate_pitch", return_value=150.0)
@patch("analysis.engine.estimate_formants")
def test_engine_scoring_with_user_formants(mock_lpc, mock_pitch):
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

    # With matching formants, scores should be positive
    assert out["vowel_score"] >= 0.0
    assert out["resonance_score"] >= 0.0
    assert out["overall"] >= 0.0
