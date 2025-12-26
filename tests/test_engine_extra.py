import numpy as np
from unittest.mock import patch, MagicMock
from analysis.engine import FormantAnalysisEngine


# ---------------------------------------------------------
# Confidence propagation
# ---------------------------------------------------------
@patch("analysis.engine.estimate_pitch", return_value=180.0)
@patch("analysis.engine.estimate_formants")
def test_engine_propagates_formant_confidence(mock_lpc, mock_pitch):
    mock_lpc.return_value = MagicMock(
        f1=600.0,
        f2=1400.0,
        f3=2600.0,
        confidence=0.42,
        method="lpc",
        lpc_order=12,
        peaks=[1],
        roots=[0.1],
        bandwidths=[120],
        debug={"ok": True},
    )

    eng = FormantAnalysisEngine()
    frame = np.random.randn(2048)

    out = eng.process_frame(frame, sr=48000)

    assert out["confidence"] == 0.42
    assert out["method"] == "lpc"
    assert out["lpc_order"] == 12
    assert out["peaks"] == [1]
    assert out["roots"] == [0.1]
    assert out["bandwidths"] == [120]
    assert out["lpc_debug"] == {"ok": True}


# ---------------------------------------------------------
# Extreme formant values (engine should not crash)
# ---------------------------------------------------------
@patch("analysis.engine.estimate_pitch", return_value=120.0)
@patch("analysis.engine.estimate_formants")
def test_engine_handles_extreme_formant_values(mock_lpc, mock_pitch):
    mock_lpc.return_value = MagicMock(
        f1=99999.0,
        f2=-500.0,
        f3=1e6,
        confidence=0.2,
        method="lpc",
        lpc_order=14,
        peaks=[],
        roots=[],
        bandwidths=[],
        debug={},
    )

    eng = FormantAnalysisEngine()
    frame = np.random.randn(4096)

    out = eng.process_frame(frame, sr=48000)

    assert out["formants"] == (99999.0, -500.0, 1e6)
    assert out["confidence"] == 0.2
    assert out["method"] == "lpc"


# ---------------------------------------------------------
# Fallback behavior when LPC confidence is too low
# ---------------------------------------------------------
@patch("analysis.engine.estimate_pitch", return_value=150.0)
@patch("analysis.engine.estimate_formants")
def test_engine_low_confidence_disables_vowel_guess(mock_lpc, mock_pitch):
    mock_lpc.return_value = MagicMock(
        f1=500.0,
        f2=1500.0,
        f3=2500.0,
        confidence=0.05,  # too low for vowel guessing
        method="lpc",
        lpc_order=10,
        peaks=[],
        roots=[],
        bandwidths=[],
        debug={},
    )

    eng = FormantAnalysisEngine()
    frame = np.random.randn(2048)

    out = eng.process_frame(frame, sr=48000)

    assert out["vowel"] is None
    assert out["vowel_confidence"] == 0.0
    assert out["vowel_score"] == 0.0
    assert out["overall"] == 0.0


# ---------------------------------------------------------
# Engine should always return a complete structured dict
# ---------------------------------------------------------
@patch("analysis.engine.estimate_pitch", return_value=None)
@patch("analysis.engine.estimate_formants")
def test_engine_output_structure_is_stable(mock_lpc, mock_pitch):
    mock_lpc.return_value = MagicMock(
        f1=None,
        f2=None,
        f3=None,
        confidence=0.0,
        method="lpc",
        lpc_order=12,
        peaks=[],
        roots=[],
        bandwidths=[],
        debug={},
    )

    eng = FormantAnalysisEngine()
    frame = np.random.randn(1024)

    out = eng.process_frame(frame, sr=48000)

    # All expected keys must exist
    expected_keys = {
        "f0",
        "formants",
        "vowel",
        "vowel_guess",
        "vowel_confidence",
        "vowel_score",
        "resonance_score",
        "overall",
        "fb_f1",
        "fb_f2",
        "segment",
        "confidence",
        "method",
        "lpc_order",
        "peaks",
        "roots",
        "bandwidths",
        "lpc_debug",
    }

    assert expected_keys.issubset(out.keys())
