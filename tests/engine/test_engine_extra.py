from unittest.mock import patch, MagicMock
import numpy as np
from analysis.engine import FormantAnalysisEngine


@patch("analysis.pitch.estimate_pitch")
@patch("analysis.lpc.estimate_formants")
def test_engine_propagates_formant_confidence(mock_lpc, mock_pitch):
    mock_pitch.return_value = MagicMock(f0=200.0)
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


@patch("analysis.pitch.estimate_pitch")
@patch("analysis.lpc.estimate_formants")
def test_engine_handles_extreme_formant_values(mock_lpc, mock_pitch):
    mock_pitch.return_value = MagicMock(f0=150.0)
    mock_lpc.return_value = MagicMock(
        f1=50_000.0,
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

    assert "f0" in out
    assert "formants" in out


@patch("analysis.pitch.estimate_pitch")
@patch("analysis.lpc.estimate_formants")
def test_engine_low_confidence_disables_vowel_guess(mock_lpc, mock_pitch):
    mock_pitch.return_value = MagicMock(f0=150.0)
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

    assert out["vowel_guess"] is None or out["vowel_confidence"] == 0.0


@patch("analysis.pitch.estimate_pitch")
@patch("analysis.lpc.estimate_formants")
def test_engine_output_structure_is_stable(mock_lpc, mock_pitch):
    mock_pitch.return_value = MagicMock(f0=None)
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
        "lpc_debug",
    }
    assert expected_keys.issubset(out.keys())
