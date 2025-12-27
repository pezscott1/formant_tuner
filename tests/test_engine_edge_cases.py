from unittest.mock import patch, MagicMock
import numpy as np
from analysis.engine import FormantAnalysisEngine


def test_engine_handles_nan_frame():
    eng = FormantAnalysisEngine()
    frame = np.array([np.nan, np.nan, np.nan])

    out = eng.process_frame(frame, sr=48000)

    # Current behavior: treat as empty/invalid, f0 becomes None
    assert out["f0"] is None
    assert out["formants"] == (None, None, None)


@patch("analysis.engine.estimate_pitch")
@patch("analysis.engine.estimate_formants")
def test_engine_handles_low_sample_rate(mock_lpc, mock_pitch):
    mock_pitch.return_value = MagicMock(f0=120.0)
    mock_lpc.return_value = MagicMock(
        f1=400.0,
        f2=1500.0,
        f3=2500.0,
        confidence=0.8,
        method="lpc",
        lpc_order=10,
        peaks=[],
        roots=[],
        bandwidths=[],
        debug={},
    )

    eng = FormantAnalysisEngine()
    frame = np.random.randn(1024)

    out = eng.process_frame(frame, sr=8000)

    assert "f0" in out
    assert "formants" in out


@patch("analysis.engine.estimate_pitch")
@patch("analysis.engine.estimate_formants")
def test_engine_handles_high_sample_rate(mock_lpc, mock_pitch):
    mock_pitch.return_value = MagicMock(f0=220.0)
    mock_lpc.return_value = MagicMock(
        f1=600.0,
        f2=1700.0,
        f3=2800.0,
        confidence=0.7,
        method="lpc",
        lpc_order=14,
        peaks=[],
        roots=[],
        bandwidths=[],
        debug={},
    )

    eng = FormantAnalysisEngine()
    frame = np.random.randn(4096)

    out = eng.process_frame(frame, sr=96000)

    assert "f0" in out
    assert "formants" in out


@patch("analysis.engine.estimate_pitch")
@patch("analysis.engine.estimate_formants")
def test_engine_handles_extreme_formant_values(mock_lpc, mock_pitch):
    mock_pitch.return_value = MagicMock(f0=150.0)
    mock_lpc.return_value = MagicMock(
        f1=50_000.0,
        f2=-200.0,
        f3=999_999.0,
        confidence=0.1,
        method="lpc",
        lpc_order=12,
        peaks=[],
        roots=[],
        bandwidths=[],
        debug={},
    )

    eng = FormantAnalysisEngine()
    frame = np.random.randn(2048)

    out = eng.process_frame(frame, sr=48000)

    # We only assert that the engine doesn't crash and returns a sane dict
    assert "f0" in out
    assert "formants" in out
    assert "overall" in out
