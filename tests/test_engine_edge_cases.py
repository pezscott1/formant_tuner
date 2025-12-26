import numpy as np
from unittest.mock import patch, MagicMock
from analysis.engine import FormantAnalysisEngine
from analysis.pitch import PitchResult


# ---------------------------------------------------------
# None / empty / zero-length frames
# ---------------------------------------------------------
def test_engine_handles_none_frame():
    eng = FormantAnalysisEngine()
    out = eng.process_frame(None, sr=48000)

    assert out["f0"] is None
    assert out["formants"] == (None, None, None)
    assert out["vowel"] is None
    assert out["overall"] == 0.0
    assert out["method"] == "none"


def test_engine_handles_zero_length_frame():
    eng = FormantAnalysisEngine()
    out = eng.process_frame(np.array([]), sr=48000)

    assert out["f0"] is None
    assert out["formants"] == (None, None, None)
    assert out["vowel"] is None
    assert out["confidence"] == 0.0
    assert out["method"] == "none"


def test_engine_handles_nan_frame():
    eng = FormantAnalysisEngine()
    frame = np.array([np.nan, np.nan, np.nan])

    out = eng.process_frame(frame, sr=48000)

    f0 = out["f0"]
    assert isinstance(f0, PitchResult)
    assert f0.f0 is None  # pitch estimator could not find a valid f0

    f1, f2, f3 = out["formants"]
    assert f1 is None or (isinstance(f1, float) and np.isnan(f1))
    assert f2 is None or (isinstance(f2, float) and np.isnan(f2))
    assert f3 is None or (isinstance(f3, float) and np.isnan(f3))

    assert out["vowel"] is None
    assert out["overall"] == 0.0

# ---------------------------------------------------------
# Extreme sample rates
# ---------------------------------------------------------


@patch("analysis.engine.estimate_pitch", return_value=120.0)
@patch("analysis.engine.estimate_formants")
def test_engine_handles_low_sample_rate(mock_lpc, mock_pitch):
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

    # Engine should still return structured output
    assert out["f0"] == 120.0
    assert out["formants"] == (400.0, 1500.0, 2500.0)
    assert out["confidence"] == 0.8
    assert out["method"] == "lpc"


@patch("analysis.engine.estimate_pitch", return_value=220.0)
@patch("analysis.engine.estimate_formants")
def test_engine_handles_high_sample_rate(mock_lpc, mock_pitch):
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

    assert out["f0"] == 220.0
    assert out["formants"] == (600.0, 1700.0, 2800.0)
    assert out["confidence"] == 0.7
    assert out["method"] == "lpc"


# ---------------------------------------------------------
# Extreme formant values
# ---------------------------------------------------------
@patch("analysis.engine.estimate_pitch", return_value=150.0)
@patch("analysis.engine.estimate_formants")
def test_engine_handles_extreme_formant_values(mock_lpc, mock_pitch):
    mock_lpc.return_value = MagicMock(
        f1=50_000.0,   # absurdly high
        f2=-200.0,     # negative
        f3=999_999.0,  # absurdly high
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

    # Engine should not crash; values propagate as-is
    assert out["formants"][0] == 50_000.0
    assert out["formants"][1] == -200.0
    assert out["formants"][2] == 999_999.0
    assert out["confidence"] == 0.1
    assert out["method"] == "lpc"
