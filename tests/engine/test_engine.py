from unittest.mock import patch, MagicMock
import numpy as np
from analysis.engine import FormantAnalysisEngine


@patch("analysis.pitch.estimate_pitch")
@patch("analysis.lpc.estimate_formants")
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

    # Modern dict-based profile
    eng.set_user_formants({
        "a": {"f1": 500, "f2": 1500, "f3": 2500}
    })

    frame = np.random.randn(2048)
    out = eng.process_frame(frame, sr=48000)

    assert "vowel_score" in out
    assert "overall" in out
