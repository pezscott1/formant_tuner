from unittest.mock import patch, MagicMock
import numpy as np
from analysis.engine import FormantAnalysisEngine


@patch("analysis.engine.estimate_pitch")
@patch("analysis.engine.estimate_formants")
def test_engine_receives_calibrated_profile(mock_lpc, mock_pitch):
    """Engine must use the calibrated profile set by ProfileManager."""
    mock_pitch.return_value = MagicMock(f0=120.0)
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
