import numpy as np
from unittest.mock import MagicMock, patch
from analysis.engine import FormantAnalysisEngine


def test_engine_singleton_behavior():
    """Each engine instance must maintain its own state."""
    e1 = FormantAnalysisEngine()
    e2 = FormantAnalysisEngine()

    # Different instances must not share calibrated profiles
    e1.calibrated_profile = {"a": (500, 1500, 2500)}
    e2.calibrated_profile = {"i": (300, 2500, 3000)}

    assert e1.calibrated_profile != e2.calibrated_profile
    assert e1.calibrated_profile == {"a": (500, 1500, 2500)}
    assert e2.calibrated_profile == {"i": (300, 2500, 3000)}

    # Latest raw must also be instance-local
    e1._latest_raw = {"f0": 100}
    e2._latest_raw = {"f0": 200}

    assert e1.get_latest_raw() == {"f0": 100}
    assert e2.get_latest_raw() == {"f0": 200}


@patch("analysis.engine.estimate_pitch", return_value=120.0)
@patch("analysis.engine.estimate_formants")
def test_engine_receives_calibrated_profile(mock_lpc, mock_pitch):
    """Engine must use the calibrated profile set by ProfileManager."""
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

    # Engine must retain the calibrated profile
    assert eng.user_formants == {"a": (500, 1500, 2500)}

    # Scoring should be computed using the calibrated profile
    assert out["vowel_score"] >= 0.0
    assert out["overall"] >= 0.0
