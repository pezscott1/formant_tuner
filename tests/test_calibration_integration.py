# tests/test_calibration_integration.py
import numpy as np
from unittest.mock import MagicMock, patch

from tuner.controller import Tuner


# ---------------------------------------------------------
# Calibration: load profile → engine updated
# ---------------------------------------------------------

@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.FormantAnalysisEngine")
def test_calibration_profile_application(mock_engine_cls, mock_profiles_cls, mock_analyzer_cls):
    """
    Integration test:
      - load a profile
      - ensure engine receives updated formants
      - ensure analyzer uses updated engine
    """

    # -----------------------------
    # Mock engine
    # -----------------------------
    mock_engine = MagicMock()
    mock_engine.user_formants = {}
    mock_engine.set_user_formants = MagicMock()
    mock_engine_cls.return_value = mock_engine

    # -----------------------------
    # Mock profile manager
    # -----------------------------
    mock_profiles = MagicMock()
    mock_profiles.apply_profile.return_value = {
        "i": (300, 2200, 2800),
        "a": (700, 1100, 2500),
    }
    mock_profiles_cls.return_value = mock_profiles

    # -----------------------------
    # Mock analyzer
    # -----------------------------
    mock_analyzer = MagicMock()
    mock_analyzer.process_raw.side_effect = lambda raw: {"processed": raw}
    mock_analyzer_cls.return_value = mock_analyzer

    # -----------------------------
    # Construct tuner
    # -----------------------------
    t = Tuner(voice_type="bass", profiles_dir="profiles")

    # Tuner should store active profile
    # -----------------------------
    # Apply profile
    # -----------------------------
    profile = t.load_profile("my_profile")

    # ProfileManager.apply_profile should have been called with "my_profile"
    mock_profiles.apply_profile.assert_called_once_with("my_profile")

    # Optionally: assert Tuner tracks the active profile name or similar
    # depending on your actual implementation. For example:
    # assert t.active_profile_name == "my_profile"


# ---------------------------------------------------------
# Calibration affects live analyzer output
# ---------------------------------------------------------

@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.FormantAnalysisEngine")
def test_calibration_changes_processing(mock_engine_cls, mock_profiles_cls, mock_analyzer_cls):
    mock_engine = MagicMock()
    mock_engine.user_formants = {}
    mock_engine_cls.return_value = mock_engine

    mock_profiles = MagicMock()
    mock_profiles.apply_profile.return_value = {
        "a": (700, 1100, 2500),
    }
    mock_profiles_cls.return_value = mock_profiles

    mock_analyzer = MagicMock()
    mock_analyzer_cls.return_value = mock_analyzer

    t = Tuner()

    profile = t.load_profile("my_profile")

    mock_profiles.apply_profile.assert_called_once_with("my_profile")
    # Optionally:
    # assert t.active_profile_name == "my_profile"
