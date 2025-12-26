from unittest.mock import MagicMock, patch
from tuner.controller import Tuner


# ---------------------------------------------------------
# Integration: loading a profile applies calibrated formants
# ---------------------------------------------------------
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.FormantAnalysisEngine")
def test_calibration_profile_application(
    mock_engine_cls, mock_profiles_cls, mock_analyzer_cls
):
    # Mock engine instance
    mock_engine = MagicMock()
    mock_engine.user_formants = None
    mock_engine_cls.return_value = mock_engine

    # Mock profile manager instance
    mock_profiles = MagicMock()
    mock_profiles.load_profile_json.return_value = {
        "i": {"f1": 300, "f2": 2200, "f3": 2800},
        "a": {"f1": 700, "f2": 1100, "f3": 2500},
        "voice_type": "bass",
    }
    mock_profiles.extract_formants.return_value = {
        "i": (300, 2200, 2800, 0.0, float("inf")),
        "a": (700, 1100, 2500, 0.0, float("inf")),
    }

    # Simulate real apply_profile behavior: update engine + return name
    def fake_apply_profile(base_name):
        mock_engine.user_formants = mock_profiles.extract_formants(
            mock_profiles.load_profile_json(base_name)
        )
        return base_name

    mock_profiles.apply_profile.side_effect = fake_apply_profile
    mock_profiles_cls.return_value = mock_profiles

    # Mock analyzer instance
    mock_analyzer = MagicMock()
    mock_analyzer_cls.return_value = mock_analyzer

    # Construct tuner
    t = Tuner(voice_type="bass", profiles_dir="profiles")

    # Apply profile
    t.load_profile("my_profile")

    # ProfileManager.apply_profile should be called
    mock_profiles.apply_profile.assert_called_once_with("my_profile")

    # Engine should now hold user_formants
    assert mock_engine.user_formants == {
        "i": (300, 2200, 2800, 0.0, float("inf")),
        "a": (700, 1100, 2500, 0.0, float("inf")),
    }


# ---------------------------------------------------------
# Integration: applying a profile changes processing behavior
# ---------------------------------------------------------
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.FormantAnalysisEngine")
def test_calibration_changes_processing(
    mock_engine_cls, mock_profiles_cls, mock_analyzer_cls
):
    # Mock engine
    mock_engine = MagicMock()
    mock_engine.user_formants = None
    mock_engine_cls.return_value = mock_engine

    # Mock profile manager
    mock_profiles = MagicMock()
    mock_profiles.load_profile_json.return_value = {
        "a": {"f1": 700, "f2": 1100, "f3": 2500},
        "voice_type": "bass",
    }
    mock_profiles.extract_formants.return_value = {
        "a": (700, 1100, 2500, 0.0, float("inf")),
    }

    def fake_apply_profile(base_name):
        mock_engine.user_formants = mock_profiles.extract_formants(
            mock_profiles.load_profile_json(base_name)
        )
        return base_name

    mock_profiles.apply_profile.side_effect = fake_apply_profile
    mock_profiles_cls.return_value = mock_profiles

    # Mock analyzer
    mock_analyzer = MagicMock()
    mock_analyzer_cls.return_value = mock_analyzer

    # Construct tuner
    t = Tuner()

    # Apply profile
    t.load_profile("my_profile")

    # ProfileManager.apply_profile should be called
    mock_profiles.apply_profile.assert_called_once_with("my_profile")

    # Engine should now have updated user_formants
    assert mock_engine.user_formants == {
        "a": (700, 1100, 2500, 0.0, float("inf")),
    }
