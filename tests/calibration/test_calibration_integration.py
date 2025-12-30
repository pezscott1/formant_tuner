from unittest.mock import MagicMock, patch
from tuner.controller import Tuner


@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.FormantAnalysisEngine")
def test_profile_application_uses_dict_formants(
    mock_engine_cls, mock_profiles_cls, mock_analyzer_cls
):
    # Mock engine
    mock_engine = MagicMock()
    mock_engine_cls.return_value = mock_engine

    # Mock profile manager
    mock_profiles = MagicMock()
    mock_profiles.load_profile_json.return_value = {
        "i": {"f1": 300, "f2": 2200, "f3": 2800},
        "a": {"f1": 700, "f2": 1100, "f3": 2500},
        "voice_type": "bass",
    }
    mock_profiles.extract_formants.return_value = {
        "i": {"f1": 300, "f2": 2200, "f3": 2800},
        "a": {"f1": 700, "f2": 1100, "f3": 2500},
    }
    mock_profiles.apply_profile.return_value = "my_profile"
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

    # Controller should store dict-based active profile
    assert t.active_profile == {
        "i": {"f1": 300, "f2": 2200, "f3": 2800},
        "a": {"f1": 700, "f2": 1100, "f3": 2500},
    }


@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_does_not_mutate_engine_formants(
    mock_engine_cls, mock_profiles_cls, mock_analyzer_cls
):
    mock_engine = MagicMock()
    mock_engine_cls.return_value = mock_engine

    mock_profiles = MagicMock()
    mock_profiles.extract_formants.return_value = {
        "a": {"f1": 700, "f2": 1100, "f3": 2500},
    }
    mock_profiles.apply_profile.return_value = "my_profile"
    mock_profiles_cls.return_value = mock_profiles

    mock_analyzer = MagicMock()
    mock_analyzer_cls.return_value = mock_analyzer

    t = Tuner()

    t.load_profile("my_profile")

    # Old path used to mutate engine.user_formants and set t.cleaned
    assert not hasattr(t, "cleaned")

    # Modern controller should NOT call engine.set_user_formants
    mock_engine.set_user_formants.assert_not_called()
