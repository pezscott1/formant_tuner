# tests/test_tuner_controller.py
from unittest.mock import MagicMock, patch
from tuner.controller import Tuner


# ---------------------------------------------------------
# Construction
# ---------------------------------------------------------
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_initialization(mock_engine, mock_analyzer, mock_profiles):
    t = Tuner(voice_type="bass", profiles_dir="profiles")

    mock_engine.assert_called_once_with(voice_type="bass")
    mock_analyzer.assert_called_once()

    _, kwargs = mock_analyzer.call_args
    assert kwargs["engine"] == mock_engine.return_value

    mock_profiles.assert_called_once()
    assert t.voice_type == "bass"
    assert t.active_profile is None

# ---------------------------------------------------------
# Profile operations
# ---------------------------------------------------------


@patch("tuner.controller.ProfileManager")
def test_load_profile(mock_profiles):
    pm = MagicMock()
    pm.apply_profile.return_value = {"i": (300, 2500, 120)}
    mock_profiles.return_value = pm

    t = Tuner()
    profile = t.load_profile("bass_default")

    pm.apply_profile.assert_called_once_with("bass_default")
    assert profile == {"i": (300, 2500, 120)}
    assert t.active_profile == profile


@patch("tuner.controller.ProfileManager")
def test_list_profiles(mock_profiles):
    pm = MagicMock()
    pm.list_profiles.return_value = ["p1", "p2"]
    mock_profiles.return_value = pm

    t = Tuner()
    profiles = t.list_profiles()

    assert profiles == ["p1", "p2"]
    pm.list_profiles.assert_called_once()


@patch("tuner.controller.ProfileManager")
def test_delete_profile(mock_profiles):
    pm = MagicMock()
    mock_profiles.return_value = pm

    t = Tuner()
    t.delete_profile("old")

    pm.delete_profile.assert_called_once_with("old")


# ---------------------------------------------------------
# Audio control
# ---------------------------------------------------------

@patch("tuner.controller.FormantAnalysisEngine")
def test_start_and_stop(mock_engine):
    engine = MagicMock()
    mock_engine.return_value = engine

    t = Tuner()
    t.start()
    t.stop()

    engine.start.assert_not_called()
    engine.stop.assert_not_called()
