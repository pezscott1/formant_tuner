
from unittest.mock import MagicMock, patch
from tuner.controller import Tuner


# ---------------------------------------------------------
# Profile listing
# ---------------------------------------------------------
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_lists_profiles(mock_engine_cls, mock_analyzer_cls, mock_profiles_cls):
    mock_engine = MagicMock()
    mock_engine_cls.return_value = mock_engine

    mock_analyzer = MagicMock()
    mock_analyzer_cls.return_value = mock_analyzer

    mock_profiles = MagicMock()
    mock_profiles.list_profiles.return_value = ["bass_default", "tenor_1"]
    mock_profiles_cls.return_value = mock_profiles

    t = Tuner()
    profiles = t.list_profiles()

    assert profiles == ["bass_default", "tenor_1"]
    mock_profiles.list_profiles.assert_called_once()


# ---------------------------------------------------------
# Profile deletion
# ---------------------------------------------------------
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_deletes_profile(mock_engine_cls, mock_analyzer_cls, mock_profiles_cls):
    mock_engine = MagicMock()
    mock_engine_cls.return_value = mock_engine

    mock_analyzer = MagicMock()
    mock_analyzer_cls.return_value = mock_analyzer

    mock_profiles = MagicMock()
    mock_profiles_cls.return_value = mock_profiles

    t = Tuner()
    t.delete_profile("my_profile")

    mock_profiles.delete_profile.assert_called_once_with("my_profile")


# ---------------------------------------------------------
# Switching profiles updates active_profile
# ---------------------------------------------------------
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_switches_profiles(mock_engine_cls, mock_analyzer_cls, mock_profiles_cls):
    mock_engine = MagicMock()
    mock_engine_cls.return_value = mock_engine

    mock_analyzer = MagicMock()
    mock_analyzer_cls.return_value = mock_analyzer

    mock_profiles = MagicMock()
    mock_profiles.apply_profile.side_effect = [
        {"a": {"f1": 700, "f2": 1100}},   # first profile
        {"i": {"f1": 300, "f2": 2500}},   # second profile
    ]
    mock_profiles_cls.return_value = mock_profiles

    t = Tuner()

    p1 = t.load_profile("profile_a")
    assert p1 == {"a": {"f1": 700, "f2": 1100}}
    assert t.active_profile == p1

    p2 = t.load_profile("profile_i")
    assert p2 == {"i": {"f1": 300, "f2": 2500}}
    assert t.active_profile == p2


# ---------------------------------------------------------
# Switching profiles changes classification behavior
# ---------------------------------------------------------
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_profile_switch_changes_classification(
    mock_engine_cls, mock_analyzer_cls, mock_profiles_cls
):
    mock_engine = MagicMock()
    mock_engine_cls.return_value = mock_engine

    # Engine always returns a raw frame
    mock_engine.get_latest_raw.return_value = {"dummy": True}

    # Analyzer returns stable formants
    processed = {
        "f0": 200.0,
        "formants": (500, 1500, 2500),
        "vowel": "a",
        "confidence": 0.9,
        "stable": True,
    }
    mock_analyzer = MagicMock()
    mock_analyzer.process_raw.return_value = processed
    mock_analyzer_cls.return_value = mock_analyzer

    # ProfileManager returns two different profiles
    mock_profiles = MagicMock()
    mock_profiles.apply_profile.side_effect = [
        {"a": {"f1": 700, "f2": 1100}},   # first profile
        {"i": {"f1": 300, "f2": 2500}},   # second profile
    ]
    mock_profiles_cls.return_value = mock_profiles

    t = Tuner()

    # Load first profile
    t.load_profile("profile_a")
    out1 = t.poll_latest_processed()
    vowel1 = out1["profile_vowel"]

    # Load second profile
    t.load_profile("profile_i")
    out2 = t.poll_latest_processed()
    vowel2 = out2["profile_vowel"]

    # Classification should differ between profiles
    assert vowel1 != vowel2


# ---------------------------------------------------------
# Voice type override behavior
# ---------------------------------------------------------
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_load_profile_resets_voice_type_when_profile_returns_string(
    mock_engine_cls, mock_analyzer_cls, mock_profiles_cls
):
    mock_engine = MagicMock()
    mock_engine.voice_type = "bass"
    mock_engine_cls.return_value = mock_engine

    mock_analyzer = MagicMock()
    mock_analyzer_cls.return_value = mock_analyzer

    mock_profiles = MagicMock()
    # Simulate apply_profile returning a string (error or message)
    mock_profiles.apply_profile.return_value = "error: missing profile"
    mock_profiles_cls.return_value = mock_profiles

    t = Tuner(voice_type="tenor")

    t.load_profile("bad_profile")

    # Controller resets engine.voice_type to tuner.voice_type
    assert mock_engine.voice_type == "tenor"
