from unittest.mock import MagicMock, patch
from tuner.controller import Tuner


# ---------------------------------------------------------------------
# Helper: create a clean Tuner with patched dependencies
# ---------------------------------------------------------------------
def make_tuner():
    with patch("tuner.controller.FormantAnalysisEngine") as mock_engine_cls, \
         patch("tuner.controller.LiveAnalyzer") as mock_analyzer_cls, \
         patch("tuner.controller.ProfileManager") as mock_profiles_cls:

        engine = MagicMock()
        analyzer = MagicMock()
        profiles = MagicMock()

        mock_engine_cls.return_value = engine
        mock_analyzer_cls.return_value = analyzer
        mock_profiles_cls.return_value = profiles

        t = Tuner()

        return t, engine, analyzer, profiles


# ---------------------------------------------------------------------
# Test: switching profiles calls ProfileManager.apply_profile
# ---------------------------------------------------------------------
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_switches_profiles(mock_engine_cls, mock_analyzer_cls, mock_profiles_cls):
    engine = MagicMock()
    analyzer = MagicMock()
    profiles = MagicMock()

    mock_engine_cls.return_value = engine
    mock_analyzer_cls.return_value = analyzer
    mock_profiles_cls.return_value = profiles

    profiles.apply_profile.side_effect = ["profile_a", "profile_i"]

    t = Tuner()

    # First switch
    t.load_profile("profile_a")
    profiles.apply_profile.assert_any_call("profile_a")

    # Second switch
    t.load_profile("profile_i")
    profiles.apply_profile.assert_any_call("profile_i")


# ---------------------------------------------------------------------
# Test: loading a profile sets active_profile to extracted centroids
# ---------------------------------------------------------------------
def test_load_profile_sets_user_formants():
    t, engine, analyzer, profiles = make_tuner()

    profiles.extract_formants.return_value = {
        "a": {"f1": 500, "f2": 1500, "f0": 100, "confidence": 0.9, "stability": 5}
    }
    profiles.apply_profile.return_value = "alpha"
    profiles.load_profile_json.return_value = {
        "voice_type": "bass",
        "a": {"f1": 500, "f2": 1500, "f3": 2500},
    }

    # Simulate ProfileManager.apply_profile building engine.user_formants
    engine.user_formants = profiles.extract_formants.return_value

    t.load_profile("alpha")

    profiles.apply_profile.assert_called_once_with("alpha")
    profiles.extract_formants.assert_called_once()

    # NEW: active_profile should mirror engine.user_formants
    assert t.active_profile == engine.user_formants


# ---------------------------------------------------------------------
# Test: profile switching changes classification output
# ---------------------------------------------------------------------
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_profile_switch_changes_classification(
    mock_engine_cls, mock_analyzer_cls, mock_profiles_cls
):
    engine = MagicMock()
    analyzer = MagicMock()
    profiles = MagicMock()

    mock_engine_cls.return_value = engine
    mock_analyzer_cls.return_value = analyzer
    mock_profiles_cls.return_value = profiles

    # Engine always returns a raw frame
    engine.get_latest_raw.return_value = {"dummy": True}

    processed = {
        "f0": 200.0,
        "formants": (500, 1500, 2500),
        "vowel": "a",
        "confidence": 0.9,
        "stable": True,
    }
    analyzer.get_latest_processed.return_value = processed

    profiles.apply_profile.side_effect = ["profile_a", "profile_i"]

    # These are the classifier centroids
    profiles.extract_formants.side_effect = [
        {"a": {"f1": 700, "f2": 1100}},
        {"i": {"f1": 300, "f2": 2500}},
    ]

    profiles.load_profile_json.side_effect = [
        {"calibrated_vowels": {"a": {"f1": 700, "f2": 1100}}},
        {"calibrated_vowels": {"i": {"f1": 300, "f2": 2500}}},
    ]

    t = Tuner()

    # Simulate ProfileManager.apply_profile populating engine.user_formants
    engine.user_formants = {"a": {"f1": 700, "f2": 1100}}
    t.load_profile("profile_a")
    out1 = t.poll_latest_processed()
    vowel1 = out1["profile_vowel"]

    engine.user_formants = {"i": {"f1": 300, "f2": 2500}}
    t.load_profile("profile_i")
    out2 = t.poll_latest_processed()
    vowel2 = out2["profile_vowel"]

    assert vowel1 != vowel2
