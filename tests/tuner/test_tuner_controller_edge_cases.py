
from unittest.mock import MagicMock, patch
from tuner.controller import Tuner


# ---------------------------------------------------------
# Case 1: engine returns None → controller returns None
# ---------------------------------------------------------
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_poll_latest_processed_returns_none_when_engine_has_no_frame(
    mock_engine_cls, mock_analyzer_cls, mock_profiles_cls
):
    mock_engine = MagicMock()
    mock_engine.get_latest_raw.return_value = None
    mock_engine_cls.return_value = mock_engine

    mock_analyzer = MagicMock()
    mock_analyzer_cls.return_value = mock_analyzer

    mock_profiles = MagicMock()
    mock_profiles_cls.return_value = mock_profiles

    t = Tuner()
    assert t.poll_latest_processed() is None


# ---------------------------------------------------------
# Case 2: analyzer returns None → controller returns None
# ---------------------------------------------------------
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_poll_latest_processed_returns_none_when_analyzer_returns_none(
    mock_engine_cls, mock_analyzer_cls, mock_profiles_cls
):
    mock_engine = MagicMock()
    mock_engine.get_latest_raw.return_value = {"dummy": True}
    mock_engine_cls.return_value = mock_engine

    mock_analyzer = MagicMock()
    mock_analyzer.process_raw.return_value = None
    mock_analyzer_cls.return_value = mock_analyzer

    mock_profiles = MagicMock()
    mock_profiles_cls.return_value = mock_profiles

    t = Tuner()
    assert t.poll_latest_processed() is None


# ---------------------------------------------------------
# Case 3: processed dict missing "formants" → controller returns processed unchanged
# ---------------------------------------------------------
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_poll_latest_processed_missing_formants_skips_profile_classification(
    mock_engine_cls, mock_analyzer_cls, mock_profiles_cls
):
    mock_engine = MagicMock()
    mock_engine.get_latest_raw.return_value = {"dummy": True}
    mock_engine_cls.return_value = mock_engine

    processed = {
        "f0": 200.0,
        "vowel": "a",
        "confidence": 0.8,
        "stable": True,
    }

    mock_analyzer = MagicMock()
    mock_analyzer.process_raw.return_value = processed
    mock_analyzer_cls.return_value = mock_analyzer

    mock_profiles = MagicMock()
    mock_profiles.apply_profile.return_value = "my_profile"
    mock_engine.user_formants = {
        "a": {"f1": 700, "f2": 1100},
    }
    mock_profiles_cls.return_value = mock_profiles

    t = Tuner()
    t.load_profile("my_profile")

    out = t.poll_latest_processed()

    # No formants → classification skipped
    assert out == processed
    assert "profile_vowel" not in out
    assert "profile_confidence" not in out


# ---------------------------------------------------------
# Case 4: formants exist but are None → classification suppressed
# ---------------------------------------------------------
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_poll_latest_processed_none_formants_suppresses_classification(
    mock_engine_cls, mock_analyzer_cls, mock_profiles_cls
):
    mock_engine = MagicMock()
    mock_engine.get_latest_raw.return_value = {"dummy": True}
    mock_engine_cls.return_value = mock_engine

    processed = {
        "f0": 200.0,
        "formants": (None, None, None),
        "vowel": "a",
        "confidence": 0.8,
        "stable": True,
    }

    mock_analyzer = MagicMock()
    mock_analyzer.process_raw.return_value = processed
    mock_analyzer_cls.return_value = mock_analyzer

    mock_profiles = MagicMock()
    mock_profiles.apply_profile.return_value = "my_profile"
    mock_engine.user_formants = {
        "a": {"f1": 700, "f2": 1100},
    }
    mock_profiles_cls.return_value = mock_profiles

    t = Tuner()
    t.load_profile("my_profile")

    out = t.poll_latest_processed()

    assert out["profile_vowel"] is None
    assert out["profile_confidence"] == 0.0


# ---------------------------------------------------------
# Case 5: profile missing fields → classification returns None
# ---------------------------------------------------------
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_poll_latest_processed_profile_missing_fields(
    mock_engine_cls, mock_analyzer_cls, mock_profiles_cls
):
    mock_engine = MagicMock()
    mock_engine.get_latest_raw.return_value = {"dummy": True}
    mock_engine_cls.return_value = mock_engine

    processed = {
        "f0": 200.0,
        "formants": (500, 1500, 2500),
        "vowel": "a",
        "confidence": 0.8,
        "stable": True,
    }

    mock_analyzer = MagicMock()
    mock_analyzer.process_raw.return_value = processed
    mock_analyzer_cls.return_value = mock_analyzer

    # Missing f2 → invalid centroid
    mock_profiles = MagicMock()
    mock_profiles.apply_profile.return_value = "my_profile"
    mock_engine.user_formants = {
        "a": {"f1": 700},  # missing f2 → invalid
    }
    mock_profiles_cls.return_value = mock_profiles

    t = Tuner()
    t.load_profile("my_profile")

    out = t.poll_latest_processed()

    assert out["profile_vowel"] is None
    assert out["profile_confidence"] == 0.0


# ---------------------------------------------------------
# Case 6: unstable frame → classification suppressed
# ---------------------------------------------------------
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_poll_latest_processed_unstable_frame_suppresses_classification(
    mock_engine_cls, mock_analyzer_cls, mock_profiles_cls
):
    mock_engine = MagicMock()
    mock_engine.get_latest_raw.return_value = {"dummy": True}
    mock_engine_cls.return_value = mock_engine

    processed = {
        "f0": 200.0,
        "formants": (500, 1500, 2500),
        "vowel": "a",
        "confidence": 0.8,
        "stable": False,  # unstable
    }

    mock_analyzer = MagicMock()
    mock_analyzer.process_raw.return_value = processed
    mock_analyzer_cls.return_value = mock_analyzer

    mock_profiles = MagicMock()
    mock_profiles.apply_profile.return_value = "my_profile"
    mock_engine.user_formants = {
        "a": {"f1": 700, "f2": 1100},
    }
    mock_profiles_cls.return_value = mock_profiles

    t = Tuner()
    t.load_profile("my_profile")

    out = t.poll_latest_processed()

    assert out["profile_vowel"] is None
    assert out["profile_confidence"] == 0.0
