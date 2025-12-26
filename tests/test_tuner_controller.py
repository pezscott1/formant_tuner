
from unittest.mock import MagicMock, patch
from tuner.controller import Tuner


@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_initializes_components(
        mock_engine_cls, mock_analyzer_cls, mock_profiles_cls):
    mock_engine = MagicMock()
    mock_engine_cls.return_value = mock_engine

    mock_analyzer = MagicMock()
    mock_analyzer_cls.return_value = mock_analyzer

    mock_profiles = MagicMock()
    mock_profiles_cls.return_value = mock_profiles

    t = Tuner(voice_type="bass", profiles_dir="profiles")  # noqa: F841

    # Engine constructed once
    mock_engine_cls.assert_called_once()

    # Analyzer receives engine + smoothers
    mock_analyzer_cls.assert_called_once()
    assert mock_analyzer_cls.call_args[1]["engine"] is mock_engine

    # ProfileManager receives profiles_dir and analyzer=engine
    mock_profiles_cls.assert_called_once_with(
        profiles_dir="profiles", analyzer=mock_engine)


@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_poll_latest_processed_passes_through_analyzer(
    mock_engine_cls, mock_analyzer_cls, mock_profiles_cls
):
    mock_engine = MagicMock()
    mock_engine_cls.return_value = mock_engine

    # Engine returns a raw dict
    raw = {
        "f0": 200.0,
        "formants": (500, 1500, 2500),
        "confidence": 0.8,
        "vowel_guess": "a",
        "vowel_confidence": 0.7,
    }
    mock_engine.get_latest_raw.return_value = raw

    # Analyzer returns processed dict
    processed = {
        "f0": 205.0,
        "formants": (510, 1490, 2500),
        "vowel": "a",
        "confidence": 0.8,
        "stable": True,
        "stability_score": 0.1,
    }
    mock_analyzer = MagicMock()
    mock_analyzer.process_raw.return_value = processed
    mock_analyzer_cls.return_value = mock_analyzer

    mock_profiles = MagicMock()
    mock_profiles_cls.return_value = mock_profiles

    t = Tuner()

    out = t.poll_latest_processed()

    # Analyzer should have been called with raw engine output
    mock_analyzer.process_raw.assert_called_once_with(raw)

    # Output should match analyzer output (no profile loaded)
    assert out == processed


@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_profile_classification_applies_when_stable(
    mock_engine_cls, mock_analyzer_cls, mock_profiles_cls
):
    mock_engine = MagicMock()
    mock_engine_cls.return_value = mock_engine

    # Engine returns raw frame
    mock_engine.get_latest_raw.return_value = {"dummy": True}

    # Analyzer returns stable processed formants
    processed = {
        "f0": 200.0,
        "formants": (500, 1500, 2500),
        "vowel": "a",
        "confidence": 0.9,
        "stable": True,
        "stability_score": 0.05,
    }
    mock_analyzer = MagicMock()
    mock_analyzer.process_raw.return_value = processed
    mock_analyzer_cls.return_value = mock_analyzer

    # ProfileManager returns a profile dict
    mock_profiles = MagicMock()
    mock_profiles.apply_profile.return_value = {
        "i": {"f1": 300, "f2": 2500},
        "a": {"f1": 700, "f2": 1100},
    }
    mock_profiles_cls.return_value = mock_profiles

    t = Tuner()
    t.load_profile("my_profile")

    out = t.poll_latest_processed()

    # Controller should add profile_vowel and profile_confidence
    assert "profile_vowel" in out
    assert "profile_confidence" in out
    assert out["profile_confidence"] >= 0.0


@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_profile_classification_skipped_when_unstable(
    mock_engine_cls, mock_analyzer_cls, mock_profiles_cls
):
    mock_engine = MagicMock()
    mock_engine_cls.return_value = mock_engine

    mock_engine.get_latest_raw.return_value = {"dummy": True}

    # Analyzer returns unstable frame
    processed = {
        "f0": 200.0,
        "formants": (500, 1500, 2500),
        "vowel": "a",
        "confidence": 0.9,
        "stable": False,
    }
    mock_analyzer = MagicMock()
    mock_analyzer.process_raw.return_value = processed
    mock_analyzer_cls.return_value = mock_analyzer

    mock_profiles = MagicMock()
    mock_profiles.apply_profile.return_value = {
        "a": {"f1": 700, "f2": 1100},
    }
    mock_profiles_cls.return_value = mock_profiles

    t = Tuner()
    t.load_profile("my_profile")

    out = t.poll_latest_processed()

    # Unstable → classification suppressed
    assert out["profile_vowel"] is None
    assert out["profile_confidence"] == 0.0
