from types import SimpleNamespace
from unittest.mock import patch
from tuner.controller import Tuner


# ---------------------------------------------------------
# Helpers for simple dummy objects (no nested MagicMocks)
# ---------------------------------------------------------
def make_engine(raw=None, user_formants=None):
    """
    Create a simple dummy engine object with the minimal API
    used by Tuner in these tests.
    """
    if raw is None:
        raw = {}

    def get_latest_raw():
        return raw

    # voice_type and vowel_hint may be set by Tuner
    return SimpleNamespace(
        get_latest_raw=get_latest_raw,
        user_formants=user_formants or {},
        voice_type="bass",
        vowel_hint=None,
    )


def make_analyzer(processed):
    """
    Create a simple dummy analyzer with a process_raw method.
    """
    def process_raw(raw):
        return processed

    return SimpleNamespace(process_raw=process_raw)


def make_profiles(apply_profile_result=None, profile_json=None, extracted_profile=None):
    """
    Create a simple dummy profile manager with:
    - apply_profile(name)
    - load_profile_json(name)
    - extract_formants(raw_json)
    """
    # apply_profile
    if apply_profile_result is None:
        def apply_profile(*args, **kwargs):
            return None
    else:
        def apply_profile(*args, **kwargs):
            return apply_profile_result

    # profile JSON returned by load_profile_json
    if profile_json is None:
        profile_json = {
            "i": {"f1": 300, "f2": 2500},
            "a": {"f1": 700, "f2": 1100},
        }

    def load_profile_json(name):
        return profile_json

    # extracted centroids returned by extract_formants
    if extracted_profile is None:
        extracted_profile = profile_json

    def extract_formants(raw):
        return extracted_profile

    return SimpleNamespace(
        apply_profile=apply_profile,
        load_profile_json=load_profile_json,
        extract_formants=extract_formants,
        list_profiles=lambda: ["my_profile"],
        delete_profile=lambda name: None,
    )


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------
@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_initializes_components(
    mock_engine_cls, mock_analyzer_cls, mock_profiles_cls
):
    # Use simple dummy objects instead of MagicMock instances
    dummy_engine = make_engine()
    dummy_analyzer = make_analyzer(processed={})
    dummy_profiles = make_profiles()

    mock_engine_cls.return_value = dummy_engine
    mock_analyzer_cls.return_value = dummy_analyzer
    mock_profiles_cls.return_value = dummy_profiles

    t = Tuner(voice_type="bass", profiles_dir="../profiles")  # noqa: F841

    # Engine constructed once
    mock_engine_cls.assert_called_once()

    # Analyzer receives engine + smoothers
    mock_analyzer_cls.assert_called_once()
    assert mock_analyzer_cls.call_args[1]["engine"] is dummy_engine

    # ProfileManager receives profiles_dir and analyzer=engine
    mock_profiles_cls.assert_called_once_with(
        profiles_dir="../profiles",
        analyzer=dummy_engine,
    )


@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_poll_latest_processed_passes_through_analyzer(
    mock_engine_cls, mock_analyzer_cls, mock_profiles_cls
):
    # Engine returns a raw dict
    raw = {
        "f0": 200.0,
        "formants": (500, 1500, 2500),
        "confidence": 0.8,
        "vowel_guess": "a",
        "vowel_confidence": 0.7,
    }
    dummy_engine = make_engine(raw=raw)
    mock_engine_cls.return_value = dummy_engine

    # Analyzer returns processed dict
    processed = {
        "f0": 205.0,
        "formants": (510, 1490, 2500),
        "vowel": "a",
        "confidence": 0.8,
        "stable": True,
        "stability_score": 0.1,
    }
    dummy_analyzer = make_analyzer(processed=processed)
    mock_analyzer_cls.return_value = dummy_analyzer

    dummy_profiles = make_profiles()
    mock_profiles_cls.return_value = dummy_profiles

    t = Tuner()

    out = t.poll_latest_processed()

    # No active profile → output should match analyzer output
    assert out == processed


@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_profile_classification_applies_when_stable(
    mock_engine_cls, mock_analyzer_cls, mock_profiles_cls
):
    # Engine with user_formants and dummy raw
    dummy_engine = make_engine(
        raw={"dummy": True},
        user_formants={
            "i": {"f1": 300, "f2": 2500},
            "a": {"f1": 700, "f2": 1100},
        },
    )
    mock_engine_cls.return_value = dummy_engine

    processed = {
        "f0": 200.0,
        "formants": (500, 1500, 2500),
        "vowel": "a",
        "confidence": 0.9,
        "stable": True,
        "stability_score": 0.05,
    }
    dummy_analyzer = make_analyzer(processed=processed)
    mock_analyzer_cls.return_value = dummy_analyzer

    # Profile with centroids for i and a
    profile_json = {
        "i": {"f1": 300, "f2": 2500},
        "a": {"f1": 700, "f2": 1100},
    }
    dummy_profiles = make_profiles(
        apply_profile_result="my_profile",
        profile_json=profile_json,
        extracted_profile=profile_json,
    )
    mock_profiles_cls.return_value = dummy_profiles

    t = Tuner()
    t.load_profile("my_profile")

    out = t.poll_latest_processed()

    assert "profile_vowel" in out
    assert "profile_confidence" in out
    # Should be some non-negative confidence
    assert out["profile_confidence"] >= 0.0


@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_profile_classification_skipped_when_unstable(
    mock_engine_cls, mock_analyzer_cls, mock_profiles_cls
):
    dummy_engine = make_engine(
        raw={"dummy": True},
        user_formants={
            "a": {"f1": 700, "f2": 1100},
        },
    )
    mock_engine_cls.return_value = dummy_engine

    # Analyzer returns unstable frame
    processed = {
        "f0": 200.0,
        "formants": (500, 1500, 2500),
        "vowel": "a",
        "confidence": 0.9,
        "stable": False,
    }
    dummy_analyzer = make_analyzer(processed=processed)
    mock_analyzer_cls.return_value = dummy_analyzer

    profile_json = {
        "a": {"f1": 700, "f2": 1100},
    }
    dummy_profiles = make_profiles(
        apply_profile_result="my_profile",
        profile_json=profile_json,
        extracted_profile=profile_json,
    )
    mock_profiles_cls.return_value = dummy_profiles

    t = Tuner()
    t.load_profile("my_profile")

    out = t.poll_latest_processed()

    # Unstable → classification suppressed
    assert out["profile_vowel"] is None
    assert out["profile_confidence"] == 0.0
