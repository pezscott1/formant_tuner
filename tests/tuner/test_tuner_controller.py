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

    return SimpleNamespace(
        get_latest_raw=get_latest_raw,
        user_formants=user_formants or {},
        voice_type="bass",
        vowel_hint=None,
    )


def make_analyzer(processed):
    """
    Create a dummy analyzer with get_latest_processed().
    """
    def get_latest_processed():
        return processed

    return SimpleNamespace(get_latest_processed=get_latest_processed)


def make_profiles(apply_profile_result=None, profile_json=None, extracted_profile=None):
    """
    Dummy ProfileManager with:
    - apply_profile(name)
    - load_profile_json(name)
    - extract_formants(raw_json)
    """
    if apply_profile_result is None:
        def apply_profile(*args, **kwargs):
            return None
    else:
        def apply_profile(*args, **kwargs):
            return apply_profile_result

    if profile_json is None:
        profile_json = {
            "i": {"f1": 300, "f2": 2500},
            "a": {"f1": 700, "f2": 1100},
        }

    def load_profile_json(name):
        return profile_json

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
    dummy_engine = make_engine()
    dummy_analyzer = make_analyzer(processed={})
    dummy_profiles = make_profiles()

    mock_engine_cls.return_value = dummy_engine
    mock_analyzer_cls.return_value = dummy_analyzer
    mock_profiles_cls.return_value = dummy_profiles

    t = Tuner(voice_type="bass", profiles_dir="../profiles")  # noqa: F841

    mock_engine_cls.assert_called_once()
    mock_analyzer_cls.assert_called_once()
    assert mock_analyzer_cls.call_args[1]["engine"] is dummy_engine

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
    raw = {"dummy": True}
    dummy_engine = make_engine(raw=raw)
    mock_engine_cls.return_value = dummy_engine

    processed = {
        "f0": 205.0,
        "formants": (510, 1490, 2500),
        "vowel": "a",
        "confidence": 0.8,
        "stable": True,
    }
    dummy_analyzer = make_analyzer(processed=processed)
    mock_analyzer_cls.return_value = dummy_analyzer

    mock_profiles_cls.return_value = make_profiles()

    t = Tuner()

    out = t.poll_latest_processed()

    # No active profile → return processed unchanged
    assert out == processed


@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_profile_classification_applies_when_stable(
    mock_engine_cls, mock_analyzer_cls, mock_profiles_cls
):
    dummy_engine = make_engine(raw={"dummy": True})
    mock_engine_cls.return_value = dummy_engine

    processed = {
        "f0": 200.0,
        "formants": (500, 1500, 2500),
        "vowel": "a",
        "confidence": 0.9,
        "stable": True,
    }
    dummy_analyzer = make_analyzer(processed=processed)
    mock_analyzer_cls.return_value = dummy_analyzer

    profile_json = {
        "i": {"f1": 300, "f2": 2500},
        "a": {"f1": 700, "f2": 1100},
    }
    mock_profiles_cls.return_value = make_profiles(
        apply_profile_result="my_profile",
        profile_json=profile_json,
        extracted_profile=profile_json,
    )

    t = Tuner()
    t.load_profile("my_profile")

    out = t.poll_latest_processed()

    assert "profile_vowel" in out
    assert "profile_confidence" in out
    assert out["profile_confidence"] >= 0.0


@patch("tuner.controller.ProfileManager")
@patch("tuner.controller.LiveAnalyzer")
@patch("tuner.controller.FormantAnalysisEngine")
def test_tuner_profile_classification_skipped_when_unstable(
    mock_engine_cls, mock_analyzer_cls, mock_profiles_cls
):
    dummy_engine = make_engine(raw={"dummy": True})
    mock_engine_cls.return_value = dummy_engine

    processed = {
        "f0": 200.0,
        "formants": (500, 1500, 2500),
        "vowel": "a",
        "confidence": 0.9,
        "stable": False,
    }
    dummy_analyzer = make_analyzer(processed=processed)
    mock_analyzer_cls.return_value = dummy_analyzer

    profile_json = {"a": {"f1": 700, "f2": 1100}}
    mock_profiles_cls.return_value = make_profiles(
        apply_profile_result="my_profile",
        profile_json=profile_json,
        extracted_profile=profile_json,
    )

    t = Tuner()
    t.load_profile("my_profile")

    out = t.poll_latest_processed()

    assert out["profile_vowel"] is None
    assert out["profile_confidence"] == 0.0
