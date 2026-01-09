import json
from unittest.mock import MagicMock, mock_open, patch
from tuner.profile_controller import ProfileManager


# ---------------------------------------------------------
# load_profile_json
# ---------------------------------------------------------
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data=json.dumps(
        {
            "a": {"f1": 700, "f2": 1100, "f3": 2500},
            "i": {"f1": 300, "f2": 2500, "f3": 3000},
            "voice_type": "bass",
        }
    ),
)
def test_profile_manager_loads_json(mock_file):
    pm = ProfileManager(profiles_dir="", analyzer=MagicMock())
    data = pm.load_profile_json("my_profile")

    # Should load the JSON content returned by open().read()
    assert data["a"]["f1"] == 700
    assert data["i"]["f2"] == 2500
    assert data["voice_type"] == "bass"

    import os

    mock_file.assert_any_call(
        os.path.join("", "my_profile_profile.json"),
        "r",
        encoding="utf-8",
    )
    mock_file.assert_any_call(
        os.path.join("", "active_profile.json"),
        "r",
        encoding="utf-8",
    )
    assert mock_file.call_count >= 2


# ---------------------------------------------------------
# extract_formants
# ---------------------------------------------------------
def test_profile_manager_extracts_formants():
    pm = ProfileManager(profiles_dir="", analyzer=MagicMock())

    json_data = {
        "a": {"f1": 700, "f2": 1100, "f3": 2500},
        "i": {"f1": 300, "f2": 2500, "f0": 3000},
    }

    out = pm.extract_formants(json_data)

    assert out["a"] == {
        "f1": 700,
        "f2": 1100,
        "f0": 2500,      # f3 → f0 fallback
        "confidence": 0.0,
        "stability": float("inf"),
    }

    assert out["i"] == {
        "f1": 300,
        "f2": 2500,
        "f0": 3000,      # explicit f0 wins
        "confidence": 0.0,
        "stability": float("inf"),
    }


# ---------------------------------------------------------
# apply_profile
# ---------------------------------------------------------
def test_profile_manager_apply_profile_updates_engine(tmp_path):
    prof_dir = tmp_path
    analyzer = MagicMock()

    # Create a profile file
    data = {
        "calibrated_vowels": {"a": {"f1": 500, "f2": 1500, "f0": 2500}},
        "interpolated_vowels": {}
    }
    with open(prof_dir / "alpha_profile.json", "w") as f:
        json.dump(data, f)

    pm = ProfileManager(prof_dir, analyzer)
    pm.apply_profile("alpha")

    analyzer.set_user_formants.assert_called_once_with({
        "a": {
            "f1": 500,
            "f2": 1500,
            "f0": 2500,
            "confidence": 0.0,
            "stability": float("inf"),
        }
    })

    assert analyzer.voice_type == "tenor"
    assert pm.active_profile_name == "alpha"


def test_apply_profile_sets_user_formants():
    mock_engine = MagicMock()
    pm = ProfileManager("", mock_engine)

    raw = {"calibrated_vowels": {"a": {"f1": 500, "f2": 1500, "f0": 2500}},
           "interpolated_vowels": {}}
    pm.load_profile_json = MagicMock(return_value=raw)

    pm.extract_formants = MagicMock(return_value={
        "a": {"f1": 500, "f2": 1500, "f0": 2500}})

    pm.apply_profile("alpha")

    pm.extract_formants.assert_called_once_with(raw["calibrated_vowels"] | raw["interpolated_vowels"])
    mock_engine.set_user_formants.assert_called_once_with(
        {"a": {"f1": 500, "f2": 1500, "f0": 2500}}
    )

# ---------------------------------------------------------
# list_profiles
# ---------------------------------------------------------


@patch("os.listdir", return_value=["a_profile.json", "b_profile.json", "notes.txt"])
def test_profile_manager_lists_profiles(mock_listdir):
    pm = ProfileManager(profiles_dir="", analyzer=MagicMock())
    profiles = pm.list_profiles()

    # list_profiles returns base names WITHOUT suffix
    assert profiles == ["a", "b"]


# ---------------------------------------------------------
# delete_profile
# ---------------------------------------------------------
@patch("os.remove")
@patch("os.path.exists", return_value=True)
def test_profile_manager_deletes_profile(mock_exists, mock_remove):
    pm = ProfileManager(profiles_dir="", analyzer=MagicMock())
    pm.delete_profile("my_profile")
    import os
    mock_remove.assert_any_call(os.path.join("", "my_profile_profile.json"))
    mock_remove.assert_any_call(os.path.join("", "my_profile_model.pkl"))


# ---------------------------------------------------------
# missing profile
# ---------------------------------------------------------
@patch.object(ProfileManager, "load_profile_json", return_value={})
def test_profile_manager_missing_profile_returns_error_string(mock_load):
    pm = ProfileManager(profiles_dir="", analyzer=MagicMock())
    out = pm.apply_profile("missing_profile")

    # When load_profile_json returns {}, apply_profile still returns the base name
    assert out == "missing_profile"
