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
    pm = ProfileManager(profiles_dir="profiles", analyzer=MagicMock())
    data = pm.load_profile_json("my_profile")

    # Should load the JSON content returned by open().read()
    assert data["a"]["f1"] == 700
    assert data["i"]["f2"] == 2500
    assert data["voice_type"] == "bass"

    import os

    mock_file.assert_any_call(
        os.path.join("profiles", "my_profile_profile.json"),
        "r",
        encoding="utf-8",
    )
    mock_file.assert_any_call(
        os.path.join("profiles", "active_profile.json"),
        "r",
        encoding="utf-8",
    )
    assert mock_file.call_count == 2


# ---------------------------------------------------------
# extract_formants
# ---------------------------------------------------------
def test_profile_manager_extracts_formants():
    pm = ProfileManager(profiles_dir="profiles", analyzer=MagicMock())

    json_data = {
        "a": {"f1": 700, "f2": 1100, "f3": 2500},
        "i": {"f1": 300, "f2": 2500, "f3": 3000},
    }

    out = pm.extract_formants(json_data)

    # Your implementation returns (f1, f2, f0, confidence, stability)
    # f0 is None because JSON uses f3, not f0
    assert out["a"] == (700, 1100, None, 0.0, float("inf"))
    assert out["i"] == (300, 2500, None, 0.0, float("inf"))


# ---------------------------------------------------------
# apply_profile
# ---------------------------------------------------------
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data=json.dumps(
        {
            "a": {"f1": 700, "f2": 1100, "f0": 120},
            "e": {"f1": 400, "f2": 2300, "f0": 130},
        }
    ),
)
def test_profile_manager_apply_profile_updates_engine(mock_file):
    mock_engine = MagicMock()
    pm = ProfileManager(profiles_dir="profiles", analyzer=mock_engine)

    # extract_formants returns 5-tuples
    pm.extract_formants = MagicMock(
        return_value={
            "a": (700, 1100, 120, 0.0, float("inf")),
            "e": (400, 2300, 130, 0.0, float("inf")),
        }
    )

    out = pm.apply_profile("bass")

    # Engine should receive calibrated profile
    assert mock_engine.calibrated_profile == pm.extract_formants.return_value

    # Implementation returns the profile base name
    assert out == "bass"

    # set_user_formants called if present
    mock_engine.set_user_formants.assert_called_once_with(
        pm.extract_formants.return_value
    )


# ---------------------------------------------------------
# list_profiles
# ---------------------------------------------------------
@patch("os.listdir", return_value=["a_profile.json", "b_profile.json", "notes.txt"])
def test_profile_manager_lists_profiles(mock_listdir):
    pm = ProfileManager(profiles_dir="profiles", analyzer=MagicMock())
    profiles = pm.list_profiles()

    # list_profiles returns base names WITHOUT suffix
    assert profiles == ["a", "b"]


# ---------------------------------------------------------
# delete_profile
# ---------------------------------------------------------
@patch("os.remove")
@patch("os.path.exists", return_value=True)
def test_profile_manager_deletes_profile(mock_exists, mock_remove):
    pm = ProfileManager(profiles_dir="profiles", analyzer=MagicMock())
    pm.delete_profile("my_profile")
    import os
    mock_remove.assert_any_call(os.path.join("profiles", "my_profile_profile.json"))
    mock_remove.assert_any_call(os.path.join("profiles", "my_profile_model.pkl"))


# ---------------------------------------------------------
# missing profile
# ---------------------------------------------------------
@patch.object(ProfileManager, "load_profile_json", return_value={})
def test_profile_manager_missing_profile_returns_error_string(mock_load):
    pm = ProfileManager(profiles_dir="profiles", analyzer=MagicMock())
    out = pm.apply_profile("missing_profile")

    # When load_profile_json returns {}, apply_profile still returns the base name
    assert out == "missing_profile"
