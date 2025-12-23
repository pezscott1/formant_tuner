# tests/test_profile_manager.py
import json
from unittest.mock import MagicMock
from tuner.profile_controller import ProfileManager


def test_list_profiles(tmp_path):
    # Create fake profile files
    (tmp_path / "bass_profile.json").write_text("{}")
    (tmp_path / "tenor_profile.json").write_text("{}")
    (tmp_path / "active_profile.json").write_text("{}")  # should be ignored
    (tmp_path / "notes.txt").write_text("ignore me")

    pm = ProfileManager(str(tmp_path), analyzer=MagicMock())
    profiles = pm.list_profiles()

    assert profiles == ["bass", "tenor"]


def test_display_name_and_base_from_display():
    assert ProfileManager.display_name("bass_profile") == "bass profile"
    assert ProfileManager.base_from_display("bass profile") == "bass_profile"

    # Special case: "➕ New Profile"
    assert ProfileManager.base_from_display("➕ New Profile") is None


def test_delete_profile_removes_json_and_model(tmp_path):
    # Create profile + model
    json_path = tmp_path / "bass_profile.json"
    model_path = tmp_path / "bass_model.pkl"

    json_path.write_text("{}")
    model_path.write_text("model")

    pm = ProfileManager(str(tmp_path), analyzer=MagicMock())
    pm.delete_profile("bass")

    assert not json_path.exists()
    assert not model_path.exists()


def test_apply_profile_loads_json_and_calls_set_user_formants(tmp_path):
    # Create fake profile JSON
    profile_path = tmp_path / "bass_profile.json"
    profile_path.write_text(json.dumps({
        "a": {"f1": 300, "f2": 2500, "f0": 120},
        "e": {"f1": 400, "f2": 2300, "f0": 130},
    }))

    mock_analyzer = MagicMock()
    pm = ProfileManager(str(tmp_path), analyzer=mock_analyzer)

    result = pm.apply_profile("bass")

    # Should return the base name
    assert result == "bass"

    # Should call analyzer.set_user_formants with extracted tuples
    mock_analyzer.set_user_formants.assert_called_once_with({
        "a": (300, 2500, 120),
        "e": (400, 2300, 130),
    })

    # Should update active_profile_name
    assert pm.active_profile_name == "bass"

    # active_profile.json should be written
    active_path = tmp_path / "active_profile.json"
    assert active_path.exists()
    data = json.loads(active_path.read_text())
    assert data["active"] == "bass"
