import json
from unittest.mock import MagicMock
from tuner.profile_controller import ProfileManager


# ---------------------------------------------------------
# Loading a profile that does not exist
# ---------------------------------------------------------

def test_load_missing_profile(tmp_path):
    pm = ProfileManager(str(tmp_path), analyzer=MagicMock())
    result = pm.load_profile_json(tmp_path / "ghost_profile.json")
    assert result == {}  # internal loader returns {} on failure


# ---------------------------------------------------------
# Loading malformed JSON
# ---------------------------------------------------------

def test_load_malformed_json(tmp_path):
    bad = tmp_path / "bad_profile.json"
    bad.write_text("{not valid json")

    pm = ProfileManager(str(tmp_path), analyzer=MagicMock())
    result = pm.load_profile_json(bad)
    assert result == {}  # malformed JSON returns empty dict


# ---------------------------------------------------------
# Extracting formants from incomplete or invalid structures
# ---------------------------------------------------------

def test_extract_formants_handles_invalid_entries(tmp_path):
    pm = ProfileManager(str(tmp_path), analyzer=MagicMock())

    raw = {
        "a": {"f1": 300, "f2": 2500},   # missing f0
        "b": "not a dict",             # invalid entry
        "c": {"f1": None, "f2": None, "f0": None},
    }

    out = pm.extract_formants(raw)

    # Expect (f1, f2, f0, confidence, stability)
    assert out["a"] == (300, 2500, None, 0.0, float("inf"))
    assert "b" not in out
    assert out["c"] == (None, None, None, 0.0, float("inf"))


# ---------------------------------------------------------
# Saving a profile writes JSON and optionally model bytes
# ---------------------------------------------------------

def test_save_profile_writes_json_and_model(tmp_path):
    pm = ProfileManager(str(tmp_path), analyzer=MagicMock())
    pm.voice_type = "baritone"
    pm.analyzer.voice_type = "baritone"
    data = {"a": {"f1": 300, "f2": 2500, "f0": 120}}
    model_bytes = b"fake model"

    pm.save_profile("bass", data, model_bytes=model_bytes)

    json_path = tmp_path / "bass_profile.json"
    model_path = tmp_path / "bass_model.pkl"

    assert json_path.exists()
    assert model_path.exists()

    loaded = json.loads(json_path.read_text())
    assert loaded == data

    assert model_path.read_bytes() == model_bytes


# ---------------------------------------------------------
# set_active_profile writes active_profile.json
# ---------------------------------------------------------

def test_set_active_profile_writes_file(tmp_path):
    pm = ProfileManager(str(tmp_path), analyzer=MagicMock())

    pm.set_active_profile("tenor")

    active_path = tmp_path / "active_profile.json"
    assert active_path.exists()

    data = json.loads(active_path.read_text())
    assert data["active"] == "tenor"
    assert pm.active_profile_name == "tenor"


# ---------------------------------------------------------
# _load_active_profile handles missing and malformed files
# ---------------------------------------------------------

def test_load_active_profile_missing_file(tmp_path):
    pm = ProfileManager(str(tmp_path), analyzer=MagicMock())
    assert pm.active_profile_name is None  # nothing loaded


def test_load_active_profile_malformed(tmp_path):
    bad = tmp_path / "active_profile.json"
    bad.write_text("{not valid json")

    pm = ProfileManager(str(tmp_path), analyzer=MagicMock())
    assert pm.active_profile_name is None  # gracefully ignored


# ---------------------------------------------------------
# apply_profile loads JSON, extracts formants, calls analyzer
# ---------------------------------------------------------

def test_apply_profile_calls_analyzer(tmp_path):
    profile_path = tmp_path / "bass_profile.json"
    profile_path.write_text(json.dumps({
        "a": {"f1": 300, "f2": 2500, "f0": 120},
        "e": {"f1": 400, "f2": 2300, "f0": 130},
    }))

    mock_analyzer = MagicMock()
    pm = ProfileManager(str(tmp_path), analyzer=mock_analyzer)

    result = pm.apply_profile("bass")
    assert result == "bass"

    mock_analyzer.set_user_formants.assert_called_once_with({
        "a": (300, 2500, 120, 0.0, float("inf")),
        "e": (400, 2300, 130, 0.0, float("inf")),
    })

# ---------------------------------------------------------
# delete_profile removes JSON, model, and clears active profile
# ---------------------------------------------------------


def test_delete_profile_clears_active(tmp_path):
    # Create profile + model + active file
    json_path = tmp_path / "bass_profile.json"
    model_path = tmp_path / "bass_model.pkl"
    active_path = tmp_path / "active_profile.json"

    json_path.write_text("{}")
    model_path.write_text("model")
    active_path.write_text(json.dumps({"active": "bass"}))

    pm = ProfileManager(str(tmp_path), analyzer=MagicMock())
    pm.active_profile_name = "bass"

    pm.delete_profile("bass")

    assert not json_path.exists()
    assert not model_path.exists()
    assert not active_path.exists()
    assert pm.active_profile_name is None


def test_profile_exists_false(tmp_path):
    pm = ProfileManager(str(tmp_path), analyzer=MagicMock())
    # No profiles exist
    assert pm.profile_exists("ghost") is False


def test_extract_formants_non_dict_type(tmp_path):
    pm = ProfileManager(str(tmp_path), analyzer=MagicMock())
    # Passing a list should return empty dict
    out = pm.extract_formants(["not", "a", "dict"])
    assert out == {}
