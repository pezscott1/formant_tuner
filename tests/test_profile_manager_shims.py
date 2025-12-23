# tests/test_profile_manager_shims.py
from unittest.mock import MagicMock
from tuner.profile_controller import ProfileManager


def test_private_load_profile_json_shim(tmp_path):
    pm = ProfileManager(str(tmp_path), analyzer=MagicMock())
    f = tmp_path / "p.json"
    f.write_text('{"voice_type": "bass"}')

    assert pm.load_profile_json(f) == pm.load_profile_json(f)


def test_private_extract_formants_shim():
    pm = ProfileManager("profiles", analyzer=MagicMock())
    raw = {"i": {"f1": 300, "f2": 2500, "f3": 100}}
    assert pm.extract_formants(raw) == pm.extract_formants(raw)
