import json
import pytest
from tuner.profile_controller import ProfileManager


# ----------------------------------------------------------------------
# Dummy analyzer
# ----------------------------------------------------------------------

class DummyAnalyzer:
    def __init__(self):
        self.voice_type = "bass"
        self.calibrated_profile = None
        self.set_user_formants_called = False
        self.reset_called = False
        self.user_formants = None

    def set_user_formants(self, d):
        self.set_user_formants_called = True
        self.user_formants = d

    def reset(self):
        self.reset_called = True


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def tmpdir_profiles(tmp_path):
    d = tmp_path / "profiles"
    d.mkdir()
    return d


@pytest.fixture
def analyzer():
    return DummyAnalyzer()


@pytest.fixture
def pm(tmpdir_profiles, analyzer):
    return ProfileManager(str(tmpdir_profiles), analyzer)


# ----------------------------------------------------------------------
# list_profiles
# ----------------------------------------------------------------------

def test_list_profiles_empty(pm):
    assert pm.list_profiles() == []


def test_list_profiles_multiple(pm, tmpdir_profiles):
    (tmpdir_profiles / "alpha_profile.json").write_text("{}")
    (tmpdir_profiles / "beta_profile.json").write_text("{}")
    (tmpdir_profiles / "active_profile.json").write_text("{}")  # ignored
    assert pm.list_profiles() == ["alpha", "beta"]


# ----------------------------------------------------------------------
# name conversion
# ----------------------------------------------------------------------

def test_display_name():
    assert ProfileManager.display_name("my_profile") == "my profile"


# ----------------------------------------------------------------------
# profile_exists
# ----------------------------------------------------------------------

def test_profile_exists(pm, tmpdir_profiles):
    path = tmpdir_profiles / "alpha_profile.json"
    path.write_text("{}")
    assert pm.profile_exists("alpha")
    assert not pm.profile_exists("beta")


# ----------------------------------------------------------------------
# save_profile
# ----------------------------------------------------------------------

def test_save_profile_writes_json_and_sets_active(pm, tmpdir_profiles, analyzer):
    pm.save_profile("alpha", {"f1": 100})

    json_path = tmpdir_profiles / "alpha_profile.json"
    assert json_path.exists()

    data = json.loads(json_path.read_text())
    assert data["voice_type"] == analyzer.voice_type

    active_path = tmpdir_profiles / "active_profile.json"
    assert active_path.exists()


def test_save_profile_model_bytes(pm, tmpdir_profiles):
    pm.save_profile("alpha", {"voice_type": "bass"}, model_bytes=b"XYZ")

    model_path = tmpdir_profiles / "alpha_model.pkl"
    assert model_path.exists()
    assert model_path.read_bytes() == b"XYZ"


# ----------------------------------------------------------------------
# load_profile_json
# ----------------------------------------------------------------------

def test_load_profile_json_path_object(pm, tmpdir_profiles):
    p = tmpdir_profiles / "custom.json"
    p.write_text('{"x": 1}')
    out = pm.load_profile_json(p)
    assert out == {"x": 1}


def test_load_profile_json_missing(pm):
    out = pm.load_profile_json("missing")
    assert out == {}  # missing → {}


def test_load_profile_json_malformed(pm, tmpdir_profiles):
    p = tmpdir_profiles / "bad_profile.json"
    p.write_text("{not json")
    out = pm.load_profile_json("bad")
    assert out == {}  # malformed → {}


def test_load_profile_json_active_file_open(pm, tmpdir_profiles):
    # Create profile
    (tmpdir_profiles / "alpha_profile.json").write_text("{}")

    # Create active file
    (tmpdir_profiles / "active_profile.json").write_text("active")

    # Should open both files without error
    out = pm.load_profile_json("alpha")
    assert isinstance(out, dict)


# ----------------------------------------------------------------------
# extract_formants
# ----------------------------------------------------------------------

def test_extract_formants_basic(pm):
    raw = {
        "voice_type": "bass",
        "a": {"f1": 500, "f2": 1500, "f0": 100, "confidence": 0.9, "stability": 5},
        "b": {"f1": 400, "f2": 1200, "f0": 90},
        "junk": 123,
    }

    out = pm.extract_formants(raw)

    assert out["a"] == {
        "f1": 500,
        "f2": 1500,
        "f0": 100,
        "confidence": 0.9,
        "stability": 5,
    }

    assert out["b"] == {
        "f1": 400,
        "f2": 1200,
        "f0": 90,
        "confidence": 0.0,
        "stability": float("inf"),
    }

    assert "junk" not in out

# ----------------------------------------------------------------------
# apply_profile
# ----------------------------------------------------------------------


def test_apply_profile(pm, tmpdir_profiles, analyzer):
    # Create profile JSON
    (tmpdir_profiles / "alpha_profile.json").write_text(
        json.dumps({
            "calibrated_vowels": {"a": {"f1": 500, "f2": 1500, "f0": 100}},
            "interpolated_vowels": {}
        })
    )

    pm.apply_profile("alpha")

    # Voice type applied
    assert analyzer.voice_type == "tenor"

    # Modern engine stores dict-based formants
    assert analyzer.user_formants["a"]["f1"] == 500
    assert analyzer.user_formants["a"]["f2"] == 1500
    assert analyzer.user_formants["a"]["f0"] == 100

    # Engine reset
    assert analyzer.set_user_formants_called

    # Active profile updated
    active_path = tmpdir_profiles / "active_profile.json"
    assert active_path.exists()


# ----------------------------------------------------------------------
# delete_profile
# ----------------------------------------------------------------------

def test_delete_profile(pm, tmpdir_profiles):
    # Create files
    (tmpdir_profiles / "alpha_profile.json").write_text("{}")
    (tmpdir_profiles / "alpha_model.pkl").write_bytes(b"X")
    (tmpdir_profiles / "active_profile.json").write_text("{}")

    pm.active_profile_name = "alpha"
    pm.delete_profile("alpha")

    assert not (tmpdir_profiles / "alpha_profile.json").exists()
    assert not (tmpdir_profiles / "alpha_model.pkl").exists()
    assert not (tmpdir_profiles / "active_profile.json").exists()
    assert pm.active_profile_name is None
