import numpy as np
from calibration.session import (
    CalibrationSession,
    normalize_profile_for_save,
)


# -----------------------------
# Helpers
# -----------------------------

class DummyPM:
    def __init__(self):
        self.saved = None

    def save_profile(self, name, data):
        self.saved = (name, data)


def always_plausible(f1, f2, **kwargs):
    return True, "ok"


def never_plausible(f1, f2, **kwargs):
    return False, "bad"


# -----------------------------
# Constructor + existing profile
# -----------------------------

def test_constructor_loads_existing_profile():
    existing = {
        "a": {"f1": 500, "f2": 1500},
        "voice_type": "bass",
    }
    sess = CalibrationSession("p", "bass", ["a"], existing_profile=existing)
    assert "a" in sess.data
    assert sess.data["a"]["f1"] == 500


# -----------------------------
# _weights_for
# -----------------------------

def test_weights_for_defaults():
    sess = CalibrationSession("p", "bass", [])
    assert sess._weights_for("zzz") == (1/3, 1/3, 1/3)


# -----------------------------
# get_calibrated_anchors
# -----------------------------

def test_get_calibrated_anchors():
    sess = CalibrationSession("p", "bass", [])
    sess.data = {
        "a": {"f1": 500, "f2": 1500},
        "b": {"f1": None, "f2": 1200},
    }
    sess.calibrated_vowels.add("a")

    anchors = sess.get_calibrated_anchors()
    assert anchors == {"a": (500.0, 1500.0)}


# -----------------------------
# barycentric_interpolate
# -----------------------------

def test_barycentric_interpolate():
    tri = {"A": (0, 0), "B": (10, 0), "C": (0, 10)}
    out = CalibrationSession.barycentric_interpolate((0.5, 0.5, 0.0), tri)
    assert np.allclose(out, [5, 0])


# -----------------------------
# compute_interpolated_vowels
# -----------------------------

def test_compute_interpolated_vowels_basic(monkeypatch):
    monkeypatch.setattr("analysis.plausibility.is_plausible_formants", always_plausible)

    sess = CalibrationSession("p", "bass", [])
    sess.data = {
        "ɛ": {"f1": 500, "f2": 1500, "f0": 100},
        "ɑ": {"f1": 600, "f2": 1100, "f0": 120},
        "i": {"f1": 300, "f2": 2500, "f0": 140},
    }
    sess.calibrated_vowels.update({"ɛ", "ɑ", "i"})
    out = sess.compute_interpolated_vowels()
    assert "æ" in out
    assert "f1" in out["æ"]
    assert "f2" in out["æ"]
    assert "f0" in out["æ"]


# -----------------------------
# increment_retry
# -----------------------------

def test_increment_retry():
    sess = CalibrationSession("p", "bass", [])
    sess.increment_retry("a")
    sess.increment_retry("a")
    assert sess.retries_map["a"] == 2


# -----------------------------
# handle_result rejection paths
# -----------------------------

def test_handle_result_invalid_formants():
    sess = CalibrationSession("p", "bass", [])
    ok, rej, msg = sess.handle_result("a", None, 1500, 100, 0.5, 0.0)
    assert not ok and rej


def test_handle_result_plausibility_reject(monkeypatch):
    monkeypatch.setattr("calibration.session.is_plausible_formants", never_plausible)
    sess = CalibrationSession("p", "bass", [])
    ok, rej, msg = sess.handle_result("a", 500, 1500, 100, 0.5, 0.0)
    assert not ok and rej


def test_handle_result_low_confidence(monkeypatch):
    monkeypatch.setattr("analysis.plausibility.is_plausible_formants", always_plausible)
    sess = CalibrationSession("p", "bass", [])
    ok, rej, msg = sess.handle_result("a", 500, 1500, 100, 0.1, 0.0)
    assert not ok and rej


# -----------------------------
# handle_result first measurement
# -----------------------------

def test_handle_result_first_measurement(monkeypatch):
    monkeypatch.setattr("analysis.plausibility.is_plausible_formants", always_plausible)
    sess = CalibrationSession("p", "bass", [])
    ok, rej, msg = sess.handle_result("a", 500, 1500, 100, 0.5, 0.0)
    assert ok and not rej
    assert "a" in sess.data


# -----------------------------
# handle_result weighted averaging
# -----------------------------

def test_handle_result_weighted_average(monkeypatch):
    monkeypatch.setattr("calibration.session.is_plausible_formants", always_plausible)
    sess = CalibrationSession("p", "bass", [])
    sess.data["a"] = {
        "f1": 500,
        "f2": 1500,
        "f0": 100,
        "confidence": 0.5,
        "stability": 0.1,
        "weight": 1.0,
        "saved_at": "x",
    }

    ok, rej, msg = sess.handle_result("a", 700, 1700, 200, 0.9, 0.3)
    assert ok and not rej
    updated = sess.data["a"]
    assert updated["weight"] == 2.0
    assert 500 < updated["f1"] < 700


# -----------------------------
# save_profile
# -----------------------------

def test_save_profile(monkeypatch):
    monkeypatch.setattr("analysis.plausibility.is_plausible_formants", always_plausible)

    pm = DummyPM()
    sess = CalibrationSession("myprof", "bass", [], profile_manager=pm)
    sess.data["a"] = {"f1": 500, "f2": 1500, "f0": 100,
                      "confidence": 1.0, "stability": 0.0,
                      "weight": 1.0, "saved_at": "x"}

    name = sess.save_profile()
    assert name == "myprof_bass"
    assert pm.saved is not None
    saved_name, saved_data = pm.saved
    assert saved_name == "myprof_bass"

    # NEW: vowel entries live inside calibrated_vowels
    assert "a" in saved_data["calibrated_vowels"]

    assert saved_data["voice_type"] == "bass"


# -----------------------------
# normalize_profile_for_save
# -----------------------------

def test_normalize_profile_for_save(monkeypatch):
    monkeypatch.setattr("analysis.plausibility.is_plausible_formants", always_plausible)

    user_formants = {
        "a": {"f1": 600, "f2": 500, "f0": 100, "confidence": 0.8, "stability": 0.2},
    }
    retries = {"a": 3}

    out = normalize_profile_for_save(user_formants, retries)
    assert out["a"]["f1"] == 500
    assert out["a"]["f2"] == 600
    assert out["a"]["retries"] == 3
    assert out["a"]["reason"].startswith("f2-out")
