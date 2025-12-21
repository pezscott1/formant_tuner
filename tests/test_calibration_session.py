import json
import math
from calibration.session import (
    CalibrationSession,
    normalize_profile_for_save,
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def test_current_vowel_and_completion():
    sess = CalibrationSession("scott", "tenor", ["a", "e"])
    assert sess.current_vowel == "a"
    assert not sess.is_complete()

    sess.current_index = 1
    assert sess.current_vowel == "e"
    assert not sess.is_complete()

    sess.current_index = 2
    assert sess.current_vowel is None
    assert sess.is_complete()


# ---------------------------------------------------------
# handle_result
# ---------------------------------------------------------

def test_handle_result_accepts_valid_formants():
    sess = CalibrationSession("scott", "tenor", ["a"])
    accepted, skipped, msg = sess.handle_result(500, 1500, 120)

    assert accepted is True
    assert skipped is False
    assert "/a/ accepted" in msg
    assert sess.results["a"] == (500.0, 1500.0, 120.0)
    assert sess.current_index == 1


def test_handle_result_accepts_missing_pitch():
    sess = CalibrationSession("scott", "tenor", ["a"])
    accepted, skipped, msg = sess.handle_result(500, 1500, None)

    assert accepted is True
    assert skipped is False
    assert sess.results["a"] == (500.0, 1500.0, None)


def test_handle_result_nan_rejected_and_retried():
    sess = CalibrationSession("scott", "tenor", ["a"])
    accepted, skipped, msg = sess.handle_result(math.nan, 1500, 120)

    assert accepted is False
    assert skipped is False
    assert "retry" in msg
    assert sess.retries_map["a"] == 1
    assert sess.current_index == 0


def test_handle_result_retry_then_skip():
    sess = CalibrationSession("scott", "tenor", ["a"])
    # Force 3 retries
    for _ in range(3):
        accepted, skipped, msg = sess.handle_result(None, None, None)
        assert accepted is False
        assert skipped is False

    # 4th attempt → skip
    accepted, skipped, msg = sess.handle_result(None, None, None)
    assert accepted is False
    assert skipped is True
    assert "skipped" in msg
    assert sess.current_index == 1


def test_handle_result_no_active_vowel():
    sess = CalibrationSession("scott", "tenor", ["a"])
    sess.current_index = 1  # past end
    accepted, skipped, msg = sess.handle_result(500, 1500, 120)

    assert accepted is False
    assert skipped is False
    assert "No vowel active" in msg


# ---------------------------------------------------------
# normalize_profile_for_save
# ---------------------------------------------------------

def test_normalize_profile_for_save_basic():
    user_formants = {"a": (500, 1500, 120)}
    retries = {"a": 2}

    out = normalize_profile_for_save(user_formants, retries)
    entry = out["a"]

    assert entry["f1"] == 500.0
    assert entry["f2"] == 1500.0
    assert entry["f0"] == 120.0
    assert entry["retries"] == 2
    assert entry["reason"] == "ok"
    assert entry["source"] == "calibration"
    assert "saved_at" in entry


def test_normalize_profile_for_save_swapped_formants():
    user_formants = {"a": (1500, 500, 120)}  # swapped
    out = normalize_profile_for_save(user_formants)

    assert out["a"]["f1"] == 500.0
    assert out["a"]["f2"] == 1500.0


def test_normalize_profile_for_save_invalid_input():
    out = normalize_profile_for_save("not a dict")
    assert out == {}


def test_normalize_profile_for_save_dict_input():
    user_formants = {"a": {"f1": 500, "f2": 1500, "f0": 120}}
    out = normalize_profile_for_save(user_formants)

    assert out["a"]["f1"] == 500.0
    assert out["a"]["f2"] == 1500.0
    assert out["a"]["f0"] == 120.0


# ---------------------------------------------------------
# save_profile
# ---------------------------------------------------------

def test_save_profile(tmp_path, monkeypatch):
    # Redirect PROFILES_DIR to a temp directory
    monkeypatch.setattr("calibration.session.PROFILES_DIR", tmp_path)

    sess = CalibrationSession("scott", "tenor", ["a"])
    sess.results = {"a": (500.0, 1500.0, 120.0)}

    base_name = sess.save_profile()
    assert base_name == "scott_tenor"

    path = tmp_path / "scott_tenor_profile.json"
    assert path.exists()

    data = json.loads(path.read_text())
    assert "a" in data
    assert data["a"]["f1"] == 500.0
    assert data["a"]["f2"] == 1500.0
    assert data["a"]["f0"] == 120.0
