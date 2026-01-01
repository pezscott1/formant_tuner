from unittest.mock import patch
from calibration.session import CalibrationSession, normalize_profile_for_save

# ---------------------------------------------------------
# CalibrationSession.handle_result
# ---------------------------------------------------------


def test_handle_result_accepts_good_capture():
    sess = CalibrationSession("my_profile", "bass", ["a", "i"])

    accepted, skipped, msg = sess.handle_result(
        "a", 500.0, 1500.0, 120.0, confidence=0.8, stability=1e4
    )

    assert accepted is True
    assert skipped is False
    assert "/a/" in msg

    entry = sess.data["a"]
    assert entry["f1"] == 500.0
    assert entry["f2"] == 1500.0
    assert entry["f0"] == 120.0
    assert entry["confidence"] == 0.8
    assert entry["stability"] == 1e4
    assert entry["weight"] == 1.0


def test_handle_result_accepts_unknown_vowel():

    sess = CalibrationSession("my_profile", "bass", ["a"])

    accepted, skipped, msg = sess.handle_result(
        "i", 500.0, 1500.0, 120.0, confidence=0.8, stability=1e4
    )

    assert accepted is True


# ---------------------------------------------------------
# normalize_profile_for_save
# ---------------------------------------------------------


@patch("calibration.session.is_plausible_formants", return_value=(True, "ok"))
def test_normalize_profile_for_save_basic(mock_plausible):
    user_formants = {
        "a": {
            "f1": 700.0,
            "f2": 1100.0,
            "f0": 120.0,
            "confidence": 0.8,
            "stability": 1e4,
        },
        "i": {
            "f1": 300.0,
            "f2": 2500.0,
            "f0": 130.0,
            "confidence": 0.9,
            "stability": 5e3,
        },
    }
    retries_map = {"a": 1, "i": 2}

    out = normalize_profile_for_save(user_formants, retries_map=retries_map)

    assert "a" in out and "i" in out
    a = out["a"]
    assert a["f1"] == 700.0
    assert a["f2"] == 1100.0
    assert a["f0"] == 120.0
    assert a["confidence"] == 0.8
    assert a["stability"] == 1e4
    assert a["retries"] == 1
    assert a["reason"] == "ok"
    assert a["source"] == "calibration"
    assert "saved_at" in a


@patch("calibration.session.is_plausible_formants", return_value=(True, "ok"))
def test_normalize_profile_for_save_swaps_reversed_f1_f2(mock_plausible):
    user_formants = {
        "a": {
            "f1": 1500.0,
            "f2": 700.0,
            "f0": 120.0,
            "confidence": 0.8,
            "stability": 1e4,
        }
    }

    out = normalize_profile_for_save(user_formants, retries_map={})
    a = out["a"]

    # f1/f2 should be swapped in the normalized output
    assert a["f1"] == 700.0
    assert a["f2"] == 1500.0


class DummyProfileManager:
    def __init__(self):
        self.calls = []

    def save_profile(self, base, data, model_bytes=None):
        self.calls.append((base, data, model_bytes))
        return base


def test_calibration_session_save_profile_creates_new_file():
    sess = CalibrationSession("my_profile", "bass", ["a"])
    sess.data["a"] = {
        "f1": 700.0,
        "f2": 1100.0,
        "f0": 120.0,
        "confidence": 0.8,
        "stability": 1e4,
        "weight": 1.0,
        "saved_at": "2025-01-01T00:00:00Z",
    }

    dummy_pm = DummyProfileManager()
    sess.profile_manager = dummy_pm

    base = sess.save_profile()
    assert base == "my_profile_bass"

    assert len(dummy_pm.calls) == 1
    call_base, call_data, call_model = dummy_pm.calls[0]
    assert call_base == base
    assert call_data["voice_type"] == "bass"
    assert "a" in call_data
