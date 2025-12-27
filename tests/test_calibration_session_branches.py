import pytest
from calibration.session import CalibrationSession, normalize_profile_for_save


# ----------------------------------------------------------------------
# 1. Existing profile loading
# ----------------------------------------------------------------------

def test_load_existing_profile():
    existing = {
        "a": {"f1": 500, "f2": 1500, "f0": 120, "confidence": 0.8, "stability": 0.9},
        "voice_type": "baritone",
    }
    sess = CalibrationSession(
        "test", "baritone", vowels=["a"], existing_profile=existing)
    assert "a" in sess.data
    assert sess.data["a"]["f1"] == 500


# ----------------------------------------------------------------------
# 2. increment_retry
# ----------------------------------------------------------------------

def test_increment_retry():
    sess = CalibrationSession("p", "t", vowels=["a"])
    sess.increment_retry("a")
    sess.increment_retry("a")
    assert sess.retries_map["a"] == 2


# ----------------------------------------------------------------------
# 3. handle_result — first measurement
# ----------------------------------------------------------------------

def test_handle_result_first_measurement():
    sess = CalibrationSession("p", "t", vowels=["a"])
    ok, retry, msg = sess.handle_result("a", 500, 1500, 120, 0.8, 0.9)
    assert ok is True
    assert retry is False
    assert "Accepted first measurement" in msg
    assert sess.data["a"]["weight"] == 1.0


# ----------------------------------------------------------------------
# 4. handle_result — weighted update with legacy fields missing
# ----------------------------------------------------------------------

def test_handle_result_weighted_update():
    sess = CalibrationSession("p", "t", vowels=["a"])
    sess.data["a"] = {
        "f1": 400,
        "f2": 1400,
        "f0": None,
        "confidence": 1.0,
        "stability": 0.0,
        "weight": 1.0,
    }

    ok, retry, msg = sess.handle_result("a", 600, 1600, 200, 0.5, 0.5)
    assert ok is True
    assert retry is False
    assert "Updated /a/" in msg

    updated = sess.data["a"]
    assert updated["weight"] == 2.0
    assert updated["f1"] == pytest.approx((400 + 600) / 2)
    assert updated["f2"] == pytest.approx((1400 + 1600) / 2)
    assert updated["f0"] == 200  # old was None → new takes f0


# ----------------------------------------------------------------------
# 5. handle_result — f0 smoothing when both old and new are present
# ----------------------------------------------------------------------

def test_handle_result_f0_smoothing():
    sess = CalibrationSession("p", "t", vowels=["a"])
    sess.data["a"] = {
        "f1": 400,
        "f2": 1400,
        "f0": 100,
        "confidence": 1.0,
        "stability": 0.0,
        "weight": 1.0,
    }

    sess.handle_result("a", 400, 1400, 200, 0.5, 0.5)
    updated = sess.data["a"]
    assert updated["f0"] == pytest.approx((100 + 200) / 2)


# ----------------------------------------------------------------------
# 6. save_profile — missing profile_manager
# ----------------------------------------------------------------------

def test_save_profile_missing_manager():
    sess = CalibrationSession("p", "t", vowels=["a"])
    with pytest.raises(RuntimeError):
        sess.save_profile()


# ----------------------------------------------------------------------
# 7. save_profile — correct naming and save call
# ----------------------------------------------------------------------

class DummyManager:
    def __init__(self):
        self.saved = None

    def save_profile(self, name, data):
        self.saved = (name, data)


def test_save_profile_calls_manager():
    mgr = DummyManager()
    sess = CalibrationSession("profile", "tenor", vowels=["a"], profile_manager=mgr)
    sess.data["a"] = {
        "f1": 500,
        "f2": 1500,
        "f0": 120,
        "confidence": 0.8,
        "stability": 0.9,
        "weight": 1.0,
        "saved_at": "2025-01-01T00:00:00Z",
    }

    name = sess.save_profile()
    assert name == "profile_tenor"
    assert mgr.saved[0] == "profile_tenor"
    assert "a" in mgr.saved[1]


# ----------------------------------------------------------------------
# 8. normalize_profile_for_save — reversed f1/f2 swap
# ----------------------------------------------------------------------

def test_normalize_swap_reversed():
    user = {
        "a": {
            "f1": 900,
            "f2": 500,
            "f0": 120,
            "confidence": 0.8,
            "stability": 0.9,
            "weight": 1.0,
            "saved_at": "2025-01-01T00:00:00Z",
        }
    }
    out = normalize_profile_for_save(user, retries_map={})
    assert out["a"]["f1"] == 500
    assert out["a"]["f2"] == 900


# ----------------------------------------------------------------------
# 9. normalize_profile_for_save — plausible vs implausible
# ----------------------------------------------------------------------

def test_normalize_plausible_and_implausible(monkeypatch):
    # Force is_plausible_formants to return False
    monkeypatch.setattr(
        "calibration.session.is_plausible_formants",
        lambda f1, f2, vowel=None: (False, "bad"),
    )

    user = {
        "a": {
            "f1": 500,
            "f2": 1500,
            "f0": 120,
            "confidence": 0.8,
            "stability": 0.9,
            "weight": 1.0,
            "saved_at": "2025-01-01T00:00:00Z",
        }
    }

    out = normalize_profile_for_save(user, retries_map={"a": 3})
    assert out["a"]["reason"] == "bad"
    assert out["a"]["retries"] == 3


# ----------------------------------------------------------------------
# 10. normalize_profile_for_save — None values skipped
# ----------------------------------------------------------------------

def test_normalize_skips_none_entries():
    user = {
        "a": None,
        "e": {
            "f1": 400,
            "f2": 2000,
            "f0": 120,
            "confidence": 0.8,
            "stability": 0.9,
            "weight": 1.0,
            "saved_at": "2025-01-01T00:00:00Z",
        },
    }
    out = normalize_profile_for_save(user, retries_map={})
    assert "a" not in out
    assert "e" in out
