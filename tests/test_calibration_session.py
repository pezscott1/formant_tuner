import json
from unittest.mock import patch, mock_open
from calibration.session import (
    CalibrationSession,
    merge_formants,
    normalize_profile_for_save,
    profile_path,
)


# ---------------------------------------------------------
# merge_formants
# ---------------------------------------------------------
@patch("calibration.session.is_plausible_formants", return_value=(True, "ok"))
def test_merge_formants_prefers_higher_confidence(mock_plausible):
    old_vals = (500.0, 1500.0, 120.0, 0.6, 1e4)
    new_vals = (510.0, 1510.0, 125.0, 0.8, 1e4)

    out = merge_formants(old_vals, new_vals, "a")
    assert out == new_vals


@patch("calibration.session.is_plausible_formants")
def test_merge_formants_rejects_implausible_new(mock_plausible):
    # new implausible, old plausible
    mock_plausible.side_effect = [
             (False, "bad"),  # new checked first
             (True, "ok"),  # old checked second
    ]

    old_vals = (500.0, 1500.0, 120.0, 0.6, 1e4)
    new_vals = (2000.0, 2100.0, 100.0, 0.9, 1e3)

    out = merge_formants(old_vals, new_vals, "a")
    assert out == old_vals


@patch("calibration.session.is_plausible_formants")
def test_merge_formants_rejects_implausible_old(mock_plausible):
    # old implausible, new plausible
    mock_plausible.side_effect = [
             (True, "ok"),  # new checked first
             (False, "bad"),  # old checked second
    ]

    old_vals = (2000.0, 2100.0, 100.0, 0.9, 1e3)
    new_vals = (500.0, 1500.0, 120.0, 0.6, 1e4)

    out = merge_formants(old_vals, new_vals, "a")
    assert out == new_vals


# ---------------------------------------------------------
# CalibrationSession.handle_result
# ---------------------------------------------------------
def test_handle_result_accepts_good_capture_and_advances_index():
    sess = CalibrationSession("my_profile", "bass", ["a", "i"])
    accepted, skipped, msg = sess.handle_result(
        f1=500.0,
        f2=1500.0,
        f0=120.0,
        confidence=0.8,
        stability=1e4,
    )

    assert accepted is True
    assert skipped is False
    assert "/a/" in msg
    assert sess.results["a"] == (500.0, 1500.0, 120.0, 0.8, 1e4)
    assert sess.current_vowel == "i"


def test_handle_result_retries_then_skips_after_max_retries():
    sess = CalibrationSession("my_profile", "bass", ["a"])

    # Bad captures: low confidence, high stability, or missing formants
    for i in range(sess.max_retries):
        accepted, skipped, msg = sess.handle_result(
            f1=None,
            f2=None,
            f0=None,
            confidence=0.1,
            stability=1e6,
        )
        assert accepted is False
        assert skipped is False
        assert "retry" in msg

    # Next bad capture → skip
    accepted, skipped, msg = sess.handle_result(
        f1=None,
        f2=None,
        f0=None,
        confidence=0.1,
        stability=1e6,
    )
    assert accepted is False
    assert skipped is True
    assert "skipped" in msg
    assert sess.is_complete()


def test_handle_result_requires_active_vowel():
    sess = CalibrationSession("my_profile", "bass", ["a"])
    sess.current_index = 1  # move past last vowel

    accepted, skipped, msg = sess.handle_result(
        f1=500.0, f2=1500.0, f0=120.0, confidence=0.8, stability=1e4
    )
    assert accepted is False
    assert skipped is False
    assert "No vowel active" in msg


# ---------------------------------------------------------
# normalize_profile_for_save
# ---------------------------------------------------------
@patch("calibration.session.is_plausible_formants", return_value=(True, "ok"))
def test_normalize_profile_for_save_basic(mock_plausible):
    user_formants = {
        "a": (700.0, 1100.0, 120.0, 0.8, 1e4),
        "i": (300.0, 2500.0, 130.0, 0.9, 5e3),
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
        "a": (1500.0, 700.0, 120.0, 0.8, 1e4),
    }

    out = normalize_profile_for_save(user_formants, retries_map={})
    a = out["a"]
    # f1 and f2 should be swapped
    assert a["f1"] == 700.0
    assert a["f2"] == 1500.0


# ---------------------------------------------------------
# save_profile (I/O behavior)
# ---------------------------------------------------------
@patch("calibration.session.os.path.exists", return_value=False)
@patch("calibration.session.open", new_callable=mock_open)
@patch("calibration.session.normalize_profile_for_save")
def test_calibration_session_save_profile_creates_new_file(
    mock_norm, mock_open_file, mock_exists
):
    mock_norm.return_value = {
        "a": {
            "f1": 700.0,
            "f2": 1100.0,
            "f0": 120.0,
            "confidence": 0.8,
            "stability": 1e4,
            "retries": 1,
            "reason": "ok",
            "saved_at": "2025-01-01T00:00:00Z",
            "source": "calibration",
        }
    }

    sess = CalibrationSession("my_profile", "bass", ["a"])
    sess.results["a"] = (700.0, 1100.0, 120.0, 0.8, 1e4)

    base = sess.save_profile()
    assert base == "my_profile_bass"

    expected_path = profile_path("my_profile_bass")
    mock_open_file.assert_called_once_with(expected_path, "w", encoding="utf-8")

    # The saved JSON should include voice_type
    handle = mock_open_file()
    written = "".join(call.args[0] for call in handle.write.mock_calls)
    data = json.loads(written)
    assert data["voice_type"] == "bass"
