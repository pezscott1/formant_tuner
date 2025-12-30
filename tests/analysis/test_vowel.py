import numpy as np
import pytest

from analysis.vowel_data import VOWEL_CENTERS
from analysis.plausibility import (
    vowel_window,
    is_plausible_formants,
    is_plausible_pitch,
)


# ----------------------------------------------------------------------
# vowel_window
# ----------------------------------------------------------------------

def test_vowel_window_reference():
    lo1, hi1, lo2, hi2 = vowel_window("tenor", "ɑ")
    base_f1, base_f2 = VOWEL_CENTERS["tenor"]["ɑ"][:2]

    assert lo1 < base_f1 < hi1
    assert lo2 < base_f2 < hi2


def test_vowel_window_calibrated_valid():
    calibrated = {"a": {"f1": 500, "f2": 1500, "f0": 200}}
    lo1, hi1, lo2, hi2 = vowel_window("tenor", "a", calibrated)

    assert lo1 == pytest.approx(500 * 0.70)
    assert hi1 == pytest.approx(500 * 1.30)
    assert lo2 == pytest.approx(1500 * 0.80)
    assert hi2 == pytest.approx(1500 * 1.20)


def test_vowel_window_calibrated_invalid():
    calibrated = {"a": {"f1": 500, "f2": 1500, "f0": None}}  # invalid f0

    lo1, hi1, lo2, hi2 = vowel_window("tenor", "a", calibrated)

    # Just assert we got a usable window.
    assert lo1 < hi1
    assert lo2 < hi2
    assert lo1 > 0
    assert lo2 > 0


def test_vowel_window_invalid_vowel():
    assert vowel_window("tenor", "zzz") is None


def test_vowel_window_voice_fallback():
    r1 = vowel_window("alien", "ɑ")
    r2 = vowel_window("tenor", "ɑ")
    assert r1 == r2


# ----------------------------------------------------------------------
# is_plausible_formants
# ----------------------------------------------------------------------

def test_plausible_missing():
    ok, reason = is_plausible_formants(None, 500)
    assert not ok
    assert reason == "missing"


def test_plausible_nan():
    ok, reason = is_plausible_formants(np.nan, 500)
    assert not ok
    assert reason == "nan"


def test_plausible_swapped():
    ok, reason = is_plausible_formants(800, 500)
    assert not ok
    assert reason == "swapped"


def test_plausible_too_low():
    ok, reason = is_plausible_formants(100, 200)
    assert not ok
    assert reason == "too-low"


def test_plausible_no_vowel_ok():
    ok, reason = is_plausible_formants(500, 1500, vowel=None)
    assert ok
    assert reason == "ok"


def test_plausible_standard_out_of_range():
    ok, reason = is_plausible_formants(2000, 5000, vowel="ɑ")
    assert not ok
    assert "out" in reason


def test_plausible_back_vowel_f2_out():
    ok, reason = is_plausible_formants(400, 5000, vowel="u")
    assert not ok
    assert reason.startswith("f2-out")


def test_plausible_back_vowel_f1_drift():
    f2_ref = VOWEL_CENTERS["tenor"]["u"][1]
    ok, reason = is_plausible_formants(150, f2_ref, vowel="u")
    assert not ok
    # Reason should indicate some out-of-range condition
    assert "out" in reason


# ----------------------------------------------------------------------
# is_plausible_pitch
# ----------------------------------------------------------------------

def test_plausible_pitch_missing():
    ok, reason = is_plausible_pitch(None)
    assert not ok
    assert reason == "missing pitch"

    ok, reason = is_plausible_pitch(np.nan)
    assert not ok
    assert reason == "missing pitch"


def test_plausible_pitch_out_of_range():
    ok, reason = is_plausible_pitch(20)
    assert not ok
    assert "out-of-range" in reason


def test_plausible_pitch_ok():
    ok, reason = is_plausible_pitch(200)
    assert ok
    assert reason == "ok"
