import numpy as np
import pytest
from analysis.vowel_data import FORMANTS
from analysis.vowel import (
    get_vowel_ranges,
    is_plausible_formants,
    is_plausible_pitch,
    guess_vowel,
    robust_guess,
)


# ----------------------------------------------------------------------
# get_vowel_ranges
# ----------------------------------------------------------------------

def test_get_vowel_ranges_calibrated_standard():
    calibrated = {"a": {"f1": 500, "f2": 1500}}
    lo1, hi1, lo2, hi2 = get_vowel_ranges("tenor", "a", calibrated)
    assert lo1 == pytest.approx(500 * 0.7)
    assert hi1 == pytest.approx(500 * 1.3)


def test_get_vowel_ranges_calibrated_back_vowel():
    calibrated = {"u": {"f1": 300, "f2": 900}}
    lo1, hi1, lo2, hi2 = get_vowel_ranges("tenor", "u", calibrated)
    assert lo2 == pytest.approx(900 * 0.5)


def test_get_vowel_ranges_reference_fallback():
    lo1, hi1, lo2, hi2 = get_vowel_ranges("unknown", "ɑ")
    assert lo1 < hi1
    assert lo2 < hi2


def test_get_vowel_ranges_invalid_vowel():
    assert get_vowel_ranges("tenor", "zzz") is None


# ----------------------------------------------------------------------
# is_plausible_formants
# ----------------------------------------------------------------------

def test_plausible_missing():
    ok, reason = is_plausible_formants(None, 500)
    assert not ok
    assert reason == "missing formant"


def test_plausible_nan():
    ok, reason = is_plausible_formants(np.nan, 500)
    assert not ok
    assert reason == "nan formant"


def test_plausible_swapped():
    ok, reason = is_plausible_formants(800, 500)
    assert not ok
    assert reason == "swapped"


def test_plausible_too_low():
    ok, reason = is_plausible_formants(100, 200)
    assert not ok
    assert reason == "too low"


def test_plausible_no_vowel_ok():
    ok, reason = is_plausible_formants(500, 1500, vowel=None)
    assert ok
    assert reason == "ok"


def test_plausible_standard_out_of_range():
    # Force out-of-range by using extreme values
    ok, reason = is_plausible_formants(2000, 5000, vowel="ɑ")
    assert not ok
    assert "f1 out of range" in reason or "f2 out of range" in reason


def test_plausible_back_vowel_f2_out():
    ok, reason = is_plausible_formants(400, 5000, vowel="u")
    assert not ok
    assert reason == "f2-out-of-range"


def test_plausible_back_vowel_f1_drift():
    # Get the reference F2 for "u" in the tenor table
    f2_ref = FORMANTS["tenor"]["u"][1]

    # Pick an F2 safely inside the allowed ±50% range
    f2 = f2_ref

    # Pick an F1 that is >=120 but outside the allowed range
    f1 = 150  # guaranteed >=120

    ok, reason = is_plausible_formants(f1, f2, vowel="u")
    assert ok
    assert reason == "f1-drift"

# ----------------------------------------------------------------------
# is_plausible_pitch
# ----------------------------------------------------------------------

def test_plausible_pitch_missing():
    ok, reason = is_plausible_pitch(None)
    assert not ok
    assert reason == "missing pitch"


def test_plausible_pitch_nan():
    ok, reason = is_plausible_pitch(np.nan)
    assert not ok
    assert reason == "missing pitch"


def test_plausible_pitch_out_of_range():
    ok, reason = is_plausible_pitch(20)
    assert not ok
    assert "f0 out of range" in reason


def test_plausible_pitch_ok():
    ok, reason = is_plausible_pitch(200)
    assert ok
    assert reason == "ok"


# ----------------------------------------------------------------------
# guess_vowel
# ----------------------------------------------------------------------

def test_guess_vowel_missing():
    assert guess_vowel(None, 500, last_guess="a") == "a"


def test_guess_vowel_nan():
    assert guess_vowel(np.nan, 500, last_guess="a") == "a"


def test_guess_vowel_too_close():
    assert guess_vowel(500, 800, last_guess="a") == "a"


def test_guess_vowel_basic():
    # Should pick the closest vowel in FORMANTS["bass"]
    v = guess_vowel(500, 1500, voice_type="bass")
    assert isinstance(v, str)


# ----------------------------------------------------------------------
# robust_guess
# ----------------------------------------------------------------------

def test_robust_guess_not_enough_valid():
    vowel, conf, second = robust_guess([None, np.nan])
    assert vowel is None
    assert conf == 0.0
    assert second is None


def test_robust_guess_basic():
    vowel, conf, second = robust_guess([500, 1500], voice_type="bass")
    assert isinstance(vowel, str)
    assert isinstance(second, str)
    assert conf > 0
