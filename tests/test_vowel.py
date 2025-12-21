import numpy as np
from analysis.vowel import (
    get_vowel_ranges,
    is_plausible_formants,
    is_plausible_pitch,
    guess_vowel,
    robust_guess,
    get_expected_formants,
)
from analysis.vowel_data import FORMANTS, VOWEL_MAP, PITCH_RANGES


# ---------------------------------------------------------
# get_vowel_ranges
# ---------------------------------------------------------

def test_get_vowel_ranges_valid():
    f1_low, f1_high, f2_low, f2_high = get_vowel_ranges("tenor", "a")
    base_f1, base_f2 = FORMANTS["tenor"]["a"][:2]
    assert f1_low < base_f1 < f1_high
    assert f2_low < base_f2 < f2_high


def test_get_vowel_ranges_invalid_vowel():
    assert get_vowel_ranges("tenor", "zzz") is None


def test_get_vowel_ranges_fallback_voice_type():
    # Unknown voice type → fallback to tenor
    r1 = get_vowel_ranges("alien", "a")
    r2 = get_vowel_ranges("tenor", "a")
    assert r1 == r2


# ---------------------------------------------------------
# is_plausible_formants
# ---------------------------------------------------------

def test_is_plausible_formants_missing():
    ok, reason = is_plausible_formants(None, 500)
    assert not ok and reason == "missing formant"


def test_is_plausible_formants_swapped():
    ok, reason = is_plausible_formants(1500, 500)
    assert not ok and "swapped" in reason


def test_is_plausible_formants_too_low():
    ok, reason = is_plausible_formants(100, 300)
    assert not ok and "too low" in reason


def test_is_plausible_formants_out_of_range():
    base_f1, base_f2 = FORMANTS["tenor"]["a"][:2]

    # Increase f1 slightly above tolerance but keep it below f2
    bad_f1 = base_f1 * 1.3
    assert bad_f1 < base_f2  # sanity check

    ok, reason = is_plausible_formants(bad_f1, base_f2, "tenor", "a")

    assert not ok
    assert "f1 out of range" in reason


def test_is_plausible_formants_ok():
    base_f1, base_f2 = FORMANTS["tenor"]["a"][:2]
    ok, reason = is_plausible_formants(base_f1, base_f2, "tenor", "a")
    assert ok and reason == "ok"


# ---------------------------------------------------------
# is_plausible_pitch
# ---------------------------------------------------------

def test_is_plausible_pitch_missing():
    ok, reason = is_plausible_pitch(None)
    assert not ok and "missing" in reason

    ok, reason = is_plausible_pitch(np.nan)
    assert not ok and "missing" in reason


def test_is_plausible_pitch_out_of_range():
    low, high = PITCH_RANGES["tenor"]
    ok, reason = is_plausible_pitch(low - 10, "tenor")
    assert not ok and "out of range" in reason


def test_is_plausible_pitch_ok():
    low, high = PITCH_RANGES["tenor"]
    ok, reason = is_plausible_pitch((low + high) / 2, "tenor")
    assert ok and reason == "ok"


# ---------------------------------------------------------
# guess_vowel
# ---------------------------------------------------------

def test_guess_vowel_missing_inputs():
    assert guess_vowel(None, 1500, "bass", last_guess="a") == "a"
    assert guess_vowel(500, None, "bass", last_guess="a") == "a"


def test_guess_vowel_too_close():
    # f2 - f1 < 500 → fallback
    assert guess_vowel(500, 800, "bass", last_guess="e") == "e"


def test_guess_vowel_basic():
    # Pick a vowel with large enough F2-F1 separation
    t1, t2 = FORMANTS["bass"]["e"][:2]
    assert guess_vowel(t1, t2, "bass") == "e"


def test_guess_vowel_too_close_returns_last_guess():
    t1, t2 = FORMANTS["bass"]["a"][:2]  # diff < 500
    assert guess_vowel(t1, t2, "bass", last_guess="a") == "a"


# ---------------------------------------------------------
# robust_guess
# ---------------------------------------------------------

def test_robust_guess_not_enough_valid():
    vowel, conf, second = robust_guess([None, np.nan], "bass")
    assert vowel is None and conf == 0.0 and second is None


def test_robust_guess_basic():
    t1, t2 = FORMANTS["bass"]["a"][:2]
    vowel, conf, second = robust_guess([t1, t2], "bass")

    assert vowel == "a"
    assert conf > 1          # ratio, not probability
    assert isinstance(conf, float)
    assert second is not None


# ---------------------------------------------------------
# get_expected_formants
# ---------------------------------------------------------

def test_get_expected_formants_valid():
    f1, f2 = get_expected_formants("tenor", "a")
    base_f1, base_f2 = FORMANTS["tenor"]["a"][:2]
    assert f1 == int(round(base_f1))
    assert f2 == int(round(base_f2))


def test_get_expected_formants_invalid_vowel():
    f1, f2 = get_expected_formants("tenor", "zzz")
    assert f1 is None and f2 is None


def test_get_expected_formants_fallback_voice_type():
    f1a, f2a = get_expected_formants("alien", "a")
    f1b, f2b = get_expected_formants("tenor", "a")
    assert (f1a, f2a) == (f1b, f2b)
