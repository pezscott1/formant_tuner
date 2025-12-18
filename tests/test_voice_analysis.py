import numpy as np


def test_is_plausible_formants_missing_and_swapped():
    from voice_analysis import is_plausible_formants
    ok, reason = is_plausible_formants(None, None, "tenor", "a")
    assert not ok and "missing" in reason
    ok, reason = is_plausible_formants(1500, 500, "tenor", "a")
    assert not ok and "swapped" in reason


def test_is_plausible_pitch_range_and_nan():
    from formant_utils import is_plausible_pitch
    ok, reason = is_plausible_pitch(200, "tenor")
    assert ok
    ok, reason = is_plausible_pitch(np.nan, "tenor")
    assert not ok


def test_guess_vowel_fallback_and_valid():
    from voice_analysis import guess_vowel
    # Missing inputs fall back to last_guess
    assert guess_vowel(None, None, "tenor", last_guess="a") == "a"
    # Valid plausible inputs return a vowel string
    vowel = guess_vowel(500, 1500, "tenor")
    assert isinstance(vowel, str)


def test_plausibility_score_and_choose_best_candidate():
    from formant_utils import plausibility_score, choose_best_candidate
    score = plausibility_score(500, 1500)
    assert isinstance(score, float)
    initial = {"f1": 500, "f2": 1500}
    retakes = [{"f1": 600, "f2": 1600}, {"f1": 400, "f2": 1400}]
    best = choose_best_candidate(initial, retakes)
    assert "f1" in best and "f2" in best


def test_live_score_formants_and_resonance_tuning_score():
    from voice_analysis import live_score_formants, resonance_tuning_score
    target = (500, 1500)
    measured = (500, 1500)
    score = live_score_formants(target, measured, tolerance=100)
    assert score > 0
    # Resonance tuning with pitch
    formants = (500, 1500)
    score2 = resonance_tuning_score(formants, 100)
    assert isinstance(score2, int)
    # Pitch None returns 0
    assert resonance_tuning_score(formants, None) == 0


def test_robust_guess_with_valid_and_insufficient():
    from formant_utils import robust_guess
    # Insufficient formants returns None
    vowel, conf, second = robust_guess((None, None), "tenor")
    assert vowel is None
    # Valid formants return a vowel guess
    vowel, conf, second = robust_guess((500, 1500), "tenor")
    assert isinstance(vowel, str)
    assert isinstance(conf, float)
    assert isinstance(second, str)
