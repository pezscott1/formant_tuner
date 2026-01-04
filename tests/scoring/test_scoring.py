import numpy as np
from analysis.scoring import (
    plausibility_score,
    live_score_formants,
    resonance_tuning_score,
)


# ---------------------------------------------------------
# plausibility_score
# ---------------------------------------------------------

def test_plausibility_score_none_inputs():
    assert plausibility_score(None, 500) == 0.0
    assert plausibility_score(500, None) == 0.0


def test_plausibility_score_basic_behavior():
    # f2 > f1 → positive separation
    s1 = plausibility_score(500, 1500)
    # f2 < f1 → separation clamped to 0
    s2 = plausibility_score(1500, 500)

    assert s1 > 0
    assert s2 <= 0


# ---------------------------------------------------------
# live_score_formants
# ---------------------------------------------------------

def test_live_score_formants_perfect_match():
    target = (500, 1500)
    measured = (500, 1500)
    assert live_score_formants(target, measured, tolerance=50) == 100


def test_live_score_formants_partial_match():
    target = (500, 1500)
    measured = (525, 1600)  # F1 close, F2 outside tolerance
    score = live_score_formants(target, measured, tolerance=50)

    # Only F1 contributes → score should be between 0 and 100
    assert 0 < score < 100


def test_live_score_formants_no_valid_pairs():
    target = (None, None)
    measured = (500, 1500)
    assert live_score_formants(target, measured) == 0


# ---------------------------------------------------------
# resonance_tuning_score
# ---------------------------------------------------------

def test_resonance_tuning_score_perfect_alignment():
    pitch = 100
    formants = [100, 200, 300]  # exact harmonics
    assert resonance_tuning_score(formants, pitch) == 100


def test_resonance_tuning_score_partial_alignment():
    pitch = 100
    formants = [105, 250, 800]  # only first is close
    score = resonance_tuning_score(formants, pitch, tolerance=20)

    assert 0 < score < 100


def test_resonance_tuning_score_invalid_pitch():
    assert resonance_tuning_score([500, 1500], None) == 0
    assert resonance_tuning_score([500, 1500], np.nan) == 0


def test_resonance_tuning_score_no_valid_formants():
    assert resonance_tuning_score([None, None], 100) == 0
