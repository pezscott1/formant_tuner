import numpy as np
from analysis.scoring import (
    directional_feedback,
    plausibility_score,
    choose_best_candidate,
    live_score_formants,
    resonance_tuning_score,
)


# ---------------------------------------------------------
# directional_feedback
# ---------------------------------------------------------

def test_directional_feedback_raise_lower():
    measured = (400, 1600)
    user = {"a": {"f1": 500, "f2": 1500}}
    fb_f1, fb_f2 = directional_feedback(measured, user, "a", tolerance=50)

    assert fb_f1 == "↑ raise F1"
    assert fb_f2 == "↓ lower F2"


def test_directional_feedback_none_when_within_tolerance():
    measured = (505, 1490)
    user = {"a": {"f1": 500, "f2": 1500}}
    fb_f1, fb_f2 = directional_feedback(measured, user, "a", tolerance=50)

    assert fb_f1 is None
    assert fb_f2 is None


def test_directional_feedback_missing_targets():
    measured = (400, 1600)
    user = {}  # no entry
    fb_f1, fb_f2 = directional_feedback(measured, user, "a", tolerance=50)

    assert fb_f1 is None
    assert fb_f2 is None


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
# choose_best_candidate
# ---------------------------------------------------------

def test_choose_best_candidate_selects_higher_score():
    initial = {"f1": 500, "f2": 1500}
    retakes = [
        {"f1": 600, "f2": 1600},  # slightly worse
        {"f1": 450, "f2": 1700},  # better separation
    ]

    best = choose_best_candidate(initial, retakes)
    assert best == retakes[1]


def test_choose_best_candidate_no_better_option():
    initial = {"f1": 500, "f2": 1500}
    retakes = [{"f1": 510, "f2": 1510}]
    best = choose_best_candidate(initial, retakes)
    assert best == initial


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
