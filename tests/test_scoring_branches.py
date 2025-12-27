import numpy as np
from analysis.scoring import (
    directional_feedback,
    plausibility_score,
    choose_best_candidate,
    live_score_formants,
    resonance_tuning_score,
)


# ----------------------------------------------------------------------
# directional_feedback
# ----------------------------------------------------------------------

def test_directional_feedback_missing():
    out = directional_feedback((None, 500), {"a": {"f1": 400, "f2": 1500}}, "a", 50)
    assert out == (None, None)


def test_directional_feedback_standard_raise_lower():
    measured = (300, 1600)
    user = {"a": {"f1": 400, "f2": 1500}}
    fb = directional_feedback(measured, user, "a", 50)
    assert fb == ("↑ raise F1", "↓ lower F2")


def test_directional_feedback_standard_ok():
    measured = (410, 1520)
    user = {"a": {"f1": 400, "f2": 1500}}
    fb = directional_feedback(measured, user, "a", 50)
    assert fb == (None, None)


def test_directional_feedback_back_vowel_f2_raise():
    measured = (500, 1200)
    user = {"u": {"f1": 400, "f2": 1500}}
    fb = directional_feedback(measured, user, "u", 50)
    assert fb == (None, "↑ raise F2")


def test_directional_feedback_back_vowel_f2_lower():
    measured = (500, 1700)
    user = {"u": {"f1": 400, "f2": 1500}}
    fb = directional_feedback(measured, user, "u", 50)
    assert fb == (None, "↓ lower F2")


def test_directional_feedback_back_vowel_f1_drift():
    measured = (700, 1500)
    user = {"u": {"f1": 400, "f2": 1500}}
    fb = directional_feedback(measured, user, "u", 50)
    assert fb == ("adjust F1 (drift)", None)


# ----------------------------------------------------------------------
# plausibility_score
# ----------------------------------------------------------------------

def test_plausibility_missing():
    assert plausibility_score(None, 500) == 0.0


def test_plausibility_swapped():
    assert plausibility_score(800, 500) == -100.0


def test_plausibility_too_close():
    assert plausibility_score(500, 650) == -50.0


def test_plausibility_ridge_penalty():
    # f2 near 2600 → penalty applied
    s1 = plausibility_score(500, 2600)
    s2 = plausibility_score(500, 2000)
    assert s1 > s2


def test_plausibility_general():
    s = plausibility_score(400, 1200)
    assert isinstance(s, float)
    assert s > 0


# ----------------------------------------------------------------------
# choose_best_candidate
# ----------------------------------------------------------------------

def test_choose_best_candidate_identical_ignored():
    initial = {"f1": 500, "f2": 1500}
    retakes = [{"f1": 510, "f2": 1510}]  # within 20 → ignored
    best = choose_best_candidate(initial, retakes)
    assert best is initial


def test_choose_best_candidate_better_retake():
    initial = {"f1": 500, "f2": 1500}
    retakes = [{"f1": 400, "f2": 2000}]  # much better spacing
    best = choose_best_candidate(initial, retakes)
    assert best is retakes[0]


def test_choose_best_candidate_worse_retake():
    initial = {"f1": 500, "f2": 1500}
    retakes = [{"f1": 600, "f2": 650}]  # too close → bad
    best = choose_best_candidate(initial, retakes)
    assert best is initial


# ----------------------------------------------------------------------
# live_score_formants
# ----------------------------------------------------------------------

def test_live_score_formants_missing():
    assert live_score_formants((500, 1500, 2500), (None, None, None)) == 0


def test_live_score_formants_nan():
    assert live_score_formants((500, 1500, 2500), (np.nan, 1500, 2500)) == 100


def test_live_score_formants_basic():
    score = live_score_formants((500, 1500, 2500), (500, 1500, 2500))
    assert score == 100


def test_live_score_formants_partial():
    score = live_score_formants((500, 1500, None), (500, 1500, None))
    assert score == 100


def test_live_score_formants_distance():
    score1 = live_score_formants((500, 1500, 2500), (500, 1500, 2500))
    score2 = live_score_formants((500, 1500, 2500), (700, 1500, 2500))
    assert score2 < score1


# ----------------------------------------------------------------------
# resonance_tuning_score
# ----------------------------------------------------------------------

class DummyPitch:
    def __init__(self, f0):
        self.f0 = f0


def test_resonance_pitch_none():
    assert resonance_tuning_score((500, 1500, 2500), None) == 0


def test_resonance_pitch_unwrap():
    out = resonance_tuning_score((500, 1500, 2500), DummyPitch(100))
    assert isinstance(out, int)


def test_resonance_pitch_non_numeric():
    assert resonance_tuning_score((500, 1500, 2500), "bad") == 0


def test_resonance_pitch_nan():
    assert resonance_tuning_score((500, 1500, 2500), float("nan")) == 0


def test_resonance_missing_formants():
    assert resonance_tuning_score((None, None, None), 100) == 0


def test_resonance_basic_alignment():
    # F1=100 aligns with harmonic 1*100
    out = resonance_tuning_score((100, None, None), 100)
    assert out == 100

    s1 = resonance_tuning_score((100, None, None), 100)   # perfect
    s2 = resonance_tuning_score((250, None, None), 100)   # not harmonic
    assert s2 < s1
