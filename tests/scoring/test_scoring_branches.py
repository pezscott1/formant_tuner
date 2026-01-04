import numpy as np
from analysis.scoring import (
    plausibility_score,
    live_score_formants,
    resonance_tuning_score,
)


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
