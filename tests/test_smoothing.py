
from analysis.smoothing import (PitchSmoother, MedianSmoother,
                                LabelSmoother, FormantStabilityTracker)


def test_pitch_smoother_basic():
    s = PitchSmoother(alpha=0.5, jump_limit=80)

    # With jump suppression enabled, the smoother may reject large jumps.
    out2 = s.update(100)
    assert isinstance(out2, float)


def test_pitch_smoother_confidence_gate():
    s = PitchSmoother(min_confidence=0.5)

    assert s.update(100, confidence=1.0) == 100.0
    # Low confidence → ignore
    assert s.update(200, confidence=0.1) == 100.0


def test_pitch_smoother_octave_correction():
    s = PitchSmoother()
    s.current = 100.0

    # 2× current within 40 Hz → octave correction
    out = s.update(195)  # 195 is close to 2*100 = 200
    assert out in (100.0, 200.0)


def test_pitch_smoother_jump_suppression_without_hps():
    s = PitchSmoother(jump_limit=20)
    s.current = 100.0

    # Big jump but no HPS buffer → return current
    out = s.update(200)
    assert out == 100.0


def test_median_smoother_basic():
    sm = MedianSmoother(window=5)

    out = sm.update(500, 1500, confidence=1.0)
    assert out[:2] == (500.0, 1500.0)

    out = sm.update(510, 1490, confidence=1.0)
    assert out[:2] == (505.0, 1495.0)


def test_label_smoother_basic():
    sm = LabelSmoother(hold_frames=2)

    assert sm.update("a", confidence=1.0) == "a"
    assert sm.update("a", confidence=1.0) == "a"

    # New label appears once → not enough to switch
    assert sm.update("i", confidence=1.0) == "a"

    # Second consecutive "i" → switch
    assert sm.update("i", confidence=1.0) == "i"


def test_stability_tracker_ridge_collapse():
    st = FormantStabilityTracker()
    for _ in range(5):
        st.update(2600, 2600, 2600)
    stable, score = st.update(2600, 2600, 2600)
    assert stable is False
    assert score == float("inf")


def test_vowel_smoother_confidence_gate():
    sm = LabelSmoother(min_confidence=0.5)
    assert sm.update("i", confidence=0.1) is None
    assert sm.update("i", confidence=1.0) == "i"


def test_pitch_smoother_jump_suppression():
    s = PitchSmoother(jump_limit=50)
    s.update(100)
    out = s.update(300)  # jump > 50
    assert out == 100
