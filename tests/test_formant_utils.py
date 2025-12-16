from formant_utils import estimate_formants_lpc, pick_formants, unpack_formants
from tests.conftest import synth_vowel


def test_pick_formants_basic():
    candidates = [250.0, 900.0, 1200.0, 2300.0]
    f1, f2 = pick_formants(candidates)
    assert 200 <= f1 <= 400
    assert 800 <= f2 <= 2500
    assert f1 < f2


def test_estimate_formants_on_synthetic():
    # synth /i/ roughly F1=300, F2=2300
    sr = 16000
    y = synth_vowel([300, 2300], sr=sr, dur=0.5, f0=220.0)
    res = estimate_formants_lpc(y, sr)
    f1, f2, _ = unpack_formants(res)
    assert f1 is not None and f2 is not None
    assert 200 <= f1 <= 800
    assert 1000 <= f2 <= 2600
