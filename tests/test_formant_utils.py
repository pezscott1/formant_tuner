from formant_utils import (
    is_plausible_formants,
    safe_spectrogram,
    estimate_formants_lpc,
    normalize_profile_for_save,
    pick_formants,
    unpack_formants,
    estimate_pitch,
    guess_vowel,
    robust_guess,
    get_expected_formants,
    get_vowel_ranges,
    smoothed_spectrum_peaks,
    lpc_envelope_peaks,
    directional_feedback,
    choose_best_candidate,
    live_score_formants,
    resonance_tuning_score,
    plausibility_score,
    is_plausible_pitch,
    LabelSmoother,
    PitchSmoother,
    hz_to_midi)
from tests.conftest import synth_vowel
import numpy as np


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


def test_pick_formants_too_few_candidates():
    f1, f2 = pick_formants([500])
    assert f1 == 500.0 and f2 is None


def test_estimate_formants_empty_signal():
    res = estimate_formants_lpc(np.array([]), 16000)
    f1, f2, f3 = unpack_formants(res)
    assert f1 is None and f2 is None


def test_estimate_formants_empty_and_short():
    # Empty signal
    result = estimate_formants_lpc(np.array([]), 16000)
    assert result == (None, None, None)
    # Very short signal
    result = estimate_formants_lpc(np.ones(10), 16000)
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_estimate_formants_with_nan(monkeypatch):
    # Patch librosa.lpc to return nonsense so we hit the exception path
    monkeypatch.setattr("formant_utils.librosa.lpc", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail")))
    result = estimate_formants_lpc(np.ones(100), 16000)
    assert result == (None, None, None)


def test_pick_formants_no_candidates(monkeypatch):
    result = pick_formants([])
    assert result == (None, None)


def test_pick_formants_minimal_candidates():
    # Flat list of plausible frequencies
    candidates = [500, 1500]
    f1, f2 = pick_formants(candidates)
    assert f1 == 500 and f2 == 1500


def test_safe_spectrogram_handles_bad_input():
    # Passing nonsense should not crash
    freqs, times, S = safe_spectrogram("not an array", 16000)
    assert isinstance(freqs, np.ndarray)
    assert isinstance(times, np.ndarray)
    assert isinstance(S, np.ndarray)


def test_get_vowel_ranges_and_is_plausible_formants():
    ranges = get_vowel_ranges("tenor", "a")
    assert len(ranges) == 4
    ok, reason = is_plausible_formants(ranges[0]+1, ranges[2]+1, "tenor", "a")
    assert ok is True
    ok, reason = is_plausible_formants(None, None)
    assert ok is False


def test_is_plausible_pitch_range_and_missing():
    ok, reason = is_plausible_pitch(200, "tenor")
    assert ok
    ok, reason = is_plausible_pitch(None, "tenor")
    assert not ok


def test_guess_vowel_with_none_and_valid():
    assert guess_vowel(None, None, "tenor", last_guess="a") == "a"
    assert isinstance(guess_vowel(500, 1500, "tenor"), str)


def test_estimate_pitch_empty_and_valid():
    assert estimate_pitch([], 16000) is None
    # Simple sine wave
    sr = 16000
    t = np.linspace(0, 0.02, int(sr*0.02), endpoint=False)
    frame = np.sin(2*np.pi*200*t)
    f0 = estimate_pitch(frame, sr)
    assert f0 is not None


def test_lpc_envelope_peaks_and_smoothed_spectrum_peaks():
    frame = np.random.randn(512)
    freqs, heights = lpc_envelope_peaks(frame, 16000)
    assert isinstance(freqs, np.ndarray)
    freqs2, heights2 = smoothed_spectrum_peaks(frame, 16000)
    assert isinstance(freqs2, np.ndarray)


def test_pick_formants_various_cases():
    assert pick_formants([]) == (None, None)
    f1, f2 = pick_formants([300, 1200, 2000])
    assert f1 is not None
    f1, f2 = pick_formants([100, 200])
    assert f1 is not None


def test_estimate_formants_lpc_empty_and_low_energy(monkeypatch):
    assert estimate_formants_lpc([], 16000) == (None, None, None)
    assert estimate_formants_lpc(np.zeros(100), 16000) == (None, None, None)


def test_unpack_formants_various_shapes():
    assert unpack_formants(None) == (None, None, None)
    assert unpack_formants((1, 2, 3)) == (1, 2, 3)
    assert unpack_formants((1, 2)) == (1, 2, None)


def test_normalize_profile_for_save_with_dict_and_tuple():
    d = {"a": (500, 1500, 200)}
    out = normalize_profile_for_save(d, retries_map={"a": 1})
    assert "a" in out
    d2 = {"e": {"f1": 400, "f2": 1200, "f3": 250}}
    out2 = normalize_profile_for_save(d2)
    assert "e" in out2


def test_safe_spectrogram_empty_and_short():
    f, t, S = safe_spectrogram([], 16000)
    assert S.shape[0] > 0
    f2, t2, S2 = safe_spectrogram(np.ones(100), 16000, n_fft=2048)
    assert S2.shape[0] > 0


def test_get_expected_formants_and_directional_feedback():
    f1, f2 = get_expected_formants("tenor", "a")
    assert isinstance(f1, int)
    fb1, fb2 = directional_feedback((400, 1200), {"a": {"f1": 500, "f2": 1500}}, "a", 50)
    assert fb1 is not None or fb2 is not None


def test_plausibility_score_and_choose_best_candidate():
    sc = plausibility_score(500, 1500)
    assert isinstance(sc, float)
    best = choose_best_candidate({"f1": 500, "f2": 1500}, [{"f1": 600, "f2": 1600}])
    assert "f1" in best


def test_live_score_formants_and_resonance_tuning_score():
    score = live_score_formants((500, 1500, 200), (500, 1500, 200))
    assert score > 0
    score2 = resonance_tuning_score((500, 1500, 200), 100)
    assert isinstance(score2, int)


def test_robust_guess_and_label_pitch_smoothers():
    vowel, conf, second = robust_guess((500, 1500), "tenor")
    assert isinstance(vowel, str) or vowel is None
    ls = LabelSmoother(window=3, min_dwell=2)
    assert ls.update("a") in (None, "a")
    ps = PitchSmoother(window=3)
    assert ps.update(200) == 200.0
    assert hz_to_midi(440) == 69
    assert hz_to_midi(None) is None
