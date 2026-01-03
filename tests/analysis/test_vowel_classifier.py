from analysis.vowel_classifier import classify_vowel


def test_classifier_unknown_voice_type_returns_none():
    res = classify_vowel(500, 1500, centers=None, voice_type="unknown_voice")
    assert res == (None, 0.0, None)


def test_classifier_missing_f1():
    res = classify_vowel(None, 1500)
    assert res[2] == {"reason": "missing_f1"}
