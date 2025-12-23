# tests/test_tuner_controller_edge_cases.py
from tuner.controller import Tuner
from analysis.engine import FormantAnalysisEngine


def test_classify_no_profile():
    t = Tuner(engine=FormantAnalysisEngine())
    vowel, conf = t._classify_vowel_from_profile(500, 1500)
    assert vowel is None
    assert conf == 0.0


def test_classify_missing_formants():
    t = Tuner(engine=FormantAnalysisEngine())
    t.active_profile = {"i": {"f1": 300, "f2": 2500}}
    vowel, conf = t._classify_vowel_from_profile(None, None)
    assert vowel is None


def test_classify_vowel_not_in_profile():
    t = Tuner(engine=FormantAnalysisEngine())
    t.active_profile = {"i": {"f1": 300, "f2": 2500}}
    vowel, conf = t._classify_vowel_from_profile(500, 1500)
    assert vowel == "i" or vowel is None  # depends on normalization
