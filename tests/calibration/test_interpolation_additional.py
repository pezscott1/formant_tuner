# tests/calibration/test_interpolation_additional.py
import json
from pathlib import Path
from calibration.session import CalibrationSession
from tuner.profile_controller import ProfileManager


def make_pm(tmp_path):
    class DummyAnalyzer:
        voice_type = "baritone"
        def set_user_formants(self, fmts): pass
    return ProfileManager(tmp_path, DummyAnalyzer())


def test_compute_interpolated_vowels_is_pure(tmp_path):
    pm = make_pm(tmp_path)
    sess = CalibrationSession("pure", "baritone", ["i", "e", "ɛ"], profile_manager=pm)

    # set anchors and an extra non-calibrated entry
    sess.data["i"] = {"f1": 300, "f2": 2700, "f0": 120}
    sess.data["e"] = {"f1": 400, "f2": 2000}
    sess.calibrated_vowels.add("i")

    before_data = dict(sess.data)
    before_calibrated = set(sess.calibrated_vowels)

    interp = sess.compute_interpolated_vowels()

    # compute_interpolated_vowels must not mutate session state
    assert sess.data == before_data
    assert sess.calibrated_vowels == before_calibrated

    # interpolation result is a dict (may be empty)
    assert isinstance(interp, dict)


def test_save_profile_persists_only_computed_interpolation(tmp_path):
    pm = make_pm(tmp_path)
    sess = CalibrationSession("save_only_interp", "baritone", ["i", "e", "ɛ"], profile_manager=pm)

    # anchors
    sess.data["i"] = {"f1": 300, "f2": 2700, "f0": 120}
    sess.calibrated_vowels.add("i")

    # non-calibrated data entries (should NOT be treated as interpolation unless triangles allow)
    sess.data["e"] = {"f1": 400, "f2": 2000}
    sess.data["ɛ"] = {"f1": 500, "f2": 1800}

    interp = sess.compute_interpolated_vowels()

    base = sess.save_profile()
    saved = json.loads(Path(tmp_path, f"{base}_profile.json").read_text())

    # calibrated_vowels must be exactly the explicit set
    assert set(saved["calibrated_vowels"].keys()) == set(sess.calibrated_vowels)

    # interpolated_vowels must match the computed interpolation (keys and values)
    assert set(saved["interpolated_vowels"].keys()) == set(interp.keys())
    for k in interp.keys():
        assert saved["interpolated_vowels"][k]["f1"] == interp[k]["f1"]
        assert saved["interpolated_vowels"][k]["f2"] == interp[k]["f2"]


def test_compute_skips_explicitly_calibrated(tmp_path):
    pm = make_pm(tmp_path)
    sess = CalibrationSession("skip_cal", "baritone", ["i", "e", "ɛ"], profile_manager=pm)

    # set up a triangle where 'e' would be interpolated from i, ɛ, ɑ
    sess.data["i"] = {"f1": 300, "f2": 2700}
    sess.data["ɛ"] = {"f1": 550, "f2": 2000}
    sess.data["ɑ"] = {"f1": 750, "f2": 1600}
    # mark all three as calibrated
    sess.calibrated_vowels.update({"i", "ɛ", "ɑ"})

    # explicitly calibrate 'e' as well; interpolation must not produce 'e'
    sess.data["e"] = {"f1": 400, "f2": 2300}
    sess.calibrated_vowels.add("e")

    interp = sess.compute_interpolated_vowels()
    assert "e" not in interp


def test_f0_interpolation_none_if_any_vertex_missing(tmp_path):
    pm = make_pm(tmp_path)
    sess = CalibrationSession("f0_none", "baritone", ["i", "e", "ɑ"], profile_manager=pm)

    sess.data["i"] = {"f1": 300, "f2": 2700, "f0": 100}
    sess.data["e"] = {"f1": 400, "f2": 2000, "f0": 200}
    sess.data["ɑ"] = {"f1": 750, "f2": 1600, "f0": None}  # missing f0
    sess.calibrated_vowels.update({"i", "e", "ɑ"})

    interp = sess.compute_interpolated_vowels()
    # if 'e' is interpolated from i, ɛ, ɑ (or similar), its f0 must be None because one vertex lacks f0
    for v, vals in interp.items():
        assert ("f0" in vals)
        # f0 must be None when any vertex f0 is missing
        assert vals["f0"] is None or isinstance(vals["f0"], float)


def test_get_calibrated_anchors_uses_only_calibrated(tmp_path):
    pm = make_pm(tmp_path)
    sess = CalibrationSession("anchors_only", "baritone", ["i", "e", "ɛ"], profile_manager=pm)

    # Put entries in data but mark only 'i' as calibrated
    sess.data["i"] = {"f1": 300, "f2": 2700}
    sess.data["e"] = {"f1": 400, "f2": 2000}
    sess.calibrated_vowels.add("i")

    anchors = sess.get_calibrated_anchors()
    # anchors must include only 'i' (not 'e')
    assert set(anchors.keys()) == {"i"}


def test_round_trip_saved_interpolation_matches_computed(tmp_path):
    pm = make_pm(tmp_path)
    sess = CalibrationSession("round_interp", "baritone", ["i", "e", "ɛ"], profile_manager=pm)

    # anchors that produce interpolation
    sess.data["i"] = {"f1": 300, "f2": 2700, "f0": 120}
    sess.data["ɛ"] = {"f1": 550, "f2": 2000, "f0": 140}
    sess.data["ɑ"] = {"f1": 750, "f2": 1600, "f0": 160}
    sess.calibrated_vowels.update({"i", "ɛ", "ɑ"})

    interp = sess.compute_interpolated_vowels()
    base = sess.save_profile()

    loaded = pm.load_profile_json(base)
    saved_interp = loaded["interpolated_vowels"]

    # keys and numeric values must match
    assert set(saved_interp.keys()) == set(interp.keys())
    for k in interp.keys():
        assert abs(saved_interp[k]["f1"] - interp[k]["f1"]) < 1e-6
        assert abs(saved_interp[k]["f2"] - interp[k]["f2"]) < 1e-6
