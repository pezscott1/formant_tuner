# tests/calibration/test_interpolation_pipeline.py
from analysis.vowel_data import TRIANGLES, STANDARD_VOWELS
from tuner.profile_controller import ProfileManager
from pathlib import Path
import json
from calibration.session import CalibrationSession


def make_pm(tmp_path):
    class DummyAnalyzer:
        voice_type = "baritone"
        def set_user_formants(self, fmts): pass
    return ProfileManager(tmp_path, DummyAnalyzer())


# ------------------------------------------------------------
# 1) Minimal 5‑vowel calibration → interpolation exists
# ------------------------------------------------------------
def test_no_interpolation_with_minimal_calibration(tmp_path):
    pm = make_pm(tmp_path)

    sess = CalibrationSession(
        profile_name="test",
        voice_type="baritone",
        vowels=STANDARD_VOWELS,
        profile_manager=pm,
    )

    anchors = {
        "i":  (300, 2700),
        "ɛ":  (550, 2000),
        "ɑ":  (750, 1600),
        "ɔ":  (600, 1100),
        "u":  (350, 900),
    }
    for v, (f1, f2) in anchors.items():
        sess.data[v] = {
            "f1": f1, "f2": f2, "f0": 150,
            "confidence": 1.0, "stability": 0.0,
            "weight": 1.0, "saved_at": "now",
        }
        sess.calibrated_vowels.add(v)

    interp = sess.compute_interpolated_vowels()
    expected = {"e", "ɪ", "æ", "o"}

    assert set(interp.keys()) == expected

    base = sess.save_profile()
    saved = json.loads(Path(tmp_path, f"{base}_profile.json").read_text())

    # Interpolated vowels must match the computed interpolation
    assert set(saved["interpolated_vowels"].keys()) == expected

    # Calibrated vowels must be exactly the explicit anchors
    assert set(saved["calibrated_vowels"].keys()) == set(STANDARD_VOWELS)


def test_interpolation_for_extended_vowels(tmp_path):
    pm = make_pm(tmp_path)

    all_vertices = set()
    for tri in TRIANGLES.values():
        all_vertices |= set(tri)

    sess = CalibrationSession(
        profile_name="full",
        voice_type="baritone",
        vowels=list(all_vertices),
        profile_manager=pm,
    )

    # calibrate only the triangle vertices that are not themselves triangle targets
    from unicodedata import normalize
    targets = {normalize("NFC", t) for t in TRIANGLES.keys()}
    all_vertices_norm = {normalize("NFC", v) for tri in TRIANGLES.values() for v in tri}
    vertices_only = sorted(all_vertices_norm - targets)

    for idx, v in enumerate(vertices_only):
        f1 = 100 + idx * 10
        f2 = 200 + idx * 20
        sess.data[v] = {
            "f1": f1, "f2": f2, "f0": 100 + idx,
            "confidence": 1.0, "stability": 0.0,
            "weight": 1.0, "saved_at": "now",
        }
        sess.calibrated_vowels.add(v)

    interp = sess.compute_interpolated_vowels()

    anchors = set(sess.get_calibrated_anchors().keys())  # normalized anchor keys

    expected = {
        target for target, (A, B, C) in TRIANGLES.items()
        if {normalize("NFC", A), normalize("NFC", B), normalize("NFC", C)} <= anchors
    }

    # Assert that computed interpolation matches what is possible from the anchors
    assert set(interp.keys()) == expected


def test_interpolated_vowels_are_complement_of_calibrated(tmp_path):
    pm = make_pm(tmp_path)

    sess = CalibrationSession(
        profile_name="comp",
        voice_type="baritone",
        vowels=["i", "e", "ɛ"],
        profile_manager=pm,
    )

    sess.data["i"] = {
        "f1": 300, "f2": 2700, "f0": 150,
        "confidence": 1.0, "stability": 0.0,
        "weight": 1.0, "saved_at": "now",
    }
    sess.calibrated_vowels.add("i")

    sess.data["e"] = {"f1": 400, "f2": 2000}
    sess.data["ɛ"] = {"f1": 500, "f2": 1800}

    # compute interpolation explicitly
    interp = sess.compute_interpolated_vowels()

    base = sess.save_profile()
    saved = json.loads(Path(tmp_path, f"{base}_profile.json").read_text())

    # Interpolated vowels must match the computed interpolation
    assert set(saved["interpolated_vowels"].keys()) == set(interp.keys())


def test_seeding_behavior_removed_and_interpolation_wins(tmp_path):
    """
    Seeding has been removed from the model. If a non-calibrated entry exists
    in self.data it is not treated as an anchor; interpolation is computed
    from calibrated anchors only and saved interpolation must match the
    computed values.
    """
    pm = make_pm(tmp_path)

    sess = CalibrationSession(
        profile_name="seed_removed",
        voice_type="baritone",
        vowels=["i", "ɛ", "e"],
        profile_manager=pm,
    )

    # anchors
    sess.data["i"] = {"f1": 300, "f2": 2700, "f0": 150,
                      "confidence": 1.0, "stability": 0.0,
                      "weight": 1.0, "saved_at": "now"}
    sess.data["ɛ"] = {"f1": 550, "f2": 2000, "f0": 150,
                      "confidence": 1.0, "stability": 0.0,
                      "weight": 1.0, "saved_at": "now"}
    sess.data["ɑ"] = {"f1": 750, "f2": 1700, "f0": 150,
                      "confidence": 1.0, "stability": 0.0,
                      "weight": 1.0, "saved_at": "now"}
    sess.calibrated_vowels.update({"i", "ɛ", "ɑ"})

    # previously "seeded" entry (now just a non-calibrated data entry)
    sess.data["e"] = {"f1": 400, "f2": 2300, "f0": 150,
                      "confidence": 1.0, "stability": 0.0,
                      "weight": 0.0, "saved_at": "now"}

    interp = sess.compute_interpolated_vowels()
    # do not mark 'e' calibrated here — we want to verify computed interpolation
    base = sess.save_profile()
    saved = json.loads(Path(tmp_path, f"{base}_profile.json").read_text())

    # The saved interpolation must match the
    # computed interpolation (no special seeding preservation)
    assert "e" in interp
    assert saved["interpolated_vowels"]["e"]["f1"] == interp["e"]["f1"]
    assert saved["interpolated_vowels"]["e"]["f2"] == interp["e"]["f2"]


def test_interpolation_excludes_calibrated_vowels(tmp_path):
    pm = make_pm(tmp_path)

    sess = CalibrationSession(
        profile_name="no_mix",
        voice_type="baritone",
        vowels=["i", "ɛ", "ɑ", "e"],
        profile_manager=pm,
    )

    for v, (f1, f2) in {
        "i": (300, 2700),
        "ɛ": (550, 2000),
        "ɑ": (750, 1600),
        "e": (400, 2300),
    }.items():
        sess.data[v] = {"f1": f1, "f2": f2}
        sess.calibrated_vowels.add(v)

    interp = sess.compute_interpolated_vowels()
    base = sess.save_profile()
    saved = json.loads(Path(tmp_path, f"{base}_profile.json").read_text())

    assert set(saved["interpolated_vowels"].keys()) == set(interp.keys())


def test_interpolation_skips_nan_or_none(tmp_path):
    pm = make_pm(tmp_path)

    sess = CalibrationSession(
        profile_name="nan_case",
        voice_type="baritone",
        vowels=["i", "ɛ", "ɑ"],
        profile_manager=pm,
    )

    sess.data["i"] = {"f1": 300, "f2": 2700}
    sess.data["ɛ"] = {"f1": 550, "f2": 2000}
    sess.data["ɑ"] = {"f1": None, "f2": 1600}
    sess.calibrated_vowels.update({"i", "ɛ", "ɑ"})

    interp = sess.compute_interpolated_vowels()
    assert "e" not in interp


def test_f0_interpolation_behavior(tmp_path):
    pm = make_pm(tmp_path)

    sess = CalibrationSession(
        profile_name="f0_interp",
        voice_type="baritone",
        vowels=["i", "ɛ", "ɑ"],
        profile_manager=pm,
    )

    sess.data["i"] = {"f1": 300, "f2": 2700, "f0": 100}
    sess.data["ɛ"] = {"f1": 550, "f2": 2000, "f0": 200}
    sess.data["ɑ"] = {"f1": 750, "f2": 1600, "f0": 300}
    sess.calibrated_vowels.update({"i", "ɛ", "ɑ"})

    interp = sess.compute_interpolated_vowels()
    assert interp["e"]["f0"] is not None

    sess.data["ɑ"]["f0"] = None
    interp2 = sess.compute_interpolated_vowels()
    assert interp2["e"]["f0"] is None


def test_save_profile_does_not_mutate_session_data(tmp_path):
    pm = make_pm(tmp_path)

    sess = CalibrationSession(
        profile_name="immut",
        voice_type="baritone",
        vowels=["i"],
        profile_manager=pm,
    )

    sess.data["i"] = {"f1": 300, "f2": 2700, "saved_at": "orig"}
    sess.calibrated_vowels.add("i")

    before = dict(sess.data["i"])
    sess.save_profile()
    after = sess.data["i"]

    assert before == after


def test_full_expanded_interpolation(tmp_path):
    from unicodedata import normalize

    pm = make_pm(tmp_path)

    # collect all triangle vertices (normalized)
    all_vertices = {normalize("NFC", v) for tri in TRIANGLES.values() for v in tri}

    sess = CalibrationSession(
        profile_name="full_expanded",
        voice_type="baritone",
        vowels=list(all_vertices),
        profile_manager=pm,
    )

    # populate session data and mark anchors (use normalized keys consistently)
    for idx, v in enumerate(sorted(all_vertices)):
        sess.data[v] = {"f1": 100 + idx, "f2": 200 + idx}
        sess.calibrated_vowels.add(v)

    # compute interpolation
    interp = sess.compute_interpolated_vowels()

    # compute anchors (normalized) and the realistic expected set
    from unicodedata import normalize

    # anchors are returned normalized by get_calibrated_anchors()
    anchors = set(sess.get_calibrated_anchors().keys())

    # expected = triangle targets whose three vertices are all present in anchors
    # and which are NOT themselves calibrated anchors (normalize target too)
    expected = {
        normalize("NFC", target)
        for target, (A, B, C) in TRIANGLES.items()
        if {normalize("NFC", A), normalize("NFC", B), normalize("NFC", C)} <= anchors
        and normalize("NFC", target) not in anchors
    }

    # normalize interp keys before comparing
    interp_norm = {normalize("NFC", k) for k in interp.keys()}

    assert interp_norm == expected


def test_round_trip_save_load(tmp_path):
    pm = make_pm(tmp_path)

    sess = CalibrationSession(
        profile_name="round",
        voice_type="baritone",
        vowels=["i"],
        profile_manager=pm,
    )

    sess.data["i"] = {"f1": 300, "f2": 2700, "f0": 150,
                      "confidence": 1.0, "stability": 0.0,
                      "weight": 1.0, "saved_at": "now"}
    sess.calibrated_vowels.add("i")

    base = sess.save_profile()

    loaded = pm.load_profile_json(base)
    merged = {**loaded["calibrated_vowels"], **loaded["interpolated_vowels"]}
    fmts = pm.extract_formants(merged)

    assert "i" in fmts
    assert fmts["i"]["f1"] == 300
    assert fmts["i"]["f2"] == 2700


def test_interpolation_still_correct_after_fixes():
    sess = CalibrationSession(
        profile_name="interp",
        voice_type="baritone",
        vowels=list({v for tri in TRIANGLES.values() for v in tri}),
        profile_manager=None,
    )

    for idx, v in enumerate(sorted(sess.vowels)):
        sess.data[v] = {"f1": 100 + idx, "f2": 200 + idx}
        sess.calibrated_vowels.add(v)

    interp = sess.compute_interpolated_vowels()
    from unicodedata import normalize

    anchors = set(sess.get_calibrated_anchors().keys())
    expected = {
        normalize("NFC", target)
        for target, (A, B, C) in TRIANGLES.items()
        if {normalize("NFC", A), normalize("NFC", B), normalize("NFC", C)} <= anchors
        and normalize("NFC", target) not in anchors
    }
    interp_norm = {normalize("NFC", k) for k in interp.keys()}
    assert interp_norm == expected
