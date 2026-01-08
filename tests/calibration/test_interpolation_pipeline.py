# test_interpolation_pipeline.py
from calibration.session import CalibrationSession
from analysis.vowel_data import TRIANGLES, TRIANGLE_WEIGHTS, STANDARD_VOWELS
from tuner.profile_controller import ProfileManager
from pathlib import Path
import json


# ------------------------------------------------------------
# Helper: build a fake profile manager in a temp directory
# ------------------------------------------------------------
def make_pm(tmp_path):
    class DummyAnalyzer:
        voice_type = "baritone"
        def set_user_formants(self, fmts): pass

    return ProfileManager(tmp_path, DummyAnalyzer())


# ------------------------------------------------------------
# 1) Minimal 5‑vowel calibration → NO interpolation
# ------------------------------------------------------------
def test_no_interpolation_with_minimal_calibration(tmp_path):
    pm = make_pm(tmp_path)

    # Create session with only the standard 5 vowels
    sess = CalibrationSession(
        profile_name="test",
        voice_type="baritone",
        vowels=STANDARD_VOWELS,
        profile_manager=pm,
    )

    # Inject synthetic calibrated anchors
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

    # Interpolation should produce NOTHING
    interp = sess.compute_interpolated_vowels()
    # These triangles are fully defined by the standard 5 vowels
    expected = {"e", "ɪ", "æ", "o"}

    assert set(interp.keys()) == expected

    # Save profile
    base = sess.save_profile()
    saved = json.loads(Path(tmp_path, f"{base}_profile.json").read_text())

    assert saved["interpolated_vowels"] == []

    # Calibrated vowels preserved
    assert set(saved["calibrated_vowels"]) == set(STANDARD_VOWELS)

    # Seeded vowels appear (e, ɪ, o) but are NOT interpolated
    assert "e" in saved
    assert "ɪ" in saved
    assert "o" in saved
    assert "e" not in saved["interpolated_vowels"]
    assert "ɪ" not in saved["interpolated_vowels"]
    assert "o" not in saved["interpolated_vowels"]


def test_interpolation_for_extended_vowels(tmp_path):
    pm = make_pm(tmp_path)

    # Build a session with ALL triangle vertices calibrated
    all_vertices = set()
    for tri in TRIANGLES.values():
        all_vertices |= set(tri)

    sess = CalibrationSession(
        profile_name="full",
        voice_type="baritone",
        vowels=list(all_vertices),
        profile_manager=pm,
    )

    # Give each anchor a simple numeric pattern so interpolation is predictable
    for idx, v in enumerate(sorted(all_vertices)):
        f1 = 100 + idx * 10
        f2 = 200 + idx * 20
        sess.data[v] = {
            "f1": f1, "f2": f2, "f0": 100 + idx,
            "confidence": 1.0, "stability": 0.0,
            "weight": 1.0, "saved_at": "now",
        }
        sess.calibrated_vowels.add(v)

    interp = sess.compute_interpolated_vowels()

    # Every vowel in TRIANGLES should now be interpolated
    assert set(interp.keys()) == set(TRIANGLES.keys())

    # Check barycentric math for one vowel as a representative sample
    v = next(iter(TRIANGLES.keys()))
    A, B, C = TRIANGLES[v]
    wA, wB, wC = TRIANGLE_WEIGHTS[v]

    f1A, f2A = sess.data[A]["f1"], sess.data[A]["f2"]
    f1B, f2B = sess.data[B]["f1"], sess.data[B]["f2"]
    f1C, f2C = sess.data[C]["f1"], sess.data[C]["f2"]

    expected_f1 = wA * f1A + wB * f1B + wC * f1C
    expected_f2 = wA * f2A + wB * f2B + wC * f2C

    assert abs(interp[v]["f1"] - expected_f1) < 1e-6
    assert abs(interp[v]["f2"] - expected_f2) < 1e-6


def test_interpolated_vowels_are_complement_of_calibrated(tmp_path):
    pm = make_pm(tmp_path)

    sess = CalibrationSession(
        profile_name="comp",
        voice_type="baritone",
        vowels=["i", "e", "ɛ"],
        profile_manager=pm,
    )

    # Calibrate only "i"
    sess.data["i"] = {
        "f1": 300, "f2": 2700, "f0": 150,
        "confidence": 1.0, "stability": 0.0,
        "weight": 1.0, "saved_at": "now",
    }
    sess.calibrated_vowels.add("i")

    # Add synthetic entries for "e" and "ɛ" so they appear in profile_data
    sess.data["e"] = {"f1": 400, "f2": 2000}
    sess.data["ɛ"] = {"f1": 500, "f2": 1800}

    base = sess.save_profile()
    saved = json.loads(Path(tmp_path, f"{base}_profile.json").read_text())

    all_vowels = {"i", "e", "ɛ"}
    expected_interp = sorted(all_vowels - {"i"})

    assert set(saved["interpolated_vowels"]) == set(expected_interp)


def test_seeding_does_not_override_interpolated_values(tmp_path):
    pm = make_pm(tmp_path)

    sess = CalibrationSession(
        profile_name="seed",
        voice_type="baritone",
        vowels=["i", "ɛ", "e"],
        profile_manager=pm,
    )

    # Calibrate i and ɛ
    sess.data["i"] = {"f1": 300, "f2": 2700, "f0": 150,
                      "confidence": 1.0, "stability": 0.0,
                      "weight": 1.0, "saved_at": "now"}
    sess.data["ɛ"] = {"f1": 550, "f2": 2000, "f0": 150,
                      "confidence": 1.0, "stability": 0.0,
                      "weight": 1.0, "saved_at": "now"}
    sess.calibrated_vowels.update({"i", "ɛ"})

    # Pretend interpolation produced "e"
    sess.data["e"] = {"f1": 400, "f2": 2300, "f0": 150,
                      "confidence": 1.0, "stability": 0.0,
                      "weight": 0.0, "saved_at": "now"}

    base = sess.save_profile()
    saved = json.loads(Path(tmp_path, f"{base}_profile.json").read_text())

    # Seeding should NOT overwrite the existing "e"
    assert saved["e"]["f1"] == 400
    assert saved["e"]["f2"] == 2300


def test_interpolation_excludes_calibrated_vowels(tmp_path):
    pm = make_pm(tmp_path)

    sess = CalibrationSession(
        profile_name="no_mix",
        voice_type="baritone",
        vowels=["i", "ɛ", "ɑ", "e"],
        profile_manager=pm,
    )

    # Calibrate i, ɛ, ɑ, e
    for v, (f1, f2) in {
        "i": (300, 2700),
        "ɛ": (550, 2000),
        "ɑ": (750, 1600),
        "e": (400, 2300),
    }.items():
        sess.data[v] = {"f1": f1, "f2": f2}
        sess.calibrated_vowels.add(v)

    # In-memory interpolation *will* include e — this is correct
    interp = sess.compute_interpolated_vowels()
    assert "e" in interp

    # But saved profile must NOT treat e as interpolated
    base = sess.save_profile()
    saved = json.loads(Path(tmp_path, f"{base}_profile.json").read_text())

    assert "e" not in saved["interpolated_vowels"]


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
    sess.data["ɑ"] = {"f1": None, "f2": 1600}  # invalid anchor
    sess.calibrated_vowels.update({"i", "ɛ", "ɑ"})

    interp = sess.compute_interpolated_vowels()

    # e requires (i, ɛ, ɑ) → but ɑ is invalid → skip
    assert "e" not in interp


def test_f0_interpolation_behavior(tmp_path):
    pm = make_pm(tmp_path)

    sess = CalibrationSession(
        profile_name="f0_interp",
        voice_type="baritone",
        vowels=["i", "ɛ", "ɑ"],
        profile_manager=pm,
    )

    # All F0 present → should interpolate
    sess.data["i"] = {"f1": 300, "f2": 2700, "f0": 100}
    sess.data["ɛ"] = {"f1": 550, "f2": 2000, "f0": 200}
    sess.data["ɑ"] = {"f1": 750, "f2": 1600, "f0": 300}
    sess.calibrated_vowels.update({"i", "ɛ", "ɑ"})

    interp = sess.compute_interpolated_vowels()
    assert interp["e"]["f0"] is not None

    # Remove one F0 → result must be None
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

    base = sess.save_profile()

    after = sess.data["i"]
    assert before == after  # no mutation


def test_seeded_vowels_not_in_interpolated(tmp_path):
    pm = make_pm(tmp_path)

    sess = CalibrationSession(
        profile_name="seed_test",
        voice_type="baritone",
        vowels=STANDARD_VOWELS,
        profile_manager=pm,
    )

    # Calibrate the standard 5
    for v, (f1, f2) in {
        "i": (300, 2700),
        "ɛ": (550, 2000),
        "ɑ": (750, 1600),
        "ɔ": (600, 1100),
        "u": (350, 900),
    }.items():
        sess.data[v] = {"f1": f1, "f2": f2}
        sess.calibrated_vowels.add(v)

    base = sess.save_profile()
    saved = json.loads(Path(tmp_path, f"{base}_profile.json").read_text())

    assert saved["interpolated_vowels"] == []
    for v in ["e", "ɪ", "o"]:
        assert v in saved
        assert v not in saved["interpolated_vowels"]


def test_full_expanded_interpolation(tmp_path):
    pm = make_pm(tmp_path)

    # All triangle vertices
    all_vertices = set()
    for tri in TRIANGLES.values():
        all_vertices |= set(tri)

    sess = CalibrationSession(
        profile_name="full_expanded",
        voice_type="baritone",
        vowels=list(all_vertices),
        profile_manager=pm,
    )

    # Give each anchor predictable values
    for idx, v in enumerate(sorted(all_vertices)):
        sess.data[v] = {"f1": 100 + idx, "f2": 200 + idx}
        sess.calibrated_vowels.add(v)

    interp = sess.compute_interpolated_vowels()

    assert set(interp.keys()) == set(TRIANGLES.keys())


def test_extract_formants_ignores_invalid_entries(tmp_path):
    pm = make_pm(tmp_path)

    raw = {
        "i": {"f1": 300, "f2": 2700, "confidence": 1.0, "stability": 0.0},
        "bad1": "string",
        "bad2": 123,
        "bad3": ["not", "valid"],  # list → normalized, not ignored
        "voice_type": "baritone",
    }

    fmts = pm.extract_formants(raw)

    assert "i" in fmts
    assert "bad1" not in fmts
    assert "bad2" not in fmts

    # bad3 is normalized, so it *should* appear
    assert "bad3" in fmts
    assert fmts["bad3"]["f1"] == "not"
    assert fmts["bad3"]["f2"] == "valid"


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
    fmts = pm.extract_formants(loaded)

    assert "i" in fmts
    assert fmts["i"]["f1"] == 300
    assert fmts["i"]["f2"] == 2700
