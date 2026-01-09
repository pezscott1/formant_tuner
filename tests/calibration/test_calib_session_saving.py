# tests/calibration/test_profile_saving_fixes.py
import json
from pathlib import Path
from calibration.session import CalibrationSession
from tuner.profile_controller import ProfileManager


def make_pm(tmp_path):
    class DummyAnalyzer:
        voice_type = "baritone"
        def set_user_formants(self, fmts): pass

    return ProfileManager(tmp_path, DummyAnalyzer())


def test_save_profile_interpolated_sorted_and_separated(tmp_path):
    pm = make_pm(tmp_path)

    sess = CalibrationSession(
        profile_name="test",
        voice_type="baritone",
        vowels=["i", "e", "ɛ"],
        profile_manager=pm,
    )

    # Provide raw formant entries but only mark "i" as explicitly calibrated.
    sess.data["i"] = {"f1": 300, "f2": 2700}
    sess.data["e"] = {"f1": 400, "f2": 2000}
    sess.data["ɛ"] = {"f1": 500, "f2": 1800}

    sess.calibrated_vowels.add("i")

    # Compute interpolation explicitly and use it as the expected source.
    interp = sess.compute_interpolated_vowels()

    base = sess.save_profile()
    saved = json.loads(Path(tmp_path, f"{base}_profile.json").read_text())

    # Calibrated vowels must be exactly the explicit set
    assert list(saved["calibrated_vowels"].keys()) == ["i"]

    # Interpolated vowels must match the triangle-based computation
    assert list(saved["interpolated_vowels"].keys()) == sorted(interp.keys())
