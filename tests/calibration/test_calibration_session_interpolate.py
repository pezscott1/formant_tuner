# tests/calibration/test_calibration_interpolation.py

import pytest
from calibration.session import CalibrationSession


@pytest.fixture
def simple_triangles(monkeypatch):
    # Define a simple triangle: x is inside the triangle formed by a, e, i
    triangles = {
        "x": ("a", "e", "i"),
    }
    weights = {
        "x": (1.0 / 3, 1.0 / 3, 1.0 / 3),
    }

    monkeypatch.setattr("calibration.session.TRIANGLES", triangles, raising=True)
    monkeypatch.setattr("calibration.session.TRIANGLE_WEIGHTS", weights, raising=True)


def test_compute_interpolated_vowels_basic(simple_triangles):
    # Set anchors for a, e, i
    s = CalibrationSession(
        profile_name="test",
        voice_type="bass",
        vowels=["a", "e", "i"],
        profile_manager=None,
        existing_profile=None,
    )

    s.data = {
        "a": {"f1": 300.0, "f2": 900.0, "f0": 100.0},
        "e": {"f1": 500.0, "f2": 1500.0, "f0": 110.0},
        "i": {"f1": 700.0, "f2": 2100.0, "f0": 120.0},
    }
    s.calibrated_vowels.update({"a", "e", "i"})
    out = s.compute_interpolated_vowels()
    assert "x" in out

    x = out["x"]
    # F1/F2/F0 should be simple averages with equal weights
    assert x["f1"] == pytest.approx((300.0 + 500.0 + 700.0) / 3.0)
    assert x["f2"] == pytest.approx((900.0 + 1500.0 + 2100.0) / 3.0)
    assert x["f0"] == pytest.approx((100.0 + 110.0 + 120.0) / 3.0)

    assert x["confidence"] == 1.0
    assert x["stability"] == 0.0
    assert x["weight"] == 0.0
    assert "saved_at" in x


def test_compute_interpolated_vowels_skips_when_missing_anchor(simple_triangles):
    s = CalibrationSession(
        profile_name="test",
        voice_type="bass",
        vowels=["a", "e", "i"],
        profile_manager=None,
        existing_profile=None,
    )

    # Only two anchors present → triangle condition fails
    s.data = {
        "a": {"f1": 300.0, "f2": 900.0, "f0": 100.0},
        "e": {"f1": 500.0, "f2": 1500.0, "f0": 110.0},
    }

    out = s.compute_interpolated_vowels()
    # Should not crash, but also not produce x
    assert "x" not in out
    assert isinstance(out, dict)
