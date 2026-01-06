# tests/calibration/test_calibration_session_handle_result.py
import pytest
from calibration.session import CalibrationSession


def make_session(existing_profile=None):
    return CalibrationSession(
        profile_name="test",
        voice_type="bass",
        vowels=["a"],
        profile_manager=None,
        existing_profile=existing_profile,
    )


@pytest.fixture
def always_plausible(monkeypatch):
    def _impl(f1, f2, voice_type=None, vowel=None, calibrated=None):
        return True, "ok"
    monkeypatch.setattr(
        "calibration.session.is_plausible_formants",
        _impl,
        raising=True,
    )
    return _impl


@pytest.fixture
def always_implausible(monkeypatch):
    def _impl(f1, f2, voice_type=None, vowel=None, calibrated=None):
        return False, "implausible"
    monkeypatch.setattr(
        "calibration.session.is_plausible_formants",
        _impl,
        raising=True,
    )
    return _impl


def test_handle_result_accepts_first_valid_measurement(always_plausible):
    s = make_session()

    accepted, retry, msg = s.handle_result(
        "a", f1=500.0, f2=1500.0, f0=120.0, confidence=0.8, stability=0.1
    )

    assert accepted is True
    assert retry is False
    assert "accepted first measurement" in msg.lower()

    assert "a" in s.data
    entry = s.data["a"]
    assert entry["f1"] == 500.0
    assert entry["f2"] == 1500.0
    assert entry["f0"] == 120.0
    assert entry["confidence"] == pytest.approx(0.8)
    assert entry["stability"] == pytest.approx(0.1)
    assert entry["weight"] == 1.0
    assert "saved_at" in entry

    # No retries incremented for successful capture
    assert s.retry_count("a") == 0


def test_handle_result_rejects_invalid_formants(always_plausible):
    s = make_session()

    accepted, retry, msg = s.handle_result(
        "a", f1=None, f2=1500.0, f0=120.0, confidence=0.8, stability=0.1
    )

    assert accepted is False
    assert retry is True
    assert "invalid formant values" in msg.lower()
    assert "a" not in s.data
    assert s.retry_count("a") == 1


def test_handle_result_rejects_implausible(always_implausible):
    s = make_session()

    accepted, retry, msg = s.handle_result(
        "a", f1=500.0, f2=1500.0, f0=120.0, confidence=0.8, stability=0.1
    )

    assert accepted is False
    assert retry is True
    assert "implausible" in msg.lower()
    assert "a" not in s.data
    assert s.retry_count("a") == 1


def test_handle_result_rejects_low_confidence(always_plausible):
    s = make_session()

    accepted, retry, msg = s.handle_result(
        "a", f1=500.0, f2=1500.0, f0=120.0, confidence=0.1, stability=0.1
    )

    assert accepted is False
    assert retry is True
    assert "low confidence" in msg.lower()
    assert "a" not in s.data
    assert s.retry_count("a") == 1


def test_handle_result_updates_existing_with_weighted_average(always_plausible):
    # Seed with an existing measurement
    existing = {
        "a": {
            "f1": 400.0,
            "f2": 1400.0,
            "f0": 100.0,
            "confidence": 0.6,
            "stability": 0.2,
            "weight": 1.0,
            "saved_at": "2025-01-01T00:00:00Z",
        }
    }
    s = make_session(existing_profile=existing)

    accepted, retry, msg = s.handle_result(
        "a", f1=600.0, f2=1600.0, f0=120.0, confidence=1.0, stability=0.0
    )

    assert accepted is True
    assert retry is False
    assert "updated /a/" in msg.lower()

    entry = s.data["a"]
    # Weight should increment
    assert entry["weight"] == pytest.approx(2.0)

    # F1/F2 should be the mean of old and new with equal weights
    assert entry["f1"] == pytest.approx((400.0 + 600.0) / 2.0)
    assert entry["f2"] == pytest.approx((1400.0 + 1600.0) / 2.0)

    # F0 also averaged
    assert entry["f0"] == pytest.approx((100.0 + 120.0) / 2.0)

    # Confidence/stability averaged
    assert entry["confidence"] == pytest.approx((0.6 + 1.0) / 2.0)
    assert entry["stability"] == pytest.approx((0.2 + 0.0) / 2.0)

    # Retry count unchanged for success
    assert s.retry_count("a") == 0


def test_retry_helpers_increment_and_reset():
    s = make_session()
    assert s.retry_count("a") == 0

    s.increment_retry("a")
    s.increment_retry("a")
    assert s.retry_count("a") == 2

    s.reset_retry("a")
    assert s.retry_count("a") == 0
