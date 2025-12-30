import time
from calibration.state_machine import CalibrationStateMachine


def test_initial_state():
    sm = CalibrationStateMachine(["a", "i"])
    assert sm.phase == "prep"
    assert sm.current_vowel == "a"
    assert sm.prep_secs == 3
    assert sm.sing_secs == 2
    assert sm.capture_secs == 2


def test_prep_phase_counts_down_then_transitions_to_sing():
    sm = CalibrationStateMachine(["a"], prep_seconds=2, sing_seconds=2)

    # First tick → prep_countdown
    out = sm.tick()
    assert out["event"] == "prep_countdown"
    assert out["secs"] == 2

    # Second tick → prep_countdown
    out = sm.tick()
    assert out["event"] == "prep_countdown"
    assert out["secs"] == 1

    # Third tick → start_sing
    out = sm.tick()
    assert out["event"] == "start_sing"
    assert out["vowel"] == "a"
    assert sm.phase == "sing"


def test_sing_phase_counts_down_then_transitions_to_capture():
    sm = CalibrationStateMachine(["a"], prep_seconds=0, sing_seconds=2)

    # Immediately in sing phase
    sm.phase = "sing"

    out = sm.tick()
    assert out["event"] == "sing_countdown"
    assert out["secs"] == 2

    out = sm.tick()
    assert out["event"] == "sing_countdown"
    assert out["secs"] == 1

    out = sm.tick()
    assert out["event"] == "start_capture"
    assert sm.phase == "capture"
    assert sm.capture_secs == 2


def test_capture_phase_counts_down_then_returns_ready():
    sm = CalibrationStateMachine(
        ["a"], prep_seconds=0, sing_seconds=0, capture_seconds=2)
    sm.phase = "capture"

    out = sm.tick()
    assert out["event"] == "capture_tick"

    out = sm.tick()
    assert out["event"] == "capture_tick"

    out = sm.tick()
    assert out["event"] == "capture_ready"


def test_advance_moves_to_next_vowel_and_resets_timers():
    sm = CalibrationStateMachine(
        ["a", "i"], prep_seconds=3, sing_seconds=2, capture_seconds=2)

    sm.phase = "capture"
    sm.index = 0

    out = sm.advance()
    assert out["event"] == "next_vowel"
    assert out["vowel"] == "i"
    assert sm.phase == "prep"
    assert sm.prep_secs == 3
    assert sm.sing_secs == 2
    assert sm.capture_secs == 2
    assert sm.current_vowel == "i"


def test_advance_finishes_after_last_vowel():
    sm = CalibrationStateMachine(["a"], prep_seconds=1)
    sm.index = 0

    out = sm.advance()
    assert out["event"] == "finished"
    assert sm.phase == "finished"
    assert sm.current_vowel is None


def test_retry_current_vowel_normal_retry():
    sm = CalibrationStateMachine(["a"])
    sm.retry_count = 0

    out = sm.retry_current_vowel()
    assert out["event"] == "retry"
    assert out["vowel"] == "a"
    assert sm.phase == "prep"
    assert sm.retry_count == 1


def test_retry_current_vowel_max_retries():
    sm = CalibrationStateMachine(["a"])
    sm.retry_count = sm.MAX_RETRIES - 1

    out = sm.retry_current_vowel()
    assert out["event"] == "max_retries"
    assert out["vowel"] == "a"
    assert sm.retry_count == sm.MAX_RETRIES


def test_force_capture_mode():
    sm = CalibrationStateMachine(["a"])
    sm.force_capture_mode()
    assert sm.phase == "capture"
    assert sm.capture_start_time is not None


def test_check_timeout_true():
    sm = CalibrationStateMachine(["a"], capture_seconds=1)
    sm.phase = "capture"
    sm.capture_start_time = time.monotonic() - 5  # simulate long delay

    assert sm.check_timeout(1.0) is True


def test_check_timeout_false_if_not_capture():
    sm = CalibrationStateMachine(["a"])
    sm.phase = "prep"
    assert sm.check_timeout(1.0) is False


def test_is_done():
    sm = CalibrationStateMachine(["a"])
    sm.phase = "finished"
    assert sm.is_done() is True
