
from time import monotonic
from calibration.state_machine import CalibrationStateMachine


def test_initial_state():
    sm = CalibrationStateMachine(["a", "e"])
    assert sm.phase == "prep"
    assert sm.current_vowel == "a"
    assert sm.retry_count == 0


def test_prep_countdown():
    sm = CalibrationStateMachine(["a"], prep_seconds=2)
    out1 = sm.tick()
    assert out1["event"] == "prep_countdown"
    assert out1["secs"] == 2

    out2 = sm.tick()
    assert out2["event"] == "prep_countdown"
    assert out2["secs"] == 1

    out3 = sm.tick()
    assert out3["event"] == "start_sing"
    assert out3["vowel"] == "a"
    assert sm.phase == "sing"


def test_sing_countdown_to_capture():
    sm = CalibrationStateMachine(["a"], sing_seconds=2)
    sm.phase = "sing"
    sm.sing_secs = 2

    out1 = sm.tick()
    assert out1["event"] == "sing_countdown"
    assert out1["secs"] == 2

    out2 = sm.tick()
    assert out2["event"] == "sing_countdown"
    assert out2["secs"] == 1

    out3 = sm.tick()
    assert out3["event"] == "start_capture"
    assert sm.phase == "capture"


def test_capture_tick_and_ready():
    sm = CalibrationStateMachine(["a"], capture_seconds=2)
    sm.phase = "capture"
    sm.capture_secs = 2

    out1 = sm.tick()
    assert out1["event"] == "capture_tick"

    out2 = sm.tick()
    assert out2["event"] == "capture_tick"

    out3 = sm.tick()
    assert out3["event"] == "capture_ready"


def test_advance_moves_to_next_vowel():
    sm = CalibrationStateMachine(["a", "e"])
    out = sm.advance()
    assert out["event"] == "next_vowel"
    assert sm.current_vowel == "e"
    assert sm.retry_count == 0
    assert sm.phase == "prep"


def test_advance_finishes_after_last_vowel():
    sm = CalibrationStateMachine(["a"])
    out = sm.advance()
    assert out["event"] == "finished"
    assert sm.phase == "finished"
    assert sm.current_vowel is None


def test_retry_increments_and_restarts_phase():
    sm = CalibrationStateMachine(["a"])
    sm.phase = "capture"

    out = sm.retry_current_vowel()
    assert out["event"] == "retry"
    assert out["vowel"] == "a"
    assert sm.retry_count == 1
    assert sm.phase == "prep"


def test_retry_hits_max_retries_and_advances():
    sm = CalibrationStateMachine(["a", "e"])
    sm.retry_count = sm.MAX_RETRIES - 1
    sm.phase = "capture"

    out = sm.retry_current_vowel()
    assert out["event"] == "max_retries"
    assert out["vowel"] == "a"
    assert out["advance"]["event"] == "next_vowel"
    assert sm.current_vowel == "e"
    assert sm.retry_count == 0


def test_check_timeout_true():
    sm = CalibrationStateMachine(["a"])
    sm.phase = "capture"
    sm.capture_start_time = monotonic() - 10

    assert sm.check_timeout(5.0) is True


def test_check_timeout_false_when_not_capture():
    sm = CalibrationStateMachine(["a"])
    sm.phase = "prep"
    assert sm.check_timeout(5.0) is False


def test_check_timeout_false_when_no_start_time():
    sm = CalibrationStateMachine(["a"])
    sm.phase = "capture"
    sm.capture_start_time = None
    assert sm.check_timeout(5.0) is False


def test_force_capture_mode():
    sm = CalibrationStateMachine(["a"])
    sm.force_capture_mode()
    assert sm.phase == "capture"
    assert sm.capture_start_time is not None


def test_is_done():
    sm = CalibrationStateMachine(["a"])
    assert sm.is_done() is False
    sm.phase = "finished"
    assert sm.is_done() is True
