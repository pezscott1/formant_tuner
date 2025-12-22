# tests/test_calibration_state_machine.py
from calibration.state_machine import CalibrationStateMachine


def test_initial_state_is_prep():
    sm = CalibrationStateMachine(["a"])
    assert sm.phase == "prep"
    assert sm.current_vowel == "a"


def test_prep_transitions_to_sing():
    sm = CalibrationStateMachine(["a"])
    sm.prep_secs = 0  # force countdown to expire

    event = sm.tick()
    assert event["event"] == "start_sing"
    assert sm.phase == "sing"


def test_sing_transitions_to_capture():
    sm = CalibrationStateMachine(["a"])
    sm.phase = "sing"
    sm.sing_secs = 0  # force countdown to expire

    event = sm.tick()
    assert event["event"] == "start_capture"
    assert sm.phase == "capture"


def test_capture_transitions_to_capture_ready():
    sm = CalibrationStateMachine(["a", "e"])
    sm.phase = "capture"
    sm.capture_secs = 0  # force immediate readiness

    event = sm.tick()
    assert event["event"] == "capture_ready"
    assert sm.phase == "capture"  # phase stays until advance()


def test_advance_moves_to_next_vowel():
    sm = CalibrationStateMachine(["a", "e"])
    sm.phase = "capture"

    # Simulate capture ready
    sm.capture_secs = 0
    sm.tick()

    event = sm.advance()
    assert event["event"] == "next_vowel"
    assert sm.phase == "prep"
    assert sm.current_vowel == "e"


def test_advance_finishes_on_last_vowel():
    sm = CalibrationStateMachine(["a"])
    sm.phase = "capture"

    # Simulate capture ready
    sm.capture_secs = 0
    sm.tick()

    event = sm.advance()
    assert event["event"] == "finished"
    assert sm.phase == "finished"
    assert sm.is_done()
