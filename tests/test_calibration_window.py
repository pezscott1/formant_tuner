# tests/test_calibration_window.py
import numpy as np
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def window(qtbot):
    """Create a CalibrationWindow with all heavy components mocked."""
    with patch("calibration.window.CalibrationSession") as MockSession, \
         patch("calibration.window.CalibrationStateMachine") as MockState, \
         patch("calibration.window.QTimer") as MockTimer:

        # Fake timer
        fake_timer = MagicMock()
        fake_timer.start = MagicMock()
        fake_timer.stop = MagicMock()
        fake_timer.timeout = MagicMock()
        fake_timer.timeout.connect = MagicMock()
        MockTimer.return_value = fake_timer

        # Mock state machine
        state = MockState.return_value
        state.current_vowel = "a"
        state.tick.return_value = {"event": "prep_countdown", "secs": 3}
        state.advance.return_value = {"event": "none"}
        state.check_timeout.return_value = False

        # Mock session
        session = MockSession.return_value
        session.handle_result.return_value = (True, False, "OK")
        session.save_profile.return_value = "profile_base"
        session.is_complete.return_value = False
        session.current_index = 1
        session.vowels = ["a", "e", "i", "o", "u"]

        # Construct window
        from calibration.window import CalibrationWindow
        win = CalibrationWindow("test_profile", voice_type="bass")
        qtbot.addWidget(win)

        return win, session, state


# ---------------------------------------------------------
# Test _poll_audio
# ---------------------------------------------------------

@patch(
    "calibration.window.safe_spectrogram",
    return_value=(
        np.array([100, 200]),          # freqs
        np.array([0.1, 0.2]),          # times
        np.array([[1, 1], [1, 1]]),    # S: 2x2
    ),
)
@patch("calibration.window.update_artists")
def test_poll_audio_processes_frame(mock_update, mock_spec, window):
    win, session, state = window

    win.engine = MagicMock()
    win.engine.get_latest_raw.return_value = {
        "f0": 120,
        "formants": (500, 1500, 2500),
        "segment": np.ones(512),
    }

    win._poll_audio()

    mock_update.assert_called_once()
    mock_spec.assert_called_once()
# ---------------------------------------------------------
# Test _process_capture
# ---------------------------------------------------------


def test_process_capture_stores_results(window):
    win, session, state = window

    win._last_frame = (500, 1500, 120)
    win._process_capture()

    session.handle_result.assert_called_once_with(500, 1500, 120)
    assert "/a/" in win.capture_panel.toPlainText()


def test_process_capture_no_audio(window):
    win, session, state = window

    if hasattr(win, "_last_frame"):
        del win._last_frame

    win._process_capture()

    assert "No audio captured" in win.status_panel.toPlainText()


# ---------------------------------------------------------
# Test _finish
# ---------------------------------------------------------

def test_finish_saves_profile_and_emits_signal(window, qtbot):
    win, session, state = window

    with qtbot.waitSignal(win.profile_calibrated, timeout=1000) as blocker:
        win._finish()

    session.save_profile.assert_called_once()
    assert blocker.args == ["profile_base"]
