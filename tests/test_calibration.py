# tests/test_calibration.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from calibration import CalibrationWindow
from PyQt5.QtWidgets import QMessageBox


@pytest.fixture
def fake_mic():
    """Fixture: fake microphone with sample rate and buffer."""
    mic = MagicMock()
    mic.sample_rate = 16000
    mic.buffer = [0.0] * 16000
    mic.start = MagicMock()
    mic.stop = MagicMock()
    return mic


def test_process_capture_accepts_and_saves(tmp_path, qtbot, fake_mic):
    """Test that process_capture runs without crashing and saves results."""
    analyzer = MagicMock()
    win = CalibrationWindow(analyzer, "testuser", "tenor", mic=fake_mic)
    qtbot.addWidget(win)  # register with pytest-qt

    # inject a fake capture buffer with numeric arrays
    win.capture_buffer = [np.ones(1600), np.ones(1600) * 0.1]

    # patch estimate_formants_lpc to return plausible values
    with patch(
        "formant_utils.estimate_formants_lpc",
        return_value=(300.0, 2300.0, 220.0),
    ):
        win.process_capture()
        assert isinstance(win.results, dict)
        assert "dummy" not in win.results  # sanity check


def test_finish_handles_save_error(tmp_path, qtbot, fake_mic):
    """Test that finish() handles save errors gracefully."""
    analyzer = MagicMock()
    win = CalibrationWindow(analyzer, "userX", "bass", mic=fake_mic)
    qtbot.addWidget(win)

    # set up minimal state
    win.results = {"dummy": 1}
    win.retries_map = {}
    win.profile_name = "userX"
    win.voice_type = "bass"

    # Patch open to raise and QMessageBox.critical to capture call
    with patch("builtins.open", side_effect=OSError("disk error")), \
         patch.object(QMessageBox, "critical") as mock_critical:
        win.finish()
        mock_critical.assert_called_once()


def test_calibrationwindow_init_sets_defaults(qtbot):
    from calibration import CalibrationWindow
    fake_analyzer = MagicMock()
    fake_mic = MagicMock()
    win = CalibrationWindow(fake_analyzer, "user", "tenor", mic=fake_mic)
    qtbot.addWidget(win)
    assert win.profile_name == "user"
    assert win.voice_type == "tenor"
    assert isinstance(win.results, dict)
    assert win.current_index == 0


def test_process_capture_with_pending_frames(qtbot):
    from calibration import CalibrationWindow
    fake_analyzer = MagicMock()
    fake_mic = MagicMock()
    win = CalibrationWindow(fake_analyzer, "user", "tenor", mic=fake_mic)
    qtbot.addWidget(win)
    win._pending_frames.append(np.ones(10))
    win.process_capture()
    # Should append status text about audio queued
    assert "audio queued" in win.status_panel.toPlainText()


def test_process_capture_timeout(qtbot, monkeypatch):
    from calibration import CalibrationWindow
    fake_analyzer = MagicMock()
    fake_mic = MagicMock()
    win = CalibrationWindow(fake_analyzer, "user", "tenor", mic=fake_mic)
    qtbot.addWidget(win)
    win.capture_start_time = 0  # force elapsed > timeout
    win.capture_timeout = -1
    win.process_capture()
    assert "capture timed out" in win.status_panel.toPlainText()


def test_finish_success_and_failure(qtbot, tmp_path, monkeypatch):
    from calibration import CalibrationWindow
    fake_analyzer = MagicMock()
    fake_mic = MagicMock()
    win = CalibrationWindow(fake_analyzer, "user", "tenor", mic=fake_mic)
    qtbot.addWidget(win)
    win.results = {"a": (500, 1500, 200)}
    win.retries_map = {"a": 0}

    # Patch PROFILES_DIR to tmp_path
    monkeypatch.setattr("calibration.PROFILES_DIR", str(tmp_path))

    # Success path
    win.finish()
    profile_file = tmp_path / "user_tenor_profile.json"
    assert profile_file.exists()

    # Failure path: patch open to raise
    monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError("disk error")))
    with patch("calibration.QMessageBox.critical") as mock_critical:
        win._finished = False  # reset flag
        win.finish()
        mock_critical.assert_called_once()


def test_on_result_ready_and_apply_compute_result(qtbot):
    from calibration import CalibrationWindow
    fake_analyzer = MagicMock()
    fake_mic = MagicMock()
    win = CalibrationWindow(fake_analyzer, "user", "tenor", mic=fake_mic)
    qtbot.addWidget(win)
    # Valid result
    result = (np.array([100,200]), np.array([0.1,0.2]), np.ones((2,2)), 500, 1500, 250)
    win._on_result_ready(result)
    assert "Calibration complete!" not in win.status_panel.toPlainText()
    # Invalid result triggers retry
    bad_result = (np.array([100]), np.array([0.1]), np.ones((1,1)), None, None, None)
    win._on_result_ready(bad_result)
    assert "retry" in win.status_panel.toPlainText()
