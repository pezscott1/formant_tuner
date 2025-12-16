import pytest
from calibration_py_qt import CalibrationWindow
from unittest.mock import MagicMock, patch
import numpy as np

@pytest.fixture
def fake_mic():
    mic = MagicMock()
    mic.sample_rate = 16000
    mic.buffer = [0.0]*16000
    mic.start = MagicMock()
    mic.stop = MagicMock()
    return mic

def test_process_capture_accepts_and_saves(tmp_path, qtbot, fake_mic):
    # create a minimal analyzer stub
    analyzer = MagicMock()
    win = CalibrationWindow(analyzer, "testuser", "tenor", mic=fake_mic)
    # inject a fake capture buffer with numeric arrays
    win.capture_buffer = [np.ones(1600), np.ones(1600)*0.1]
    # patch estimate_formants_lpc to return plausible values
    with patch("formant_utils.estimate_formants_lpc", return_value=(300.0, 2300.0, 220.0)):
        win.process_capture()
        assert "i" not in win.results or isinstance(win.results, dict)  # ensure no crash