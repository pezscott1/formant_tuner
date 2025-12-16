import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from calibration import CalibrationWindow


@pytest.fixture
def fake_mic():
    """Fixture: fake microphone with sample rate and buffer."""
    mic = MagicMock()
    mic.sample_rate = 16000
    mic.buffer = [0.0] * 16000
    mic.start = MagicMock()
    mic.stop = MagicMock()
    return mic


def test_process_capture_accepts_and_saves(tmp_path, _qtbot, fake_mic):
    """Test that process_capture runs without crashing and saves results."""
    analyzer = MagicMock()
    win = CalibrationWindow(analyzer, "testuser", "tenor", mic=fake_mic)

    # inject a fake capture buffer with numeric arrays
    win.capture_buffer = [np.ones(1600), np.ones(1600) * 0.1]

    # patch estimate_formants_lpc to return plausible values
    with patch("formant_utils.estimate_formants_lpc", return_value=(300.0, 2300.0, 220.0)):
        win.process_capture()
        assert isinstance(win.results, dict)
