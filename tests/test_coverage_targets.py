# tests/test_coverage_targets.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PyQt5.QtWidgets import QMessageBox
from formant_utils import estimate_formants_lpc, pick_formants
from formant_tuner import FormantTunerApp
from calibration import CalibrationWindow
# --- calibration.py ---


@pytest.mark.parametrize("buffer", [
    ([], {}),  # empty buffer
    ([np.ones(1600)], {"dummy": 1}),  # fake single frame
])
def test_process_capture_handles_buffers(buffer, qtbot):
    fake_analyzer = MagicMock()
    fake_mic = MagicMock()
    win = CalibrationWindow(fake_analyzer, "user", "bass", mic=fake_mic)
    qtbot.addWidget(win)
    win.capture_buffer = buffer
    with patch("formant_utils.estimate_formants_lpc", return_value=(300, 2300, 220)):
        win.process_capture()
    # Should not crash regardless of buffer
    assert isinstance(win.results, dict)


@pytest.mark.parametrize("open_side_effect,expect_warning", [
    (None, False),  # normal save
    (OSError("disk error"), True),  # save fails
])
def test_finish_handles_save_errors(open_side_effect, expect_warning, qtbot, tmp_path):
    fake_analyzer = MagicMock()
    fake_mic = MagicMock()
    win = CalibrationWindow(fake_analyzer, "user", "bass", mic=fake_mic)
    qtbot.addWidget(win)
    win.results = {"dummy": 1}
    win.profile_name = "user"
    win.voice_type = "bass"
    with patch("builtins.open", side_effect=open_side_effect), \
         patch.object(QMessageBox, "critical") as mock_critical:
        win.finish()
        if expect_warning:
            mock_critical.assert_called_once()
        else:
            mock_critical.assert_not_called()

# --- formant_tuner.py ---


@pytest.mark.parametrize("method", ["apply_selected_profile", "delete_profile"])
def test_profile_methods_warn_on_none_selected(method, qtbot):
    fake_analyzer = MagicMock()
    fake_analyzer.voice_type = "bass"
    win = FormantTunerApp(fake_analyzer)
    qtbot.addWidget(win)
    win.profile_list.clear()
    with patch.object(QMessageBox, "warning") as mock_warn:
        getattr(win, method)()
        mock_warn.assert_called()

# --- formant_utils.py ---


@pytest.mark.parametrize("signal", [
    np.array([]),  # empty
    np.random.randn(50),  # short noisy
])
def test_estimate_formants_edge_cases(signal):
    # Should not crash, may return None or tuple
    result = estimate_formants_lpc(signal, 16000)
    assert result is None or isinstance(result, tuple)


@pytest.mark.parametrize("candidates", [
    [],  # no candidates
    [500, 1500],  # minimal
])
def test_pick_formants_handles_few_candidates(candidates):
    result = pick_formants(candidates)
    # Should return a tuple even if candidates are few
    assert isinstance(result, tuple)

# --- mic_analyzer.py ---


def test_micanalyzer_start_and_stop(monkeypatch):
    from mic_analyzer import MicAnalyzer

    fake_mic = MagicMock()
    fake_mic.sample_rate = 16000
    fake_tol_provider = MagicMock()
    fake_pitch_provider = MagicMock()

    # Patch sounddevice.InputStream to avoid real audio hardware
    class DummyStream:
        def __init__(self, **kwargs):
            self.active = True

        def start(self): pass
        def stop(self): self.active = False
        def close(self): pass

    monkeypatch.setattr("mic_analyzer.sd.InputStream", DummyStream)

    analyzer = MicAnalyzer(fake_mic, fake_tol_provider, fake_pitch_provider)

    analyzer.start()
    assert analyzer.is_running is True
    assert isinstance(analyzer.stream, DummyStream)

    analyzer.stop()
    assert analyzer.is_running is False
    assert analyzer.stream is None
