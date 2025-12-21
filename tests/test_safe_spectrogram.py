# tests/test_safe_spectrogram.py
import numpy as np
from unittest.mock import patch, MagicMock

from calibration.plotter import safe_spectrogram


# ---------------------------------------------------------
# Empty input
# ---------------------------------------------------------

def test_safe_spectrogram_empty():
    f, t, S = safe_spectrogram([], sr=16000)

    assert f.size == 128
    assert t.size == 1
    assert S.shape == (128, 1)
    assert np.all(S == 0)


# ---------------------------------------------------------
# Too-short input
# ---------------------------------------------------------

def test_safe_spectrogram_too_short():
    y = np.random.randn(100)  # shorter than n_fft=2048
    f, t, S = safe_spectrogram(y, sr=16000)

    assert f.size >= 64
    assert t.size == 1
    assert S.shape[1] == 1


# ---------------------------------------------------------
# Normal librosa path
# ---------------------------------------------------------

@patch("calibration.plotter.librosa.stft")
@patch("calibration.plotter.librosa.fft_frequencies")
@patch("calibration.plotter.librosa.frames_to_time")
def test_safe_spectrogram_librosa_success(mock_frames, mock_freqs, mock_stft):
    # Fake STFT output
    mock_stft.return_value = np.abs(np.random.randn(1025, 10)) ** 2
    mock_freqs.return_value = np.linspace(0, 8000, 1025)
    mock_frames.return_value = np.linspace(0, 1, 10)

    y = np.random.randn(5000)
    f, t, S = safe_spectrogram(y, sr=16000)

    assert S.shape == (1025, 10)
    assert f.size == 1025
    assert t.size == 10


# ---------------------------------------------------------
# librosa failure → FFT fallback
# ---------------------------------------------------------

@patch("calibration.plotter.librosa.stft", side_effect=Exception("librosa fail"))
def test_safe_spectrogram_fft_fallback(mock_stft):
    y = np.random.randn(5000)

    f, t, S = safe_spectrogram(y, sr=16000)

    # FFT fallback produces rfft frames
    assert S.shape[0] == 2048 // 2 + 1
    assert t.size == S.shape[1]
    assert f.size == S.shape[0]


# ---------------------------------------------------------
# FFT failure → zero fallback
# ---------------------------------------------------------

@patch("calibration.plotter.librosa.stft", side_effect=Exception("librosa fail"))
@patch("calibration.plotter.np.fft.rfft", side_effect=Exception("fft fail"))
def test_safe_spectrogram_total_failure(mock_fft, mock_stft):
    y = np.random.randn(5000)

    f, t, S = safe_spectrogram(y, sr=16000)

    assert f.size == 128
    assert t.size == 1
    assert S.shape == (128, 1)
    assert np.all(S == 0)
