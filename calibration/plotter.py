# calibration/plotter.py
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )


def safe_spectrogram(y, sr, n_fft=1024, hop_length=256, window_seconds=1.0):
    """
    Clean, speech-optimized spectrogram:
      - pre-emphasis
      - librosa STFT (fallback FFT)
      - rolling window
      - stable for 48 kHz mics
    """
    if y is None or len(y) == 0:
        f = np.linspace(0, sr / 2, 128)
        t = np.array([0.0])
        Sxx = np.zeros((f.size, t.size))
        return f, t, Sxx

    # Pre-emphasis
    try:
        y = np.asarray(y, dtype=float)
        if len(y) > 1:
            y = np.append(y[0], y[1:] - 0.95 * y[:-1])
    except Exception:
        pass

    # Try librosa STFT
    try:
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        times = librosa.frames_to_time(
            np.arange(S.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft
        )
    except Exception as e:
        logger.debug("librosa spectrogram failed: %s", e)

        # Fallback FFT
        try:
            win = np.hanning(n_fft)
            frames = []
            step = hop_length

            for i in range(0, len(y) - n_fft + 1, step):
                frame = y[i:i+n_fft] * win
                spec = np.abs(np.fft.rfft(frame)) ** 2
                frames.append(spec)

            if frames:
                S = np.column_stack(frames)
            else:
                S = np.zeros((n_fft // 2 + 1, 1))

            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            times = np.arange(S.shape[1]) * (hop_length / float(sr))
            times = np.asarray(times).reshape(-1)
        except Exception as e2:
            logger.debug("fft spectrogram failed: %s", e2)
            f = np.linspace(0, sr / 2, 128)
            t = np.array([0.0])
            Sxx = np.zeros((f.size, t.size))
            return f, t, Sxx

    # Rolling window
    if times.size > 1 and window_seconds is not None:
        t_end = float(times[-1])
        t_start = max(0.0, t_end - window_seconds)
        keep = times >= t_start

        if keep.sum() > 1:
            times = times[keep]
            S = S[:, keep]

    return freqs, times, S


def update_spectrogram(self, freqs, times, S):
    """
    Draw ONLY the spectrogram.
    Vowel anchors are handled by CalibrationWindow.
    """
    if freqs is None or times is None or S is None:
        return
    freqs = np.asarray(freqs)
    # Limit to 4 kHz
    mask = freqs <= 4000
    if mask.sum() < 2:
        mask = np.arange(len(freqs))

    S_small = S[mask, :]
    freqs_small = freqs[mask]

    # dB scaling
    arr_db = 10 * np.log10(S_small + 1e-12)
    arr_db_max = np.max(arr_db)
    db_floor = arr_db_max - 60
    arr_db = np.clip(arr_db, db_floor, arr_db_max)

    # Draw
    self.ax_spec.clear()
    mesh = self.ax_spec.pcolormesh(
        times,
        freqs_small,
        arr_db,
        shading="auto",
        cmap="magma",
    )

    # Colorbar
    if not hasattr(self, "_spec_colorbar") or self._spec_colorbar is None:
        self._spec_colorbar = self.ax_spec.figure.colorbar(
            mesh, ax=self.ax_spec, fraction=0.046, pad=0.04
        )
    else:
        self._spec_colorbar.update_normal(mesh)

    self.ax_spec.set_ylim(0, 4000)
    self.ax_spec.set_xlabel("Time (s)")
    self.ax_spec.set_ylabel("Frequency (Hz)")
    self.ax_spec.set_title("Spectrogram")

    self.canvas.draw_idle()
