# calibration/plotter.py
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter
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


def update_spectrogram(self, freqs, times, s):
    """
    Calibration spectrogram tuned to match PyQt:
    - log-frequency resampling
    - per-column dB normalization
    - crisp harmonics across full range
    """

    if freqs is None or times is None or s is None:
        return

    # Limit to 50–4000 Hz
    freqs: np.ndarray
    mask = (freqs >= 150) & (freqs <= 4000)
    if mask.sum() < 2:
        return

    S = s[mask, :]
    freqs_small = freqs[mask]

    # Log-frequency resampling
    num_bins = 512
    log_freqs = np.logspace(np.log10(150), np.log10(4000), num_bins)
    S_log = np.zeros((num_bins, S.shape[1]))
    for i in range(S.shape[1]):
        S_log[:, i] = np.interp(log_freqs, freqs_small, S[:, i])
    S = S_log
    freqs_small = log_freqs

    # Per-column dB normalization
    arr_db = 10 * np.log10(S + 1e-12)
    arr_db -= np.max(arr_db, axis=0, keepdims=True)
    arr_db = np.clip(arr_db, -60, 0)

    # Compute frequency and time edges for pcolormesh
    freq_edges = np.zeros(len(freqs_small) + 1)
    freq_edges[1:-1] = (freqs_small[:-1] + freqs_small[1:]) / 2
    freq_edges[0] = freqs_small[0] - (freqs_small[1] - freqs_small[0]) / 2
    freq_edges[-1] = freqs_small[-1] + (freqs_small[-1] - freqs_small[-2]) / 2

    time_edges = np.zeros(len(times) + 1)
    time_edges[1:-1] = (times[:-1] + times[1:]) / 2
    time_edges[0] = times[0] - (times[1] - times[0]) / 2
    time_edges[-1] = times[-1] + (times[-1] - times[-2]) / 2

    # Draw spectrogram
    self.ax_spec.clear()
    mesh = self.ax_spec.pcolormesh(
        time_edges,
        freq_edges,
        arr_db,
        shading="auto",
        cmap="magma",
        vmin=-60,
        vmax=0,
    )

    self.ax_spec.set_yscale("log")
    self.ax_spec.set_ylim(150, 4000)
    # Log-locator for correct spacing
    self.ax_spec.yaxis.set_major_locator(LogLocator(base=10, subs=[1, 2, 4, 8]))
    # Clean labels
    self.ax_spec.yaxis.set_major_formatter(ScalarFormatter())
    self.ax_spec.yaxis.set_minor_formatter(NullFormatter())
    # Explicit ticks
    self.ax_spec.set_yticks([150, 300, 600, 1200, 2400, 4000])
    self.ax_spec.set_ylabel("Frequency (Hz)")
    self.ax_spec.set_xlabel("Time (s)")
    self.ax_spec.set_title("Spectrogram (log-frequency, PyQt-matched)")

    # Colorbar
    if not hasattr(self, "_spec_colorbar") or self._spec_colorbar is None:
        self._spec_colorbar = self.ax_spec.figure.colorbar(
            mesh, ax=self.ax_spec, fraction=0.046, pad=0.04
        )
    else:
        self._spec_colorbar.update_normal(mesh)
    # Match tuner window: always show full 3-second window
    self.ax_spec.set_xlim(times[0], times[-1])
    self.canvas.draw_idle()
