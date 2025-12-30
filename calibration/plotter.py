# calibration/plotter.py
import numpy as np
import time
import librosa
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )


def update_artists(self, freqs, times, s, f1, f2, vowel):
    """
    Modernized calibration plotter:
      - Uses hybrid metadata
      - Uses dict-based calibrated targets
      - Uses vowel plausibility windows
      - Draws spectrogram + target lines + vowel scatter
    """

    if freqs is None or times is None or s is None:
        return

    # Only show vowel scatter during capture
    if self.state.phase != "capture":
        vowel = None

    # ---------------------------------------------------------
    # Pull latest raw frame (hybrid-aware)
    # ---------------------------------------------------------
    raw = None
    if hasattr(self, "analyzer"):
        try:
            raw = self.analyzer.get_latest_raw()
        except Exception:
            raw = None

    # Hybrid metadata
    if raw and "hybrid_method" in raw:
        method = raw["hybrid_method"]
        conf = float(raw.get("confidence", 0.0))
    else:
        method = raw.get("method", "none") if raw else "none"
        conf = float(raw.get("confidence", 0.0)) if raw else 0.0

    # Stability from smoother
    stable = getattr(self.formant_smoother, "formants_stable", True)

    # ---------------------------------------------------------
    # Ridge suppression (visual only)
    # ---------------------------------------------------------
    if f1 is not None and 2400 < f1 < 2800:
        f1 = None
    if f2 is not None and 2400 < f2 < 2800:
        f2 = None

    # ---------------------------------------------------------
    # Spectrogram (<= 4 kHz)
    # ---------------------------------------------------------
    mask = freqs <= 4000
    if isinstance(mask, np.ndarray) and mask.sum() < 2:
        mask = np.arange(len(freqs))

    max_time_bins = 400
    step = s.shape[1] // max_time_bins if s.shape[1] > max_time_bins else 1

    S_small = s[mask, ::step]
    times_small = times[::step]

    arr_db = 10 * np.log10(S_small + 1e-12)
    arr_db_max = np.max(arr_db)
    db_floor = arr_db_max - 60
    arr_db = np.clip(arr_db, db_floor, arr_db_max)

    # ---------------------------------------------------------
    # Draw spectrogram
    # ---------------------------------------------------------
    self.ax_spec.clear()
    self._spec_mesh = self.ax_spec.pcolormesh(
        times_small,
        freqs[mask],
        arr_db,
        shading="auto",
        cmap="magma",
    )

    if not hasattr(self, "_spec_colorbar") or self._spec_colorbar is None:
        self._spec_colorbar = self.ax_spec.figure.colorbar(
            self._spec_mesh, ax=self.ax_spec, fraction=0.046, pad=0.04
        )
    else:
        self._spec_colorbar.update_normal(self._spec_mesh)

    # ---------------------------------------------------------
    # Target formant lines (from calibrated profile)
    # ---------------------------------------------------------
    if vowel and vowel in self.session.data:
        entry = self.session.data[vowel]
        f1_t = entry.get("f1")
        f2_t = entry.get("f2")

        if stable and f1_t:
            self.ax_spec.axhline(
                f1_t, color="cyan", linestyle="--", linewidth=1.2, alpha=0.7
            )
        if stable and f2_t:
            self.ax_spec.axhline(
                f2_t, color="lime", linestyle="--", linewidth=1.2, alpha=0.7
            )

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------
    self.ax_spec.set_title(f"Spectrogram  [{method}, conf={conf:.2f}]")
    self.ax_spec.set_xlabel("Time (s)")
    self.ax_spec.set_ylabel("Frequency (Hz)")
    self.ax_spec.set_ylim(0, 4000)

    self.canvas.draw_idle()

    # ---------------------------------------------------------
    # Vowel scatter (only if stable + confident)
    # ---------------------------------------------------------
    if f1 is None or f2 is None or vowel is None or conf < 0.25 or not stable:
        now = time.time()
        if now - self._last_draw >= self._min_draw_interval:
            self.canvas.draw_idle()
            self._last_draw = now
        return

    # Create scatter if missing
    if vowel not in self._vowel_scatters:
        scatter = self.ax_vowel.scatter(
            [f2],
            [f1],
            c=self._vowel_colors.get(vowel, "black"),
            s=70,
            zorder=4,
            label=f"/{vowel}/",
        )
        self._vowel_scatters[vowel] = scatter

        self.ax_vowel.set_title("Vowel Space")
        self.ax_vowel.set_xlabel("F2 (Hz)")
        self.ax_vowel.set_ylabel("F1 (Hz)")
        try:
            self.ax_vowel.set_xlim(4000, 0)
            self.ax_vowel.set_ylim(1200, 0)
        except Exception:
            pass
        self.ax_vowel.legend(loc="best")

    else:
        try:
            self._vowel_scatters[vowel].set_offsets(np.column_stack(([f2], [f1])))
        except Exception:
            try:
                self._vowel_scatters[vowel].remove()
            except Exception:
                pass
            scatter = self.ax_vowel.scatter(
                [f2],
                [f1],
                c=self._vowel_colors.get(vowel, "black"),
                s=70,
                zorder=4,
                label=f"/{vowel}/",
            )
            self._vowel_scatters[vowel] = scatter
            self.ax_vowel.legend(loc="best")

    # Final throttled draw
    now = time.time()
    if now - self._last_draw >= self._min_draw_interval:
        self.canvas.draw_idle()
        self._last_draw = now


# ---------------------------------------------------------
# Spectrogram helper
# ---------------------------------------------------------

def safe_spectrogram(y, sr, n_fft=1024, hop_length=256, window_seconds=3.0):
    """
    Speech-optimized spectrogram with:
      - pre-emphasis
      - librosa STFT or fallback FFT
      - rolling window
      - ridge-safe behavior for 48 kHz mics
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
        t_end = float(times[-1]) if times.size > 0 else 0.0
        t_start = max(0.0, t_end - window_seconds)
        keep = times >= t_start

        if keep.sum() > 1:
            times = times[keep]
            S = S[:, keep]

    return freqs, times, S
