# calibration/plotter.py
import numpy as np
import time
import traceback
import librosa
import logging


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )


def update_artists(self, freqs, times, s, f1, f2, vowel):  # noqa: C901
    if freqs is None or times is None or s is None:
        return

    # Only show vowel scatter during capture
    if self.state.phase != "capture":
        vowel = None

    # ---------------------------------------------------------
    # Spectrogram (<= 4 kHz)
    # ---------------------------------------------------------
    mask = freqs <= 4000
    if isinstance(mask, np.ndarray) and mask.sum() < 2:
        mask = np.arange(len(freqs))

    # Much smoother: no aggressive downsampling
    # Keep time resolution high but cap max bins
    max_time_bins = 400
    if s.shape[1] > max_time_bins:
        step = s.shape[1] // max_time_bins
    else:
        step = 1

    S_small = s[mask, ::step]
    times_small = times[::step]

    # Convert to dB
    arr_db = 10 * np.log10(S_small + 1e-12)
    ny, nx = arr_db.shape

    # ---------------------------------------------------------
    # Create or update mesh
    # ---------------------------------------------------------
    if self._spec_mesh is None:
        self.ax_spec.clear()
        try:
            self._spec_mesh = self.ax_spec.pcolormesh(
                times_small,
                freqs[mask],
                arr_db,
                shading="auto"
            )
        except Exception:
            mean_spec = np.mean(S_small, axis=1) \
                if S_small.size else np.zeros_like(freqs[mask])
            self.ax_spec.plot(freqs[mask], 10 * np.log10(mean_spec + 1e-12))
    else:
        try:
            expected_size = ny * nx
            current_size = self._spec_mesh.get_array().size

            if current_size != expected_size:
                # Recreate mesh if dimensions changed
                self.ax_spec.clear()
                self._spec_mesh = self.ax_spec.pcolormesh(
                    times_small,
                    freqs[mask],
                    arr_db,
                    shading="auto"
                )
            else:
                # Update existing mesh
                self._spec_mesh.set_array(arr_db.ravel())
        except Exception:
            traceback.print_exc()

    # Axis labels
    self.ax_spec.set_title("Spectrogram")
    self.ax_spec.set_xlabel("Time (s)")
    self.ax_spec.set_ylabel("Frequency (Hz)")
    self.ax_spec.set_ylim(0, 4000)

    # Draw
    self.canvas.draw_idle()

    # ---------------------------------------------------------
    # Vowel scatter
    # ---------------------------------------------------------
    if f1 is None or f2 is None or vowel is None:
        now = time.time()
        if now - self._last_draw >= self._min_draw_interval:
            try:
                self.canvas.draw_idle()
            except Exception:
                pass
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
        # Update existing scatter
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
    try:
        self.canvas.draw_idle()
    except Exception:
        pass

    now = time.time()
    if now - self._last_draw >= self._min_draw_interval:
        try:
            self.canvas.draw_idle()
        except Exception:
            pass
        self._last_draw = now


# -------------------------
# Spectrogram and expected formants
# -------------------------

def safe_spectrogram(y, sr, n_fft=2048, hop_length=512):
    """Compute a safe spectrogram, returning freqs, times, and power."""
    if y is None or len(y) == 0:
        f = np.linspace(0, sr / 2, 128)
        t = np.array([0.0])
        Sxx = np.zeros((f.size, t.size))
        return f, t, Sxx

    try:
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        times = librosa.frames_to_time(
            np.arange(S.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft
        )
        return freqs, times, S
    except Exception as e:  # noqa: E722
        logger.debug("librosa spectrogram failed: %s", e)

    try:
        win = np.hanning(n_fft)
        frames = []
        for i in range(0, max(1, len(y) - n_fft + 1), hop_length):
            frame = y[i: i + n_fft] * win
            spec = np.abs(np.fft.rfft(frame)) ** 2
            frames.append(spec)
        S = (
            np.column_stack(frames)
            if frames
            else np.zeros((n_fft // 2 + 1, 1))
        )
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        times = np.arange(S.shape[1]) * (hop_length / float(sr))
        return freqs, times, S
    except Exception as e:  # noqa: E722
        logger.debug("fft spectrogram failed: %s", e)
        f = np.linspace(0, sr / 2, 128)
        t = np.array([0.0])
        Sxx = np.zeros((f.size, t.size))
        return f, t, Sxx
