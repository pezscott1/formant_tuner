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
    arr_db_max = np.max(arr_db)
    db_floor = arr_db_max - 60  # 60 dB dynamic range
    arr_db = np.clip(arr_db, db_floor, arr_db_max)
    ny, nx = arr_db.shape

    # ---------------------------------------------------------
    # Create or update mesh
    # ---------------------------------------------------------
    if self._spec_mesh is None:
        self.ax_spec.clear()

        # Create mesh once
        self._spec_mesh = self.ax_spec.pcolormesh(
            times_small,
            freqs[mask],
            arr_db,
            shading="auto",
            cmap="magma"
        )

        # Create colorbar once
        if not hasattr(self, "_spec_colorbar") or self._spec_colorbar is None:
            self._spec_colorbar = self.ax_spec.figure.colorbar(
                self._spec_mesh, ax=self.ax_spec, fraction=0.046, pad=0.04
            )

    else:
        # Update existing mesh
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
                    shading="auto",
                    cmap="magma"
                )
            else:
                # Update mesh values
                self._spec_mesh.set_array(arr_db.ravel())
        except Exception:
            traceback.print_exc()
        # ---------------------------------------------------------
        # Formant overlays during capture (F1/F2 target lines)
        # ---------------------------------------------------------
    if getattr(self.state, "phase", None) == "capture":
        try:
            # You must define this dict somewhere upstream
            # Example: self.current_vowel_calibration = {"f1": ..., "f2": ...}
            calib = getattr(self, "current_vowel_calibration", {})
            f1_target = calib.get("f1")
            f2_target = calib.get("f2")

            # Optional: only show overlays if formants are stable
            is_stable = getattr(self, "formants_stable", True)

            if is_stable and f1_target:
                if not hasattr(self, "_f1_line") or self._f1_line is None:
                    self._f1_line = self.ax_spec.axhline(
                        f1_target, color="cyan", linestyle="--", linewidth=1.2, alpha=0.7
                    )
                else:
                    self._f1_line.set_ydata([f1_target, f1_target])

            if is_stable and f2_target:
                if not hasattr(self, "_f2_line") or self._f2_line is None:
                    self._f2_line = self.ax_spec.axhline(
                        f2_target, color="lime", linestyle="--", linewidth=1.2, alpha=0.7
                    )
                else:
                    self._f2_line.set_ydata([f2_target, f2_target])

        except Exception:
            traceback.print_exc()

    # Always update colorbar if it exists
    if getattr(self, "_spec_colorbar", None) is not None:
        self._spec_colorbar.update_normal(self._spec_mesh)

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
def safe_spectrogram(y, sr, n_fft=1024, hop_length=256, window_seconds=3.0):
    """
    Compute a safe, speech-optimized spectrogram with:
      - pre-emphasis
      - librosa STFT when available
      - consistent fallback FFT
      - rolling time window for clarity
    """
    # ---------------------------------------------------------
    # Handle empty input
    # ---------------------------------------------------------
    if y is None or len(y) == 0:
        f = np.linspace(0, sr / 2, 128)
        t = np.array([0.0])
        Sxx = np.zeros((f.size, t.size))
        return f, t, Sxx

    # ---------------------------------------------------------
    # Pre-emphasis (boosts clarity of F2/F3)
    # ---------------------------------------------------------
    try:
        y = np.asarray(y, dtype=float)
        if len(y) > 1:
            y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    except Exception:
        pass

    # ---------------------------------------------------------
    # Try librosa STFT first
    # ---------------------------------------------------------
    try:
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        times = librosa.frames_to_time(
            np.arange(S.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft
        )
    except Exception as e:
        logger.debug("librosa spectrogram failed: %s", e)

        # -----------------------------------------------------
        # Fallback FFT path (consistent with librosa)
        # -----------------------------------------------------
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
            # Compute times
            times = np.arange(S.shape[1]) * (hop_length / float(sr))
            # Ensure 1-D
            times = np.asarray(times).reshape(-1)
        except Exception as e2:
            logger.debug("fft spectrogram failed: %s", e2)
            f = np.linspace(0, sr / 2, 128)
            t = np.array([0.0])
            Sxx = np.zeros((f.size, t.size))
            return f, t, Sxx

    # ---------------------------------------------------------
    # Optional: rolling time window for clarity (default 3 sec)
    # ---------------------------------------------------------
    if times.size > 1 and window_seconds is not None:
        if times.size > 0:
            t_end = float(times[-1])
        else:
            t_end = 0.0
        t_start = max(0.0, t_end - window_seconds)
        keep = times >= t_start

        if keep.sum() > 1:
            times = times[keep]
            S = S[:, keep]

    return freqs, times, S
