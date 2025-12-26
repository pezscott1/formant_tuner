# analysis/smoothing.py
from collections import deque
import numpy as np


# ---------------------------------------------------------
# Harmonic Product Spectrum (for pitch fallback)
# ---------------------------------------------------------
def hps_pitch(signal, sr, max_f0=500, min_f0=50):
    if signal is None or len(signal) < 1024:
        return None

    window = np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(signal * window))

    hps = spectrum.copy()
    for h in (2, 3, 4):
        dec = spectrum[::h]
        hps[:len(dec)] *= dec

    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    mask = (freqs >= min_f0) & (freqs <= max_f0)
    if not np.any(mask):
        return None

    idx = np.argmax(hps[mask])
    return float(freqs[mask][idx])


# ---------------------------------------------------------
# Pitch smoothing
# ---------------------------------------------------------
class PitchSmoother:
    def __init__(self, alpha=0.25, jump_limit=80, min_confidence=0.0,
                 hps_enabled=False, sr=48000):
        self.alpha = alpha
        self.jump_limit = jump_limit
        self.min_confidence = min_confidence
        self.hps_enabled = hps_enabled
        self.current = None
        self._sr = sr

    # -------------------------
    # Octave correction (test‑expected)
    # -------------------------
    def _octave_correct(self, new):
        if self.current is None or new is None:
            return new

        # Snap to 2×current if within 40 Hz
        if abs(new - 2 * self.current) <= 40:
            return 2 * self.current

        # Snap to 0.5×current if within 20 Hz
        if abs(new - 0.5 * self.current) <= 20:
            return 0.5 * self.current

        return new

    def reset(self):
        self.current = None

    # -------------------------
    # Update logic (test‑expected)
    # -------------------------
    def update(self, new, confidence=1.0):
        # --- unwrap PitchResult objects ---
        if hasattr(new, "frequency"):
            new = new.frequency

        # ignore None
        if new is None:
            return self.current

        try:
            new = float(new)
        except Exception:
            return self.current

        # first frame
        if self.current is None:
            self.current = new
            return self.current

        # confidence gating
        if confidence < self.min_confidence:
            return self.current

        # jump suppression
        if abs(new - self.current) > self.jump_limit:
            return self.current

        # EMA smoothing
        self.current = self.alpha * new + (1 - self.alpha) * self.current
        return self.current


# ---------------------------------------------------------
# Formant smoothing
# ---------------------------------------------------------
class MedianSmoother:
    """
    Rolling median smoother for F1/F2/F3 with:
      - outlier rejection
      - LPC confidence gating
      - ridge suppression for 48 kHz mics
      - 3‑formant stability tracking
    """

    def __init__(self, window=5, outlier_thresh=500, min_confidence=0.25):
        self.window = window
        self.outlier_thresh = outlier_thresh
        self.min_confidence = min_confidence

        # Three independent rolling buffers
        self.buf_f1 = deque(maxlen=window)
        self.buf_f2 = deque(maxlen=window)
        self.buf_f3 = deque(maxlen=window)

        # Stability tracker (now 3‑formant aware)
        self.stability = FormantStabilityTracker(
            window_size=6,
            var_threshold=1e5,
            min_full_frames=3,
            trim_pct=10,
        )

        self.formants_stable = False
        self._stability_score = float("inf")

    def reset(self):
        self.buf_f1.clear()
        self.buf_f2.clear()
        self.buf_f3.clear()

    def update(self, f1=None, f2=None, f3=None, confidence=1.0):
        # Reject low-confidence LPC frames
        if confidence < self.min_confidence:
            return None, None, None

        # Convert to floats or NaN
        f1 = np.nan if f1 is None else float(f1)
        f2 = np.nan if f2 is None else float(f2)
        f3 = np.nan if f3 is None else float(f3)

        # Suppress 2600 Hz ridge (mic artifact)
        if 2400 < f1 < 2800:
            f1 = np.nan
        if 2400 < f2 < 2800:
            f2 = np.nan
        if 2400 < f3 < 2800:
            f3 = np.nan

        # Append to buffers
        self.buf_f1.append(f1)
        self.buf_f2.append(f2)
        self.buf_f3.append(f3)

        # Convert to arrays
        arr_f1 = np.asarray(self.buf_f1, dtype=float)
        arr_f2 = np.asarray(self.buf_f2, dtype=float)
        arr_f3 = np.asarray(self.buf_f3, dtype=float)

        # Outlier rejection for each formant
        def reject_outliers(arr):
            if arr.size < 3 or not np.any(np.isfinite(arr)):
                return arr
            med = np.nanmedian(arr)
            mask = np.abs(arr - med) > self.outlier_thresh
            arr = arr.copy()
            arr[mask] = np.nan
            return arr

        arr_f1 = reject_outliers(arr_f1)
        arr_f2 = reject_outliers(arr_f2)
        arr_f3 = reject_outliers(arr_f3)

        # Compute medians
        f1_s = None if np.all(np.isnan(arr_f1)) else float(np.nanmedian(arr_f1))
        f2_s = None if np.all(np.isnan(arr_f2)) else float(np.nanmedian(arr_f2))
        f3_s = None if np.all(np.isnan(arr_f3)) else float(np.nanmedian(arr_f3))

        # Stability tracking (now 3‑formant aware)
        stable_bool, score = self.stability.update(f1_s, f2_s, f3_s)
        self.formants_stable = stable_bool
        self._stability_score = score

        return f1_s, f2_s, f3_s


# ---------------------------------------------------------
# Label smoothing (vowel)
# ---------------------------------------------------------
class LabelSmoother:
    """
    Stabilizes vowel labels using hysteresis + confidence threshold.
    Now uses LPC confidence as well.
    """

    def __init__(self, hold_frames=4, min_confidence=0.2):
        self.hold_frames = hold_frames
        self.min_confidence = min_confidence
        self.current = None
        self.last = None
        self.counter = 0

    def reset(self):
        self.current = None
        self.last = None
        self.counter = 0

    def update(self, new_label, confidence=1.0):
        if new_label is None or confidence < self.min_confidence:
            return self.current

        if self.current is None:
            self.current = new_label
            self.last = new_label
            self.counter = 0
            return self.current

        if new_label == self.current:
            self.counter = 0
            self.last = new_label
            return self.current

        if new_label == self.last:
            self.counter += 1
        else:
            self.last = new_label
            self.counter = 1

        if self.counter >= self.hold_frames:
            self.current = new_label
            self.counter = 0

        return self.current


# ---------------------------------------------------------
# Formant stability tracker
# ---------------------------------------------------------
class FormantStabilityTracker:
    """
    Tracks stability of F1/F2/F3 over time.
    Detects:
      - ridge collapse (all formants near 2600 Hz)
      - low-confidence frames
      - high variance instability
    """

    def __init__(
        self,
        window_size=6,
        var_threshold=1e5,
        min_full_frames=3,
        trim_pct=10,
    ):
        self.window_size = int(window_size)
        self.var_threshold = float(var_threshold)
        self.min_full_frames = int(min_full_frames)
        self.trim_pct = float(trim_pct)

        self.f1_buf = deque(maxlen=self.window_size)
        self.f2_buf = deque(maxlen=self.window_size)
        self.f3_buf = deque(maxlen=self.window_size)

    def update(self, f1, f2, f3):
        self.f1_buf.append(np.nan if f1 is None else float(f1))
        self.f2_buf.append(np.nan if f2 is None else float(f2))
        self.f3_buf.append(np.nan if f3 is None else float(f3))

        f1_arr = np.asarray(self.f1_buf, dtype=float)
        f2_arr = np.asarray(self.f2_buf, dtype=float)
        f3_arr = np.asarray(self.f3_buf, dtype=float)

        # Only frames where all 3 formants are finite
        full_mask = np.isfinite(f1_arr) & np.isfinite(f2_arr) & np.isfinite(f3_arr)
        full_count = int(np.sum(full_mask))

        if full_count < self.min_full_frames:
            return False, float("inf")

        f1_full = f1_arr[full_mask]
        f2_full = f2_arr[full_mask]
        f3_full = f3_arr[full_mask]

        # Ridge collapse detection
        def is_ridge_band(x):
            return np.all((2400 < x) & (x < 2800))

        if is_ridge_band(f1_full) and is_ridge_band(f2_full) and is_ridge_band(f3_full):
            return False, float("inf")

        # Robust trimming
        def trim(arr):
            if arr.size <= 2 or self.trim_pct <= 0:
                return arr
            lo = np.percentile(arr, self.trim_pct / 2.0)
            hi = np.percentile(arr, 100 - self.trim_pct / 2.0)
            trimmed = arr[(arr >= lo) & (arr <= hi)]
            return trimmed if trimmed.size >= self.min_full_frames else arr

        f1_full = trim(f1_full)
        f2_full = trim(f2_full)
        f3_full = trim(f3_full)

        # Variance-based stability
        f1_var = float(np.var(f1_full)) if f1_full.size > 0 else float("inf")
        f2_var = float(np.var(f2_full)) if f2_full.size > 0 else float("inf")
        f3_var = float(np.var(f3_full)) if f3_full.size > 0 else float("inf")

        stability_score = f1_var + f2_var + f3_var
        is_stable = stability_score <= self.var_threshold

        return bool(is_stable), float(stability_score)

    def reset(self):
        self.f1_buf.clear()
        self.f2_buf.clear()
        self.f3_buf.clear()
