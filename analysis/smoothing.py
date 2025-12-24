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
    """
    Exponential smoother + HPS fallback + octave correction.
    Eliminates doubling/halving, suppresses jumps, and avoids runaway drift.
    """

    def __init__(self, alpha=0.25, jump_limit=80, sr=44100, hps_window=2048,
                 min_f0=50, max_f0=500, voice_type=None):
        self.alpha = alpha
        self.jump_limit = jump_limit
        self.current = None

        self.hps_window = hps_window
        self.audio_buffer = deque(maxlen=hps_window)
        self.sr = sr
        self.min_f0 = float(min_f0)
        self.max_f0 = float(max_f0)
        self.voice_type = voice_type  # optional, can be set later

    def reset(self):
        self.current = None
        self.audio_buffer.clear()

    def _octave_correct(self, f0):
        """
        Light octave correction:
        - For bass voices, try to avoid halving < ~90 Hz.
        - If we're near 2× or ½× of the current, snap to the closer octave.
        """
        if f0 is None or np.isnan(f0):
            return f0

        # Bass-specific heuristic: very low f0 likely halved
        if self.voice_type == "bass" and f0 < 90:
            f0_candidate = f0 * 2.0
            if self.min_f0 <= f0_candidate <= self.max_f0:
                f0 = f0_candidate

        if self.current is not None and self.jump_limit is not None:
            # Check if f0 is roughly double or half of current
            if abs(f0 - 2 * self.current) < 40:
                f0 = 2 * self.current
            elif abs(2 * f0 - self.current) < 40:
                f0 = 0.5 * self.current

        return f0

    def update(self, f0):
        # Reject NaN or None
        if f0 is None or np.isnan(f0):
            return self.current

        f0 = float(f0)

        # First, apply octave correction relative to current + voice_type
        f0 = self._octave_correct(f0)

        # Sudden jump suppression (after octave correction)
        if self.current is not None and self.jump_limit is not None:
            if abs(f0 - self.current) > self.jump_limit:
                # Try HPS fallback
                if len(self.audio_buffer) >= self.hps_window:
                    hps_f0 = hps_pitch(np.array(self.audio_buffer), self.sr,
                                       max_f0=int(self.max_f0), min_f0=int(self.min_f0))
                    if hps_f0 is not None:
                        f0 = self._octave_correct(hps_f0)
                else:
                    # Clamp to previous
                    f0 = self.current

        # First stable value
        if self.current is None:
            self.current = f0
        else:
            # Exponential smoothing
            self.current = self.alpha * f0 + (1 - self.alpha) * self.current

        return self.current

    def push_audio(self, signal):
        if signal is None:
            return
        sig = np.asarray(signal, dtype=float).flatten()
        for x in sig:
            self.audio_buffer.append(x)


# ---------------------------------------------------------
# Formant smoothing
# ---------------------------------------------------------
class MedianSmoother:
    """
    Rolling median smoother for F1/F2 with outlier rejection.
    """

    def __init__(self, window=5, outlier_thresh=500):
        self.window = window
        self.outlier_thresh = outlier_thresh
        self.buffer = deque(maxlen=window)
        self.stability = FormantStabilityTracker(
            window_size=6, var_threshold=1e5, min_full_frames=3, trim_pct=10)
        self.formants_stable = False
        self._stability_score = float("inf")

    def reset(self):
        self.buffer.clear()

    def update(self, f1, f2):
        f1 = np.nan if f1 is None else float(f1)
        f2 = np.nan if f2 is None else float(f2)

        # Append
        self.buffer.append((f1, f2))
        arr = np.array(self.buffer, dtype=float)

        # Outlier rejection (LPC glitches)
        if len(arr) >= 3:
            med_f1 = np.nanmedian(arr[:, 0])
            med_f2 = np.nanmedian(arr[:, 1])

            # Reject values far from median
            arr[:, 0][np.abs(arr[:, 0] - med_f1) > self.outlier_thresh] = np.nan
            arr[:, 1][np.abs(arr[:, 1] - med_f2) > self.outlier_thresh] = np.nan

        # Compute smoothed values
        f1_s = None if np.all(np.isnan(arr[:, 0])) else float(np.nanmedian(arr[:, 0]))
        f2_s = None if np.all(np.isnan(arr[:, 1])) else float(np.nanmedian(arr[:, 1]))
        stable_bool, score = self.stability.update(f1_s, f2_s)
        self.formants_stable = stable_bool
        self._stability_score = score
        return f1_s, f2_s


# ---------------------------------------------------------
# Label smoothing (vowel)
# ---------------------------------------------------------
class LabelSmoother:
    """
    Stabilizes vowel labels using hysteresis + confidence threshold.
    Simple, low-latency smoother.
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
        # Reject low-confidence labels
        if new_label is None or confidence < self.min_confidence:
            return self.current

        # First label
        if self.current is None:
            self.current = new_label
            self.last = new_label
            self.counter = 0
            return self.current

        # Same label → reset
        if new_label == self.current:
            self.counter = 0
            self.last = new_label
            return self.current

        # New label → hysteresis
        if new_label == self.last:
            self.counter += 1
        else:
            self.last = new_label
            self.counter = 1

        # Accept if persistent
        if self.counter >= self.hold_frames:
            self.current = new_label
            self.counter = 0

        return self.current

# Put this in analysis/smoothing.py (or the module that defines your stability logic)


class FormantStabilityTracker:
    def __init__(self, window_size=6,
                 var_threshold=1e5, min_full_frames=3, trim_pct=10):
        """
        window_size: number of recent frames to consider
        var_threshold: allowed variance (sum of F1+F2 variances) to declare stable
        min_full_frames: minimum number of frames
        in window that must have both F1 and F2
        trim_pct: percent to trim from each tail when computing variance (robust)
        """
        self.window_size = int(window_size)
        self.var_threshold = float(var_threshold)
        self.min_full_frames = int(min_full_frames)
        self.trim_pct = float(trim_pct)

        self.f1_buf = deque(maxlen=self.window_size)
        self.f2_buf = deque(maxlen=self.window_size)

    def update(self, f1, f2):
        # Append np.nan for missing values so we keep alignment
        self.f1_buf.append(np.nan if f1 is None else float(f1))
        self.f2_buf.append(np.nan if f2 is None else float(f2))

        f1_arr = np.asarray(self.f1_buf, dtype=float)
        f2_arr = np.asarray(self.f2_buf, dtype=float)
        full_mask = np.isfinite(f1_arr) & np.isfinite(f2_arr)
        full_count = int(np.sum(full_mask))

        if full_count < self.min_full_frames:
            return False, float("inf")

        f1_full = f1_arr[full_mask]
        f2_full = f2_arr[full_mask]

        # Robust trimming to remove outliers
        if self.trim_pct > 0 and f1_full.size > 2:
            lo = np.percentile(f1_full, self.trim_pct / 2.0)
            hi = np.percentile(f1_full, 100.0 - (self.trim_pct / 2.0))
            f1_full = f1_full[(f1_full >= lo) & (f1_full <= hi)]

        # If trimming removed too many frames, fall back to untrimmed
        if f1_full.size < self.min_full_frames or f2_full.size < self.min_full_frames:
            f1_full = f1_arr[full_mask]
            f2_full = f2_arr[full_mask]

        f1_var = float(np.var(f1_full)) if f1_full.size > 0 else float("inf")
        f2_var = float(np.var(f2_full)) if f2_full.size > 0 else float("inf")

        stability_score = f1_var + f2_var
        is_stable = stability_score <= self.var_threshold

        return bool(is_stable), float(stability_score)

    def reset(self):
        self.f1_buf.clear()
        self.f2_buf.clear()
