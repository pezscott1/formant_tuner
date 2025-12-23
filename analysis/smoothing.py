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
    Exponential smoother + HPS fallback + stability window.
    Eliminates doubling/halving, suppresses jumps, and avoids runaway drift.
    """

    def __init__(self, alpha=0.25, jump_limit=80, sr=44100, hps_window=2048):
        self.alpha = alpha
        self.jump_limit = jump_limit
        self.current = None

        # HPS fallback buffer (rolling window)
        self.hps_window = hps_window
        self.audio_buffer = deque(maxlen=hps_window)
        self.sr = sr

    def update(self, f0):
        # Reject NaN or None
        if f0 is None or np.isnan(f0):
            return self.current

        # Doubling/halving suppression
        if self.current is not None and self.jump_limit is not None:
            if abs(f0 - 2 * self.current) < 40:
                f0 = self.current
            elif abs(2 * f0 - self.current) < 40:
                f0 = self.current

        # Sudden jump suppression
        if self.current is not None and self.jump_limit is not None:
            if abs(f0 - self.current) > self.jump_limit:
                # Try HPS fallback
                if len(self.audio_buffer) >= self.hps_window:
                    hps_f0 = hps_pitch(np.array(self.audio_buffer), self.sr)
                    if hps_f0 is not None:
                        f0 = hps_f0
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
