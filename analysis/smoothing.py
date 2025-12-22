# analysis/smoothing.py
from collections import deque
import numpy as np


def hps_pitch(signal, sr, max_f0=500, min_f0=50):
    """
    Harmonic Product Spectrum pitch estimator.
    Returns None if no stable pitch is found.
    """
    if signal is None or len(signal) < 1024:
        return None

    # Window
    window = np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(signal * window))

    # Downsampled products
    hps = spectrum.copy()
    for h in (2, 3, 4):
        dec = spectrum[::h]
        hps[:len(dec)] *= dec

    # Frequency axis
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)

    # Limit search range
    mask = (freqs >= min_f0) & (freqs <= max_f0)
    if not np.any(mask):
        return None

    idx = np.argmax(hps[mask])
    f0 = freqs[mask][idx]

    return float(f0)


# ---------------------------------------------------------
# Pitch smoothing (F0)
# ---------------------------------------------------------
class PitchSmoother:
    """
    Exponential smoother + HPS fallback.
    Eliminates doubling/halving and stabilizes F0.
    """

    def __init__(self, alpha=0.2, jump_limit=80, sr=44100):
        """
        alpha: smoothing factor
        jump_limit: maximum allowed jump between frames.
                    Set to None to disable jump suppression.
        sr: sample rate for HPS fallback
        """
        self.alpha = alpha
        self.jump_limit = jump_limit
        self.current = None

        # Required for HPS fallback
        self.audio_buffer = []
        self.sr = sr

    def update(self, f0):
        """
        f0: raw pitch estimate from engine
        """
        # Reject NaN or None
        if f0 is None or np.isnan(f0):
            return self.current

        # If we have a previous stable pitch, check doubling/halving
        if self.current is not None and self.jump_limit is not None:
            # Doubling
            if abs(f0 - 2 * self.current) < 40:
                f0 = self.current
            # Halving
            elif abs(2 * f0 - self.current) < 40:
                f0 = self.current

        # Sudden jump suppression (optional)
        if self.current is not None and self.jump_limit is not None:
            if abs(f0 - self.current) > self.jump_limit:
                # Try HPS fallback if enough audio is buffered
                if len(self.audio_buffer) >= 2048:
                    hps_f0 = hps_pitch(
                        np.array(self.audio_buffer, dtype=float),
                        self.sr
                    )
                    if hps_f0 is not None:
                        f0 = hps_f0
                else:
                    # No fallback available → clamp to previous
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
        if sig.size > 0:
            self.audio_buffer.extend(sig)


# ---------------------------------------------------------
# Formant smoothing (F1/F2 only)
# ---------------------------------------------------------

class MedianSmoother:
    """
    Rolling median smoother for F1/F2.
    Returns None when smoothing is impossible (all NaN).
    """

    def __init__(self, window=5):
        self.window = window
        self.buffer = deque(maxlen=window)

    def update(self, f1, f2):
        # Normalize missing values
        f1 = np.nan if f1 is None else f1
        f2 = np.nan if f2 is None else f2

        # Append new pair
        self.buffer.append((f1, f2))

        arr = np.array(self.buffer, dtype=float)

        # If all values are NaN → return None
        if np.all(np.isnan(arr[:, 0])):
            f1_s = None
        else:
            f1_s = float(np.nanmedian(arr[:, 0]))

        if np.all(np.isnan(arr[:, 1])):
            f2_s = None
        else:
            f2_s = float(np.nanmedian(arr[:, 1]))

        return f1_s, f2_s


# ---------------------------------------------------------
# Vowel label smoothing
# ---------------------------------------------------------
class LabelSmoother:
    """
    Stabilizes vowel labels using hysteresis:
      - If a new label appears briefly, ignore it.
      - If it persists, accept it.
    """

    def __init__(self, hold_frames=4):
        self.hold_frames = hold_frames
        self.current = None
        self.last = None
        self.counter = 0

    def update(self, new_label):
        if new_label is None:
            return self.current

        # First label ever
        if self.current is None:
            self.current = new_label
            self.last = new_label
            self.counter = 0
            return self.current

        # Same label → reset counter
        if new_label == self.current:
            self.counter = 0
            self.last = new_label
            return self.current

        # New label → increment counter
        if new_label == self.last:
            self.counter += 1
        else:
            self.last = new_label
            self.counter = 1

        # Accept new label if it persists long enough
        if self.counter >= self.hold_frames:
            self.current = new_label
            self.counter = 0

        return self.current
