# analysis/smoothing.py
from collections import deque
import numpy as np


# ---------------------------------------------------------
# Pitch smoothing (F0)
# ---------------------------------------------------------
class PitchSmoother:
    """
    Simple exponential smoother for pitch.
    Keeps pitch stable but responsive.
    """

    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.current = None

    def update(self, f0):
        if f0 is None or np.isnan(f0):
            return self.current

        if self.current is None:
            self.current = f0
        else:
            self.current = self.alpha * f0 + (1 - self.alpha) * self.current

        return self.current


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
