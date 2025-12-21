import numpy as np


def estimate_pitch(frame, sr):
    """Estimate pitch F0 from a frame using autocorrelation."""
    frame = np.asarray(frame, dtype=float)
    if frame.size == 0:
        return None
    frame = frame - np.mean(frame)
    corr = np.correlate(frame, frame, mode="full")
    corr = corr[len(corr) // 2:]
    d = np.diff(corr)
    pos = np.where(d > 0)[0]
    if pos.size == 0:
        return None
    start = pos[0]
    peak = np.argmax(corr[start:]) + start
    if peak == 0:
        return None
    return sr / peak
