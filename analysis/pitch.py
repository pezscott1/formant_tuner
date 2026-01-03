# analysis/pitch.py
# NOT CURRENTLY USED OR IMPORTED, RETAINING AS LEGACY
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PitchResult:
    f0: Optional[float]
    confidence: float
    method: str
    debug: dict


def estimate_pitch(  # noqa: C901
    frame: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 800.0,
    debug: bool = False,
) -> PitchResult:
    """
    Robust autocorrelation-based pitch estimator with:
      - short-frame fallback
      - clipping suppression
      - parabolic interpolation
      - confidence scoring
    """
    frame = np.asarray(frame, dtype=float)
    n = frame.size

    if n == 0:
        return PitchResult(None, 0.0, "none", {"reason": "empty"})

    # -------------------------
    # Very short frame fallback
    # -------------------------
    if n < 256:
        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]
        d = np.diff(corr)
        pos = np.where(d > 0)[0]
        if pos.size == 0:
            return PitchResult(None, 0.0, "short", {"reason": "no rising slope"})
        start = pos[0]
        peak = np.argmax(corr[start:]) + start
        if peak <= 0:
            return PitchResult(None, 0.0, "short", {"reason": "peak <= 0"})
        f0 = sr / peak
        if not (fmin <= f0 <= fmax):
            return PitchResult(None, 0.0, "short", {"reason": "out_of_range"})
        return PitchResult(float(f0), 0.4, "short", {"peak": peak})

    # -------------------------
    # Normal path
    # -------------------------
    frame = frame - np.mean(frame)

    # Clipping suppression
    clip_level = 0.6 * np.max(np.abs(frame))
    if clip_level > 0:
        frame = np.where(
            frame >= clip_level,
            frame - clip_level,
            np.where(frame <= -clip_level, frame + clip_level, frame),
        )

    corr = np.correlate(frame, frame, mode="full")
    corr = corr[n:]

    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    max_lag = min(max_lag, len(corr) - 1)

    if min_lag >= max_lag:
        return PitchResult(None, 0.0, "none", {"reason": "lag window invalid"})

    segment = corr[min_lag:max_lag]
    if segment.size == 0:
        return PitchResult(None, 0.0, "none", {"reason": "empty segment"})

    peak_idx = np.argmax(segment) + min_lag

    # Parabolic interpolation
    if 1 <= peak_idx < len(corr) - 1:
        y0, y1, y2 = corr[peak_idx - 1], corr[peak_idx], corr[peak_idx + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom != 0:
            peak_idx = peak_idx + 0.5 * (y0 - y2) / denom

    if peak_idx <= 0:
        return PitchResult(None, 0.0, "none", {"reason": "peak <= 0"})

    f0 = sr / peak_idx
    if not (fmin <= f0 <= fmax):
        return PitchResult(None, 0.0, "none", {"reason": "out_of_range"})

    # Confidence: normalized autocorrelation peak
    raw_peak = corr[int(round(peak_idx))]
    conf = float(np.clip(raw_peak / (corr[0] + 1e-6), 0.0, 1.0))

    dbg = {}
    if debug:
        dbg = {
            "peak_idx": peak_idx,
            "raw_peak": float(raw_peak),
            "corr0": float(corr[0]),
        }

    return PitchResult(float(f0), conf, "autocorr", dbg)
