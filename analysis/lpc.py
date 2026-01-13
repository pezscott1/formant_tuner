"""
Simple, stable LPC-based formant estimator for 48 kHz classical voice.

Design:
- Internally downsample to 16 kHz for LPC stability
- Fixed LPC order (12)
- Moderate pre-emphasis (0.97)
- 40 ms window
- Light cepstral smoothing (lifter_cut=20)
- Clean fallback using spectral envelope peaks
- Returns f1, f2, f3, confidence, method
"""
from __future__ import annotations
from typing import Optional

import librosa
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class FormantResult:
    f1: float | None
    f2: float | None
    f3: float | None
    confidence: float
    method: str
    peaks: list[float]
    roots: list[complex]
    bandwidths: list[float]
    debug: dict
    lpc_order: Optional[int] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_formants(
    y: NDArray[np.float64] | NDArray[np.float32],
    sr: int,
    debug: bool = False,
) -> FormantResult:
    """
    Stable LPC formant estimator for 48 kHz classical singing.
    Internally downsamples to ~16 kHz for LPC stability.
    """

    if y is None or sr <= 0:
        return _empty("invalid_input")

    y = np.asarray(y, dtype=float).flatten()
    if y.size == 0:
        return _empty("empty_frame")

    # ---------------------------------------------------------
    # 1. Pre-emphasis
    # ---------------------------------------------------------
    pre = 0.97 if sr >= 44100 else 0.95
    y = np.append(y[0], y[1:] - pre * y[:-1])

    # ---------------------------------------------------------
    # 2. Downsampling for LPC
    # ---------------------------------------------------------
    y_ds = librosa.resample(y, orig_sr=sr, target_sr=16000)
    sr_eff = 16000

    # ---------------------------------------------------------
    # 3. Windowing (50 ms)
    # ---------------------------------------------------------
    win_len = int(sr_eff * 0.050)
    if win_len < 256:
        return _empty("window_too_short")

    N = min(len(y_ds), win_len)
    if N < 256:
        return _empty("window_too_short")

    frame = y_ds[:N] * np.hamming(N)

    # ---------------------------------------------------------
    # 4. LPC order (fixed)
    # ---------------------------------------------------------
    order = 12
    if len(frame) < 3 * order:
        return _empty("insufficient_samples")

    A = _compute_lpc(frame, order)
    if A is None:
        return _fallback(frame, sr_eff, "lpc_fail", order)
    # ---------------------------------------------------------
    # 5. LPC roots → formants
    # ---------------------------------------------------------
    try:
        roots = np.roots(A)
    except Exception:
        return _fallback(frame, sr_eff, "root_fail", order)

    ang = np.angle(roots)
    freqs = ang * (sr_eff / (2 * np.pi))
    bw = -0.5 * (sr_eff / np.pi) * np.log(np.abs(roots))

    mask = (freqs > 80) & (freqs < 4000) & (bw < 1000)
    freqs_f = freqs[mask].real
    bw_f = bw[mask].real
    roots_f = roots[mask]

    if freqs_f.size < 1:
        # Only bail out if literally nothing survives
        return _fallback(frame, sr_eff, "no_valid_poles", order)

    freqs_sorted = np.sort(freqs_f)

    # 3) Normal formant extraction
    f1, f2, f3 = _extract_formants(freqs_sorted)

    # If LPC extracted no formants, keep method="lpc", not fallback
    if f1 is None and f2 is None:
        return FormantResult(
            f1=None,
            f2=None,
            f3=None,
            confidence=0.0,
            method="lpc",
            peaks=[],
            roots=roots_f.tolist(),
            bandwidths=bw_f.tolist(),
            debug={"reason": "no_formants"},
            lpc_order=order,
        )

    # 4) Envelope peaks for confidence
    peak_freqs, peak_heights = _smooth_peaks(
        frame, sr_eff, lifter_cut=20, nfft=4096
    )

    conf = _confidence(f1, f2, peak_freqs)

    dbg = {}
    if debug:
        dbg = {
            "freqs_poles": freqs_sorted.tolist(),
            "bw_poles": bw_f.tolist(),
            "peak_freqs": peak_freqs.tolist(),
            "peak_heights": peak_heights.tolist(),
            "pre_emph": pre,
            "sr_eff": sr_eff,
        }

    return FormantResult(
        f1=f1,
        f2=f2,
        f3=f3,
        confidence=conf,
        method="lpc",
        peaks=peak_freqs.tolist(),
        roots=roots_f.tolist(),
        bandwidths=bw_f.tolist(),
        debug=dbg,
        lpc_order=order,
    )


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _fallback(frame, sr, reason: str, order: Optional[int]) -> FormantResult:
    peak_freqs, peak_heights = _smooth_peaks(
        frame, sr, lifter_cut=20, nfft=4096
    )
    print(
        f"DEBUG _fallback: reason={reason} "
        f"n_peaks={peak_freqs.size} "
        f"first_peaks={peak_freqs[:5] if peak_freqs.size > 0 else []}"
    )

    # No peaks at all → empty fallback
    if peak_freqs.size == 0:
        return FormantResult(
            f1=None,
            f2=None,
            f3=None,
            confidence=0.0,
            method="fallback",
            peaks=[],
            roots=[],
            bandwidths=[],
            debug={"reason": f"fallback_no_peaks:{reason}"},
            lpc_order=order,
        )

    # Extract fallback formants from spectral peaks
    f1, f2, f3 = _extract_formants(np.sort(peak_freqs))
    conf = _confidence(f1, f2, peak_freqs)

    dbg = {
        "reason": reason,
        "peak_freqs": peak_freqs.tolist(),
        "peak_heights": peak_heights.tolist(),
    }

    return FormantResult(
        f1=f1,
        f2=f2,
        f3=f3,
        confidence=conf,
        method="fallback",
        peaks=peak_freqs.tolist(),
        roots=[],
        bandwidths=[],
        debug=dbg,
        lpc_order=order,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty(reason: str) -> FormantResult:
    return FormantResult(
        f1=None,
        f2=None,
        f3=None,
        confidence=0.0,
        method="none",
        peaks=[],
        roots=[],
        bandwidths=[],
        debug={"reason": reason},
        lpc_order=None,
    )


def _compute_lpc(frame, order):
    try:
        R = np.correlate(frame, frame, mode="full")
        mid = len(R) // 2
        r = R[mid: mid + order + 1]
        return _levinson(r, order)
    except Exception:
        return None


def _levinson(r, order):
    if r[0] <= 0:
        return None

    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]

    for i in range(1, order + 1):
        acc = sum(a[j] * r[i - j] for j in range(1, i))
        k = -(r[i] + acc) / e

        a_prev = a.copy()
        a[1:i] = a_prev[1:i] + k * a_prev[i - 1:0:-1]
        a[i] = k

        e *= (1 - k * k)
        if e <= 0:
            return None

    return a


def _extract_formants(freqs_sorted):
    """
    Choose F1, F2, F3 from sorted pole frequencies with broad ranges
    suitable for classical/baritone voice.
    """
    if freqs_sorted.size == 0:
        return None, None, None

    freqs = np.asarray(freqs_sorted, float)

    # F1: very broad range
    f1_candidates = freqs[(freqs >= 150) & (freqs <= 900)]
    f1 = float(f1_candidates[0]) if f1_candidates.size > 0 else None
    if f1 is None:
        # last-resort: if there is any peak below 1500, treat the lowest as F1
        low_candidates = freqs[(freqs >= 150) & (freqs <= 1500)]
        if low_candidates.size > 0:
            f1 = float(low_candidates[0])

    # F2: next pole above F1; if no F1, lowest pole above 800 Hz
    f2 = None
    if f1 is not None:
        f2_candidates = freqs[(freqs > f1 + 50) & (freqs <= 2500)]
        if f2_candidates.size > 0:
            f2 = float(f2_candidates[0])
    else:
        f2_candidates = freqs[(freqs >= 800) & (freqs <= 3500)]
        if f2_candidates.size > 0:
            f2 = float(f2_candidates[0])

    # F3: next pole above F2
    f3 = None
    if f2 is not None:
        f3_candidates = freqs[(freqs > f2 + 150) & (freqs <= 4000)]
        if f3_candidates.size > 0:
            f3 = float(f3_candidates[0])

    return f1, f2, f3


def _smooth_peaks(frame, sr, lifter_cut=20, nfft=4096):
    win = frame * np.hamming(len(frame))
    X = np.abs(np.fft.rfft(win, n=nfft))
    logX = np.log(X + 1e-12)

    cep = np.fft.irfft(logX)
    cep[lifter_cut:-lifter_cut] = 0

    smooth_log = np.fft.rfft(cep, n=nfft).real
    env = np.exp(smooth_log)

    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    mask = (freqs >= 80) & (freqs <= 4000)

    env_m = env[mask]
    freqs_m = freqs[mask]

    if env_m.size == 0:
        return np.array([]), np.array([])

    base = np.max(env_m)

    # Catch weak F1
    thresh = base * 0.005   # was 0.02
    peaks: np.ndarray
    peaks, _ = find_peaks(env_m, height=thresh)

    # If still nothing, relax again
    if peaks.size == 0:
        peaks, _ = find_peaks(env_m, height=base * 0.002)

    if peaks.size == 0:
        return np.array([]), np.array([])

    pf = np.round(freqs_m[peaks], 1)
    ph = np.round(env_m[peaks], 2)
    return pf, ph


def _confidence(f1, f2, peak_freqs):
    if f1 is None and f2 is None:
        return 0.0

    score = 0.7

    if peak_freqs.size > 0 and f1 is not None:
        d = np.min(np.abs(peak_freqs - f1))
        score += float(np.exp(-d / 200.0)) * 0.2

    if peak_freqs.size > 1 and f2 is not None:
        d = np.min(np.abs(peak_freqs - f2))
        score += float(np.exp(-d / 300.0)) * 0.2

    return float(max(0.0, min(1.0, score)))
