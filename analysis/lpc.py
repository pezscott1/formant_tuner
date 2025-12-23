import numpy as np
from typing import Tuple, List, Optional
from scipy.signal import find_peaks
from numpy.typing import NDArray
import logging

logger = logging.getLogger(__name__)


def _levinson_durbin(r: np.ndarray, order: int) -> Optional[np.ndarray]:
    """
    Levinson–Durbin recursion for LPC coefficients from autocorrelation.
    Returns LPC coeffs a[0..order] or None on failure.
    """
    if r.size < order + 1 or r[0] <= 0:
        return None

    a = np.zeros(order + 1, dtype=float)
    e = float(r[0])
    a[0] = 1.0

    for i in range(1, order + 1):
        acc = 0.0
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = -(r[i] + acc) / e

        a_prev = a.copy()
        a[1:i] = a_prev[1:i] + k * a_prev[i - 1:0:-1]
        a[i] = k
        e *= (1.0 - k * k)
        if e <= 0:
            return None

    return a


def estimate_formants_lpc(  # noqa: C901
    y,
    sr,
    order=None,
    win_len_ms=30,
    pre_emph=0.0,
    debug: bool = False,
):
    """
    LPC-based formant estimator with:
      - adaptive LPC order
      - bandwidth expansion
      - pole filtering
      - spectral fallback

    Returns:
        (f1, f2, f3) or (f1, f2, f3, debug_data) if debug=True.
    """

    # 1. Input validation
    if y is None or sr is None or sr <= 0:
        return (None, None, None) if not debug else (None, None, None, [])

    y = np.asarray(y, dtype=float).flatten()
    if y.size == 0:
        return (None, None, None) if not debug else (None, None, None, [])

    # 2. Pre-emphasis
    if pre_emph and pre_emph > 0:
        y = np.append(y[0], y[1:] - pre_emph * y[:-1])

    # 3. Windowing
    win_len = int(sr * win_len_ms / 1000)
    win_len = min(win_len, len(y))
    if win_len < 256:
        return (None, None, None) if not debug else (None, None, None, [])

    y = y[:win_len] * np.hamming(win_len)

    # 4. Adaptive LPC order
    if order is None:
        # Classic heuristic: ~ 2 + sr/1000, clamped
        order = max(8, min(16, int(2 + sr / 1000)))

    if len(y) < 3 * order:
        return (None, None, None) if not debug else (None, None, None, [])

    # 5. Autocorrelation + Levinson–Durbin
    try:
        R_full = np.correlate(y, y, mode="full")
        mid = len(R_full) // 2
        r = R_full[mid: mid + order + 1]
        A = _levinson_durbin(r, order)
        if A is None:
            raise RuntimeError("Levinson–Durbin failed")
    except Exception:
        return (None, None, None) if not debug else (None, None, None, [])

    # 6. Bandwidth expansion (slight)
    bw = 0.994
    A = A * (bw ** np.arange(len(A)))

    # 7. Root finding
    try:
        roots = np.roots(A)
    except Exception:
        return (None, None, None) if not debug else (None, None, None, [])

    # Keep all roots; we’ll filter by freq/bandwidth
    if roots.size == 0:
        return (None, None, None) if not debug else (None, None, None, [])

    # 8. Convert roots → frequencies + bandwidths
    ang = np.angle(roots)
    freqs = ang * (sr / (2 * np.pi))
    bw_vals = -0.5 * (sr / np.pi) * np.log(np.abs(roots))

    # 9. Filter poles (physiological + bandwidth)
    mask = (
        (freqs > 80) &
        (freqs < 4000) &
        (bw_vals > 0) &
        (bw_vals < 600)
    )
    freqs = freqs[mask]

    if freqs.size == 0:
        # spectral fallback
        peaks, _ = smoothed_spectrum_peaks(y, sr)
        if len(peaks) >= 2:
            peaks = np.sort(peaks)
            f1 = peaks[0]
            f2 = peaks[1]
            f3 = peaks[2] if len(peaks) > 2 else None
            return (f1, f2, f3) if not debug else (f1, f2, f3, peaks.tolist())
        return (None, None, None) if not debug else (None, None, None, [])

    freqs = np.sort(freqs.real)

    # 10. Extract F1/F2/F3
    plausible = freqs.copy()
    f1 = plausible[0] if len(plausible) > 0 else None
    f2 = plausible[1] if len(plausible) > 1 else None
    f3 = plausible[2] if len(plausible) > 2 else None

    if debug:
        return f1, f2, f3, plausible.tolist()
    return f1, f2, f3


def smoothed_spectrum_peaks(
    frame: NDArray[np.float64],
    sr: int,
    lifter_cut: int = 60,
    nfft: int = 8192,
    low: int = 50,
    high: int = 4000,
    peak_thresh: float = 0.02,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Cepstral lifter smoothing to extract spectral-envelope peaks.
    Returns (peak_freqs_array, heights_array).
    """
    try:
        win = frame * np.hamming(len(frame))
        X = np.abs(np.fft.rfft(win, n=nfft))
        logX = np.log(X + 1e-12)
        cep = np.fft.irfft(logX)
        lifter_cut = max(1, lifter_cut)
        cep[lifter_cut:-lifter_cut] = 0
        smooth_log = np.fft.rfft(cep, n=nfft).real
        env = np.exp(smooth_log)
        freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return np.array([]), np.array([])
        return _extract_peak_data(env, freqs, mask, peak_thresh)

    except Exception as e:
        logger.exception("smoothed_spectrum_peaks failed: %s", e)
        return np.array([]), np.array([])


def _extract_peak_data(
    env: np.ndarray, freqs: np.ndarray, mask: np.ndarray, peak_thresh: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared helper to extract peak frequencies and heights."""
    env_masked = env[mask]
    if env_masked.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    peaks, _ = find_peaks(env_masked, height=np.max(env_masked) * peak_thresh)

    idx_list: List[int] = [int(p) for p in np.asarray(peaks).ravel()]
    if not idx_list:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    peak_freqs = np.asarray(
        np.round(freqs[mask][idx_list], 1), dtype=np.float64
    )
    heights = np.asarray(np.round(env_masked[idx_list], 2), dtype=np.float64)
    return peak_freqs, heights
