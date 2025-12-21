import numpy as np
from typing import Tuple, List
from scipy.signal import find_peaks
from numpy.typing import NDArray
import logging

logger = logging.getLogger(__name__)


def estimate_formants_lpc(
    y,
    sr,
    order=None,
    win_len_ms=30,
    pre_emph=0.0,
    debug=False
):
    """
    Robust LPC-based formant estimator.
    Returns (f1, f2, f3) or (f1, f2, f3, candidates) if debug=True.
    """

    # -----------------------------
    # 1. Input validation
    # -----------------------------
    if y is None or sr is None or sr <= 0:
        return (None, None, None) if not debug else (None, None, None, [])

    y = np.asarray(y, dtype=float).flatten()
    if y.size == 0:
        return (None, None, None) if not debug else (None, None, None, [])

    # -----------------------------
    # 2. Pre-emphasis (optional)
    # -----------------------------
    if pre_emph and pre_emph > 0:
        y = np.append(y[0], y[1:] - pre_emph * y[:-1])

    # -----------------------------
    # 3. Windowing
    # -----------------------------
    win_len = int(sr * win_len_ms / 1000)
    if win_len > len(y):
        win_len = len(y)

    y = y[:win_len] * np.hamming(win_len)

    # -----------------------------
    # 4. LPC order selection
    # -----------------------------
    if order is None:
        # Stable for synthetic vowels and real speech
        order = 12

    # Short-signal guard
    if len(y) < 3 * order:
        return (None, None, None) if not debug else (None, None, None, [])

    # -----------------------------
    # 5. Autocorrelation LPC
    # -----------------------------
    try:
        R = np.correlate(y, y, mode="full")
        mid = len(R) // 2
        R = R[mid: mid + order + 1]

        # Solve Yule-Walker
        A = np.zeros(order + 1)
        A[0] = 1.0

        # Toeplitz matrix
        T = np.zeros((order, order))
        for i in range(order):
            T[i, :] = R[abs(i - np.arange(order))]

        rhs = -R[1: order + 1]
        coeffs = np.linalg.solve(T, rhs)
        A[1:] = coeffs

    except Exception:
        return (None, None, None) if not debug else (None, None, None, [])

    # -----------------------------
    # 6. Bandwidth expansion
    # -----------------------------
    bw = 0.994
    A = A * (bw ** np.arange(len(A)))

    # -----------------------------
    # 7. Root finding
    # -----------------------------
    try:
        roots = np.roots(A)
    except Exception:
        return (None, None, None) if not debug else (None, None, None, [])

    # Keep only complex-conjugate pairs
    roots = roots[np.imag(roots) >= 0.001]

    # -----------------------------
    # 8. Convert roots to frequencies
    # -----------------------------
    ang = np.angle(roots)
    freqs = ang * (sr / (2 * np.pi))

    # Bandwidths
    bw_vals = -0.5 * (sr / np.pi) * np.log(np.abs(roots))

    # -----------------------------
    # 9. Filter poles
    # -----------------------------
    mask = (
        (freqs > 90) & (freqs < 4000) &  # speech band
        (bw_vals < 700)                 # reject unstable poles
    )
    freqs = freqs[mask]

    if freqs.size == 0:
        return (None, None, None) if not debug else (None, None, None, [])

    freqs = np.sort(freqs)

    # -----------------------------
    # 10. Extract formants
    # -----------------------------
    f1 = freqs[0] if len(freqs) > 0 else None
    f2 = freqs[1] if len(freqs) > 1 else None
    f3 = freqs[2] if len(freqs) > 2 else None

    if debug:
        return f1, f2, f3, freqs.tolist()
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

    except Exception as e:  # noqa: E722
        logger.exception("smoothed_spectrum_peaks failed: %s", e)
        return np.array([]), np.array([])


def _extract_peak_data(
    env: np.ndarray, freqs: np.ndarray, mask: np.ndarray, peak_thresh: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared helper to extract peak frequencies and heights."""
    peaks, _ = find_peaks(env[mask], height=np.max(env[mask]) * peak_thresh)

    # Convert to a plain Python list[int],
    # so the IDE/type checker accepts it as Iterable/Sized
    idx_list: List[int] = [int(p) for p in np.asarray(peaks).ravel()]

    if not idx_list:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # Use list indexing (NumPy accepts list of ints)
    peak_freqs = np.asarray(
        np.round(freqs[mask][idx_list], 1), dtype=np.float64
    )
    heights = np.asarray(np.round(env[mask][idx_list], 2), dtype=np.float64)
    return peak_freqs, heights
