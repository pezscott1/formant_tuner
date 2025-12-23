# analysis.utils
import numpy as np


def safe_array(x):
    """
    Convert x into a safe 1D float64 numpy array.
    Returns a zero-length array if x is None or invalid.
    """
    if x is None:
        return np.zeros(0, dtype=float)

    try:
        arr = np.asarray(x, dtype=float).flatten()
        arr = arr[np.isfinite(arr)]  # remove NaN/inf
        return arr
    except Exception:
        return np.zeros(0, dtype=float)


def _extract_audio_array(item):
    """
    Robustly extract a 1D float numpy array from:
      - ndarray
      - list/tuple
      - dict with common audio keys
      - nested dicts
      - scalars
      - memoryviews / buffers

    Always returns a 1D float64 array or None if impossible.
    """

    if item is None:
        return None

    try:
        # Direct ndarray
        if isinstance(item, np.ndarray):
            arr = item.astype(float).flatten()
            return arr[np.isfinite(arr)]

        # List/tuple
        if isinstance(item, (list, tuple)):
            arr = np.atleast_1d(item).astype(float).flatten()
            return arr[np.isfinite(arr)]

        # Bytes-like (rare but possible)
        if isinstance(item, (bytes, bytearray, memoryview)):
            arr = np.frombuffer(item, dtype=np.float32).astype(float)
            return arr[np.isfinite(arr)]

        # Dict: check common audio keys
        if isinstance(item, dict):
            for key in ("data", "audio", "frame", "samples", "chunk", "segment"):
                if key in item:
                    return safe_array(item[key])

            # Fallback: search nested values
            for v in item.values():
                out = _extract_audio_array(v)
                if out is not None and out.size > 0:
                    return out

            return None

        # Scalar fallback
        arr = np.atleast_1d(item).astype(float).flatten()
        return arr[np.isfinite(arr)]

    except Exception:
        return None
