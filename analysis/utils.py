import numpy as np


def safe_array(x):
    if x is None:
        return np.zeros(1)
    return np.asarray(x, dtype=float).flatten()


def _extract_audio_array(item):
    """Return a 1D float numpy array from ndarray, list, or dict wrappers."""
    if item is None:
        return None
    try:
        if isinstance(item, np.ndarray):
            return item.astype(float).flatten()
        if isinstance(item, (list, tuple)):
            return np.atleast_1d(item).astype(float).flatten()
        if isinstance(item, dict):
            for key in (
                "data",
                "audio",
                "frame",
                "samples",
                "chunk",
                "segment",
            ):
                if key in item:
                    return np.atleast_1d(item[key]).astype(float).flatten()
            for v in item.values():
                if isinstance(v, np.ndarray):
                    return v.astype(float).flatten()
                if isinstance(v, (list, tuple)):
                    return np.atleast_1d(v).astype(float).flatten()
        return np.atleast_1d(item).astype(float).flatten()
    except Exception:  # noqa: E722
        return None
