import numpy as np
from analysis.utils import safe_array, _extract_audio_array


# -----------------------------
# Tests for safe_array
# -----------------------------

def test_safe_array_none():
    out = safe_array(None)
    assert isinstance(out, np.ndarray)
    assert out.size == 0


def test_safe_array_list():
    out = safe_array([1, 2, 3])
    assert np.allclose(out, [1.0, 2.0, 3.0])


def test_safe_array_filters_nan_inf():
    out = safe_array([1, np.nan, np.inf, 2])
    assert np.allclose(out, [1.0, 2.0])


def test_safe_array_invalid_type():
    class Bad:
        pass
    out = safe_array(Bad())   # triggers exception → zero-length
    assert out.size == 0


# -----------------------------
# Tests for _extract_audio_array
# -----------------------------

def test_extract_none():
    assert _extract_audio_array(None) is None


def test_extract_ndarray():
    arr = np.array([1.0, 2.0, np.nan])
    out = _extract_audio_array(arr)
    assert np.allclose(out, [1.0, 2.0])


def test_extract_list():
    out = _extract_audio_array([1, 2, 3])
    assert np.allclose(out, [1.0, 2.0, 3.0])


def test_extract_bytes_like():
    buf = (np.array([1.0, 2.0], dtype=np.float32)).tobytes()
    out = _extract_audio_array(buf)
    assert np.allclose(out, [1.0, 2.0])


def test_extract_dict_direct_key():
    d = {"audio": [1, 2, np.nan]}
    out = _extract_audio_array(d)
    assert np.allclose(out, [1.0, 2.0])


def test_extract_dict_nested():
    d = {"outer": {"inner": {"samples": [5, 6, np.nan]}}}
    out = _extract_audio_array(d)
    assert np.allclose(out, [5.0, 6.0])


def test_extract_dict_no_valid_values():
    d = {"x": {}, "y": []}
    out = _extract_audio_array(d)
    assert out is None


def test_extract_scalar():
    out = _extract_audio_array(3.14)
    assert np.allclose(out, [3.14])


def test_extract_invalid_type():
    class Bad:
        pass
    out = _extract_audio_array(Bad())  # triggers exception → None
    assert out is None
