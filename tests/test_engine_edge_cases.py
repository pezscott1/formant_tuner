# tests/test_engine_edge_cases.py
import numpy as np
from analysis.engine import FormantAnalysisEngine


def test_engine_handles_empty_array():
    eng = FormantAnalysisEngine()
    eng.process_frame(np.array([]), 44100)
    raw = eng.get_latest_raw()
    assert raw["f0"] is None
    assert raw["formants"] == (None, None, None)


def test_engine_missing_formants():
    eng = FormantAnalysisEngine()
    eng.process_frame(np.zeros(2048), 44100)
    raw = eng.get_latest_raw()
    assert "formants" in raw


def test_engine_missing_pitch():
    eng = FormantAnalysisEngine()
    eng.process_frame(np.zeros(2048), 44100)
    raw = eng.get_latest_raw()
    assert "f0" in raw


def test_engine_handles_empty_frame():
    eng = FormantAnalysisEngine()
    eng.process_frame(np.array([]), 44100)
    raw = eng.get_latest_raw()

    assert raw["f0"] is None
    assert raw["formants"] == (None, None, None)


def test_engine_handles_short_frame():
    eng = FormantAnalysisEngine()
    eng.process_frame(np.array([0.1, -0.2]), 44100)
    raw = eng.get_latest_raw()

    assert "f0" in raw
    assert "formants" in raw


def test_engine_handles_silence():
    eng = FormantAnalysisEngine()
    eng.process_frame(np.zeros(2048), 44100)
    raw = eng.get_latest_raw()

    assert raw["f0"] is None
    assert raw["formants"] == (None, None, None)


def test_engine_short_frame_fallback():
    eng = FormantAnalysisEngine()
    eng.process_frame(np.array([0.1]), 44100)
    raw = eng.get_latest_raw()
    assert raw["f0"] is None
    assert raw["formants"] == (None, None, None)


def test_engine_nan_frame_fallback():
    eng = FormantAnalysisEngine()
    eng.process_frame(np.array([np.nan, np.nan, np.nan]), 44100)
    raw = eng.get_latest_raw()
    assert raw["f0"] is None
    assert raw["formants"] == (None, None, None)


def test_engine_lpc_failure_path():
    eng = FormantAnalysisEngine()
    # Very short frame → LPC cannot run
    eng.process_frame(np.zeros(10), 44100)
    raw = eng.get_latest_raw()
    assert raw["formants"] == (None, None, None)


def test_engine_pitch_failure_path():
    eng = FormantAnalysisEngine()
    # Silence → pitch estimator returns None
    eng.process_frame(np.zeros(2048), 44100)
    raw = eng.get_latest_raw()
    assert raw["f0"] is None


def test_engine_valid_frame_runs_without_error():
    eng = FormantAnalysisEngine()
    eng.process_frame(np.random.randn(2048), 44100)
    raw = eng.get_latest_raw()
    assert "f0" in raw
    assert "formants" in raw


def test_engine_handles_unprocessable_frame():
    eng = FormantAnalysisEngine()

    # A frame that is long enough to enter processing,
    # but contains only NaNs so LPC, pitch, and formants all fail.
    frame = np.full(2048, np.nan)

    eng.process_frame(frame, 44100)
    raw = eng.get_latest_raw()

    # All fallback branches should trigger
    assert raw["f0"] is None
    assert raw["formants"] == (None, None, None)
    assert raw["fb_f1"] is None
    assert raw["fb_f2"] is None
