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


def test_engine_handles_nan_input():
    eng = FormantAnalysisEngine()
    eng.process_frame(np.array([np.nan, np.nan, np.nan]), 44100)
    raw = eng.get_latest_raw()

    assert raw["f0"] is None
    assert raw["formants"] == (None, None, None)


def test_engine_handles_silence():
    eng = FormantAnalysisEngine()
    eng.process_frame(np.zeros(2048), 44100)
    raw = eng.get_latest_raw()

    assert raw["f0"] is None
    assert raw["formants"] == (None, None, None)
