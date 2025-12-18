import time
import queue
from mic_analyzer import MicAnalyzer
from tests.conftest import synth_vowel
import pytest
import numpy as np
from unittest.mock import MagicMock


def vowel_provider():
    return "a"


def tol_provider():
    return 50


def pitch_provider():
    return 220.0


def test_worker_posts_status(monkeypatch):
    sr = 16000
    pitch = pitch_provider()
    vowel = vowel_provider()
    tol = tol_provider()

    ma = MicAnalyzer(
        vowel,
        tol,
        pitch,
        sample_rate=sr,
        frame_ms=40,
        analyzer=None,
        processing_window_s=0.25,
        lpc_win_ms=30,
        processing_queue_max=2,
        rms_gate=1e-8,
        debug=True,
    )

    # start worker and stream (do not open real stream)
    ma.start()
    try:
        # push a synthetic segment into processing queue directly
        seg = synth_vowel([700, 1200], sr=sr, dur=0.3, f0=130.0)
        ma.processing_queue.put_nowait(seg)

        # wait for worker to process
        time.sleep(0.5)

        # check results_queue has at least one status
        status = None
        try:
            status = ma.results_queue.get_nowait()
        except queue.Empty:
            pass

        assert status is not None
        assert "formants" in status and isinstance(status["formants"], tuple)
    finally:
        ma.stop()


def test_micanalyzer_start_failure(monkeypatch):
    """Simulate InputStream.start() raising an exception to hit error handling."""
    from mic_analyzer import MicAnalyzer

    fake_mic = MagicMock()
    fake_mic.sample_rate = 16000
    fake_tol_provider = MagicMock()
    fake_pitch_provider = MagicMock()

    class FailingStream:
        def __init__(self, **kwargs): pass
        def start(self): raise RuntimeError("Stream failed")
        def stop(self): pass
        def close(self): pass

    monkeypatch.setattr("mic_analyzer.sd.InputStream", FailingStream)

    analyzer = MicAnalyzer(fake_mic, fake_tol_provider, fake_pitch_provider)
    analyzer.start()
    # After failure, is_running should be False and stream reset
    assert analyzer.is_running is False
    assert analyzer.stream is None


def test_micanalyzer_double_stop(monkeypatch):
    """Call stop twice to cover cleanup branches."""
    from mic_analyzer import MicAnalyzer

    fake_mic = MagicMock()
    fake_mic.sample_rate = 16000
    fake_tol_provider = MagicMock()
    fake_pitch_provider = MagicMock()

    class DummyStream:
        def __init__(self, **kwargs): self.active = True
        def start(self): pass
        def stop(self): self.active = False
        def close(self): pass

    monkeypatch.setattr("mic_analyzer.sd.InputStream", DummyStream)

    analyzer = MicAnalyzer(fake_mic, fake_tol_provider, fake_pitch_provider)
    analyzer.start()
    analyzer.stop()
    # Call stop again to exercise error branches
    analyzer.stop()
    assert analyzer.is_running is False
    assert analyzer.stream is None


def test_micanalyzer_worker_cleanup(monkeypatch):
    """Ensure worker thread is joined and cleaned up."""
    from mic_analyzer import MicAnalyzer

    fake_mic = MagicMock()
    fake_mic.sample_rate = 16000
    fake_tol_provider = MagicMock()
    fake_pitch_provider = MagicMock()

    class DummyStream:
        def __init__(self, **kwargs): self.active = True
        def start(self): pass
        def stop(self): self.active = False
        def close(self): pass

    monkeypatch.setattr("mic_analyzer.sd.InputStream", DummyStream)

    analyzer = MicAnalyzer(fake_mic, fake_tol_provider, fake_pitch_provider)
    analyzer.start()
    # Stop should join worker and clear it
    analyzer.stop()
    assert analyzer._worker is None
    assert analyzer.is_running is False
