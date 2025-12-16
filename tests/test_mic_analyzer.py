import time
import queue
from mic_analyzer import MicAnalyzer
from tests.conftest import synth_vowel


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
