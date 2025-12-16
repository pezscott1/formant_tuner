import time
import queue
from mic_analyzer import MicAnalyzer, results_queue
from tests.conftest import synth_vowel

def test_worker_posts_status(monkeypatch):
    sr = 16000
    # simple providers
    vowel_provider = lambda: "a"
    tol_provider = lambda: 50
    pitch_provider = lambda: 220.0

    ma = MicAnalyzer(vowel_provider, tol_provider, pitch_provider,
                     sample_rate=sr, frame_ms=40, analyzer=None,
                     processing_window_s=0.25, lpc_win_ms=30, processing_queue_max=2, rms_gate=1e-8, debug=True)
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
            status = results_queue.get_nowait()
        except queue.Empty:
            pass
        assert status is not None
        assert "formants" in status and isinstance(status["formants"], tuple)
    finally:
        ma.stop()