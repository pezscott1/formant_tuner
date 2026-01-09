import numpy as np
from tuner.live_analyzer import LiveAnalyzer

# ----------------------------------------------------------------------
# Dummy components for isolation
# ----------------------------------------------------------------------


class DummyPitch:
    def __init__(self):
        self.current = None
        self.audio_buffer = []

    def update(self, f0, confidence):
        # Return f0 * 2 for easy verification
        return None if f0 is None else f0 * 2


class DummyFormants:
    def __init__(self):
        self.buf_f1 = [4, 5]
        self.buf_f2 = [4, 5]
        self.buf_f3 = [4, 5]
        self.formants_stable = True
        self._stability_score = 123

        class DummyStability:
            def reset(self_inner): pass
        self.stability = DummyStability()

    def update(self, f1, f2, f3, confidence):
        # Return formants + 100 for easy verification
        return (
            None if f1 is None else f1 + 100,
            None if f2 is None else f2 + 100,
            None if f3 is None else f3 + 100,
        )


class DummyLabel:
    def __init__(self):
        self.current = None
        self.last = None
        self.counter = 0

    def update(self, label, confidence):
        # Return uppercase label for easy verification
        return None if label is None else label.upper()


class DummyEngine:
    def __init__(self, return_value):
        self.return_value = return_value
        self.calls = 0

    def process_frame(self, segment, sr):
        self.calls += 1
        return self.return_value


class DummyEngineError:
    def process_frame(self, segment, sr):
        raise RuntimeError("boom")


# ----------------------------------------------------------------------
# process_raw
# ----------------------------------------------------------------------

def test_process_raw_basic():
    engine = DummyEngine({})
    pitch = DummyPitch()
    formants = DummyFormants()
    labels = DummyLabel()

    la = LiveAnalyzer(engine, pitch, formants, labels)

    raw = {
        "f0": 100,
        "formants": (500, 1500, 2500),
        "vowel": "a",
        "confidence": 0.8,
        "vowel_score": 0.9,
        "resonance_score": 0.7,
        "overall": 0.8,
        "fb_f1": 111,
        "fb_f2": 222,
        "method": "lpc",
        "roots": [1, 2],
        "peaks": [3, 4],
        "lpc_order": 12,
        "lpc_debug": {"x": 1},
        "segment": np.zeros(10),
    }

    out = la.process_raw(raw)

    # Pitch smoothing
    assert out["f0"] == 200  # 100 * 2

    # Formant smoothing
    assert out["formants"] == (600, 1600, 2600)

    # Vowel smoothing
    assert out["vowel"] == "A"

    # Raw vowel guess
    assert out["vowel_guess"] == "a"

    # Stability
    assert out["stable"] is True
    assert out["stability_score"] == 123.0

    # Fallback fields
    assert out["fb_f1"] == 111
    assert out["fb_f2"] == 222

    # Debug fields
    assert out["method"] == "lpc"
    assert out["roots"] == [1, 2]
    assert out["peaks"] == [3, 4]

    # Latest processed stored
    assert la.get_latest_processed() == out
    assert la.get_latest_raw() == raw


# ----------------------------------------------------------------------
# submit_audio_segment
# ----------------------------------------------------------------------

def test_submit_audio_segment_none():
    la = LiveAnalyzer(None, DummyPitch(), DummyFormants(), DummyLabel())
    la.submit_audio_segment(None)  # should not crash
    assert la._audio_queue.empty()


def test_submit_audio_segment_overflow():
    la = LiveAnalyzer(None, DummyPitch(), DummyFormants(), DummyLabel())

    # Fill queue
    for _ in range(8):
        la.submit_audio_segment(np.zeros(1))

    # Overflow should not block or crash
    la.submit_audio_segment(np.zeros(1))
    assert la._audio_queue.qsize() == 8


# ----------------------------------------------------------------------
# worker loop: engine success
# ----------------------------------------------------------------------

def test_worker_loop_success():
    raw = {"f0": 50, "formants": (100, 200, 300)}
    engine = DummyEngine(raw)
    la = LiveAnalyzer(engine, DummyPitch(), DummyFormants(), DummyLabel())

    la.submit_audio_segment(np.zeros(1))
    la.start_worker()

    # Give thread time to run
    import time
    time.sleep(0.1)

    la.stop_worker()

    # Engine should have been called
    assert engine.calls > 0

    # Processed queue should have one item
    assert not la.processed_queue.empty()


# ----------------------------------------------------------------------
# worker loop: engine error
# ----------------------------------------------------------------------

def test_worker_loop_engine_error():
    engine = DummyEngineError()
    la = LiveAnalyzer(engine, DummyPitch(), DummyFormants(), DummyLabel())

    la.submit_audio_segment(np.zeros(1))
    la.start_worker()

    import time
    time.sleep(0.1)

    la.stop_worker()

    # Engine error should not crash worker
    # Queue should remain empty because processing never succeeded
    assert la.processed_queue.empty()


# ----------------------------------------------------------------------
# worker loop: processed queue overflow
# ----------------------------------------------------------------------

def test_worker_loop_processed_overflow():
    raw = {"f0": 10, "formants": (20, 30, 40)}
    engine = DummyEngine(raw)
    la = LiveAnalyzer(engine, DummyPitch(), DummyFormants(), DummyLabel())

    # Fill processed queue
    for _ in range(8):
        la.processed_queue.put({"dummy": True})

    la.submit_audio_segment(np.zeros(1))
    la.start_worker()

    import time
    time.sleep(0.1)

    la.stop_worker()

    # Queue should still be full but oldest item replaced
    assert la.processed_queue.qsize() == 8


# ----------------------------------------------------------------------
# reset()
# ----------------------------------------------------------------------

def test_reset():
    pitch = DummyPitch()
    formants = DummyFormants()
    labels = DummyLabel()

    la = LiveAnalyzer(None, pitch, formants, labels)

    pitch.current = 123
    pitch.audio_buffer = [1, 2, 3]
    formants.buffer = [4, 5]
    labels.current = "x"
    labels.last = "y"
    labels.counter = 99

    la.reset()

    assert formants.buf_f1 == []
    assert formants.buf_f2 == []
    assert formants.buf_f3 == []
    assert formants.formants_stable is False
    assert formants._stability_score == float("inf")
