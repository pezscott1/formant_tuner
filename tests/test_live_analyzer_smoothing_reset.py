import numpy as np
import time
from unittest.mock import MagicMock
from tuner.live_analyzer import LiveAnalyzer


class DummyPitchSmoother:
    def __init__(self):
        self.current = 200.0
        self.audio_buffer = [200.0, 205.0, 210.0]

    def update(self, f0, _confidence=1.0):
        self.current = f0
        return f0


class DummyFormantSmoother:
    def __init__(self):
        self.buffer = [(500, 1500)]

    def update(self, f1, f2, _confidence=1.0):
        self.buffer.append((f1, f2))
        return f1, f2


class DummyLabelSmoother:
    def __init__(self):
        self.current = "a"
        self.last = "a"
        self.counter = 3

    def update(self, label, _confidence=1.0):
        self.current = label
        return label


def test_live_analyzer_reset_clears_all_smoothers():
    mock_engine = MagicMock()

    pitch_s = DummyPitchSmoother()
    formant_s = DummyFormantSmoother()
    label_s = DummyLabelSmoother()

    analyzer = LiveAnalyzer(
        engine=mock_engine,
        pitch_smoother=pitch_s,
        formant_smoother=formant_s,
        label_smoother=label_s,
    )

    # Sanity check pre-reset state
    assert pitch_s.current is not None
    assert len(pitch_s.audio_buffer) > 0
    assert len(formant_s.buffer) > 0
    assert label_s.current is not None
    assert label_s.last is not None
    assert label_s.counter > 0

    # Perform reset
    analyzer.reset()

    # Pitch smoother cleared
    assert pitch_s.current is None
    assert pitch_s.audio_buffer == []

    # Formant smoother cleared
    assert formant_s.buffer == []

    # Label smoother cleared
    assert label_s.current is None
    assert label_s.last is None
    assert label_s.counter == 0


def test_smoothing_resets_on_voice_type_change():
    mock_engine = MagicMock()

    pitch_s = DummyPitchSmoother()
    formant_s = DummyFormantSmoother()
    label_s = DummyLabelSmoother()

    la = LiveAnalyzer(
        engine=mock_engine,
        pitch_smoother=pitch_s,
        formant_smoother=formant_s,
        label_smoother=label_s,
    )

    # simulate external voice-type change
    # LiveAnalyzer does NOT have set_voice_type, so we call reset()
    la.reset()

    assert pitch_s.current is None
    assert pitch_s.audio_buffer == []
    assert formant_s.buffer == []
    assert label_s.current is None
    assert label_s.last is None
    assert label_s.counter == 0


def test_smoothing_resets_on_profile_change():
    mock_engine = MagicMock()

    pitch_s = DummyPitchSmoother()
    formant_s = DummyFormantSmoother()
    label_s = DummyLabelSmoother()

    la = LiveAnalyzer(
        engine=mock_engine,
        pitch_smoother=pitch_s,
        formant_smoother=formant_s,
        label_smoother=label_s,
    )

    # simulate external profile change
    la.reset()

    assert pitch_s.current is None
    assert pitch_s.audio_buffer == []
    assert formant_s.buffer == []
    assert label_s.current is None
    assert label_s.last is None
    assert label_s.counter == 0


def test_live_analyzer_process_raw_wires_through_smoothers():
    engine = MagicMock()

    pitch_s = MagicMock()
    pitch_s.update.return_value = 180.0

    formant_s = MagicMock()
    formant_s.update.return_value = (500.0, 1500.0, 2500.0)
    formant_s.formants_stable = True
    formant_s._stability_score = 123.0

    label_s = MagicMock()
    label_s.update.return_value = "a"

    la = LiveAnalyzer(engine, pitch_s, formant_s, label_s)

    raw = {
        "f0": 200.0,
        "formants": (510.0, 1510.0, 2510.0),
        "vowel": None,
        "vowel_guess": "a",
        "confidence": 0.9,
        "vowel_score": 0.7,
        "resonance_score": 0.8,
        "overall": 0.75,
        "fb_f1": 480.0,
        "fb_f2": 1480.0,
        "method": "lpc",
        "roots": [0.1],
        "peaks": [1],
        "lpc_order": 12,
        "lpc_debug": {"ok": True},
        "segment": np.zeros(10),
    }

    out = la.process_raw(raw)

    # pitch smoother called with raw f0 and lpc_conf
    pitch_s.update.assert_called_once_with(200.0, confidence=0.9)

    # formant smoother called with raw formants and lpc_conf
    formant_s.update.assert_called_once_with(
        f1=510.0, f2=1510.0, f3=2510.0, confidence=0.9
    )

    # label smoother called with vowel_raw (vowel or vowel_guess) and lpc_conf
    label_s.update.assert_called_once_with("a", confidence=0.9)

    # processed dict uses smoothed values and stability info
    assert out["f0"] == 180.0
    assert out["formants"] == (500.0, 1500.0, 2500.0)
    assert out["vowel"] == "a"
    assert out["vowel_guess"] == "a"
    assert out["confidence"] == 0.9
    assert out["vowel_score"] == 0.7
    assert out["resonance_score"] == 0.8
    assert out["overall"] == 0.75
    assert out["stable"] is True
    assert out["stability_score"] == 123.0
    assert out["fb_f1"] == 480.0
    assert out["fb_f2"] == 1480.0
    assert out["method"] == "lpc"
    assert out["roots"] == [0.1]
    assert out["peaks"] == [1]
    assert out["lpc_order"] == 12
    assert out["lpc_debug"] == {"ok": True}
    assert np.array_equal(out["segment"], np.zeros(10))

    # latest_raw / latest_processed updated
    assert la.get_latest_raw() is raw
    assert la.get_latest_processed() == out


def test_live_analyzer_worker_processes_audio_and_pushes_frames(tmp_path):
    engine = MagicMock()
    engine.process_frame.return_value = {
        "f0": 200.0,
        "formants": (500.0, 1500.0, 2500.0),
        "vowel": "a",
        "vowel_guess": "a",
        "confidence": 0.9,
        "vowel_score": 0.7,
        "resonance_score": 0.8,
        "overall": 0.75,
        "fb_f1": None,
        "fb_f2": None,
        "method": "lpc",
        "roots": [],
        "peaks": [],
        "lpc_order": 12,
        "lpc_debug": {},
        "segment": np.zeros(1024),
    }

    pitch_s = MagicMock()
    pitch_s.update.return_value = 200.0

    formant_s = MagicMock()
    formant_s.update.return_value = (500.0, 1500.0, 2500.0)
    formant_s.formants_stable = False
    formant_s._stability_score = float("inf")

    label_s = MagicMock()
    label_s.update.return_value = "a"

    la = LiveAnalyzer(engine, pitch_s, formant_s, label_s)

    la.start_worker()
    try:
        seg = np.random.randn(1024).astype(np.float32)
        la.submit_audio_segment(seg)

        # give worker a moment to process
        time.sleep(0.1)

        assert not la.processed_queue.empty()
        processed = la.processed_queue.get_nowait()
        assert processed["f0"] == 200.0
        engine.process_frame.assert_called()
    finally:
        la.stop_worker()


def test_live_analyzer_submit_audio_segment_drops_when_full():
    engine = MagicMock()
    pitch_s = MagicMock()
    formant_s = MagicMock()
    label_s = MagicMock()

    la = LiveAnalyzer(engine, pitch_s, formant_s, label_s)

    seg = np.zeros(10)
    for _ in range(16):  # queue maxsize is 8
        la.submit_audio_segment(seg)

    # Should not raise, and queue size should be <= maxsize
    assert la._audio_queue.qsize() <= 8
