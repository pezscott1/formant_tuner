
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
