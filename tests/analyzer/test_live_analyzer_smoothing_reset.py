from unittest.mock import MagicMock
from tuner.live_analyzer import LiveAnalyzer


class DummyPitchSmoother:
    def __init__(self):
        self.current = None
        self.audio_buffer = []

    def update(self, f0, _confidence=1.0):
        self.current = f0
        return f0


class DummyFormantSmoother:
    def __init__(self):
        self.buffer = []

    def update(self, f1, f2, _confidence=1.0):
        self.buffer.append((f1, f2))
        return f1, f2


class DummyLabelSmoother:
    def __init__(self):
        self.current = None
        self.last = None
        self.counter = 0

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

    # Pre-reset state: all smoothers start empty in the new architecture
    assert pitch_s.current is None
    assert pitch_s.audio_buffer == []
    assert formant_s.buffer == []
    assert label_s.current is None
    assert label_s.last is None
    assert label_s.counter == 0

    # Perform reset
    analyzer.reset()

    # After reset, everything should still be cleared
    assert pitch_s.current is None
    assert pitch_s.audio_buffer == []
    assert formant_s.buffer == []
    assert label_s.current is None
    assert label_s.last is None
    assert label_s.counter == 0
