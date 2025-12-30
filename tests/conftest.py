# tests/conftest.py
from types import SimpleNamespace
from unittest.mock import MagicMock
import matplotlib
matplotlib.use("Agg")


def pytest_runtest_call(item):
    if item.get_closest_marker("noqt"):
        return item.runtest()


class FakeWindow:
    def __init__(self):
        # Axes
        self.ax_chart = MagicMock()
        self.ax_vowel = MagicMock()

        # Canvas
        self.canvas = MagicMock()
        self.canvas.draw_idle = MagicMock()

        # Analyzer + hybrid metadata
        self.analyzer = MagicMock()
        self.analyzer.get_latest_raw.return_value = {
            "method": "hybrid",
            "confidence": 0.9,
            "segment": [0.1, 0.2, 0.3, 0.4],
        }

        # Smoother
        self.analyzer.formant_smoother = SimpleNamespace(formants_stable=True)

        # Sample rate
        self.sample_rate = 48000

        # Artists created by update_vowel_chart
        self.vowel_measured_artist = None
        self.vowel_line_artist = None
