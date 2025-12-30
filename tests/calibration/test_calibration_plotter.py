# tests/test_calibration_plotter.py

from types import SimpleNamespace
from unittest.mock import MagicMock


class FakePlotter:
    def __init__(self):
        # Calibration state
        self.state = SimpleNamespace(phase="capture")

        # Analyzer (hybrid metadata)
        self.analyzer = MagicMock()
        self.analyzer.get_latest_raw.return_value = {
            "method": "hybrid",
            "confidence": 0.9,
        }

        # Formant smoother
        self.formant_smoother = SimpleNamespace(formants_stable=True)

        # Session data for target lines
        self.session = SimpleNamespace(
            data={
                "i": {"f1": 300, "f2": 2200},
                "a": {"f1": 700, "f2": 1100},
            }
        )

        # Axes
        self.ax_spec = MagicMock()
        self.ax_vowel = MagicMock()

        # Canvas
        self.canvas = MagicMock()
        self.canvas.draw_idle = MagicMock()

        # Internal plotter state
        self._spec_mesh = MagicMock()
        self._spec_colorbar = MagicMock()
        self._vowel_scatters = {}
        self._vowel_colors = {"i": "blue", "a": "red"}

        # Draw throttling
        self._last_draw = 0
        self._min_draw_interval = 0

        # Legacy attributes (silence PyCharm)
        self.spec = MagicMock()
        self.vowel_scatter = MagicMock()
