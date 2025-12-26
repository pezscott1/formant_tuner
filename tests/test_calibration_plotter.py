import numpy as np
from unittest.mock import MagicMock
from calibration.plotter import update_artists


# ---------------------------------------------------------
# Helper FakePlotter with all required attributes
# ---------------------------------------------------------
class FakePlotter:
    def __init__(self):
        # Axes
        self.ax_spec = MagicMock()
        self.ax_vowel = MagicMock()

        # Canvas
        self.canvas = MagicMock()

        # State
        self.state = MagicMock(phase="capture")

        # Analyzer + smoother
        self.analyzer = MagicMock()
        self.analyzer.formant_smoother = MagicMock(
            formants_stable=True,
            _stability_score=0.0,
        )
        self.latest_raw = {"confidence": 1.0, "method": "lpc"}

        # Calibration
        self.current_vowel_calibration = {"f1": 500, "f2": 1500}

        # Internal plotter state
        self._spec_mesh = None
        self._spec_colorbar = None
        self._vowel_scatters = {}
        self._vowel_colors = {"i": "blue", "a": "red"}
        self._last_draw = 0
        self._min_draw_interval = 0


# ---------------------------------------------------------
# Basic spectrogram rendering
# ---------------------------------------------------------
def test_update_artists_basic_spectrogram():
    plotter = FakePlotter()

    freqs = np.linspace(0, 5000, 256)
    times = np.linspace(0, 1.0, 100)
    S = np.abs(np.random.randn(256, 100)) ** 2

    update_artists(plotter, freqs, times, S, f1=None, f2=None, vowel=None)

    assert plotter._spec_mesh is not None
    # Title is dynamic, so only check prefix
    title = plotter.ax_spec.set_title.call_args[0][0]
    assert plotter.ax_spec.get_title().startswith("Spectrogram")


# ---------------------------------------------------------
# Vowel scatter creation when stable + confident
# ---------------------------------------------------------
def test_update_artists_with_vowel_scatter():
    plotter = FakePlotter()

    freqs = np.linspace(0, 5000, 256)
    times = np.linspace(0, 1.0, 100)
    S = np.abs(np.random.randn(256, 100)) ** 2

    update_artists(plotter, freqs, times, S, f1=500, f2=1500, vowel="i")

    assert "i" in plotter._vowel_scatters


# ---------------------------------------------------------
# Vowel scatter suppressed when unstable or low confidence
# ---------------------------------------------------------
def test_update_artists_suppresses_scatter_when_unstable():
    plotter = FakePlotter()
    plotter.analyzer.formant_smoother.formants_stable = False

    freqs = np.linspace(0, 5000, 256)
    times = np.linspace(0, 1.0, 100)
    S = np.abs(np.random.randn(256, 100)) ** 2

    update_artists(plotter, freqs, times, S, f1=500, f2=1500, vowel="i")

    assert "i" in plotter._vowel_scatters


def test_update_artists_suppresses_scatter_when_low_confidence():
    plotter = FakePlotter()
    plotter.latest_raw["confidence"] = 0.1  # below threshold

    freqs = np.linspace(0, 5000, 256)
    times = np.linspace(0, 1.0, 100)
    S = np.abs(np.random.randn(256, 100)) ** 2

    update_artists(plotter, freqs, times, S, f1=500, f2=1500, vowel="i")

    assert "i" not in plotter._vowel_scatters
