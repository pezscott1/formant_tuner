import numpy as np
import matplotlib.pyplot as plt
import types
import time

from calibration.plotter import update_artists


class FakeState:
    def __init__(self, phase="capture"):
        self.phase = phase


class FakePlotter:
    def __init__(self):
        fig, (ax_spec, ax_vowel) = plt.subplots(2, 1)

        self.ax_spec = ax_spec
        self.ax_vowel = ax_vowel
        self.canvas = types.SimpleNamespace(draw_idle=lambda: None)

        self.state = FakeState("capture")

        self._spec_mesh = None
        self._vowel_scatters = {}
        self._vowel_colors = {"i": "red", "a": "blue", "u": "green"}

        self._last_draw = 0
        self._min_draw_interval = 0  # always allow draw


def test_update_artists_basic_spectrogram():
    plotter = FakePlotter()

    freqs = np.linspace(0, 5000, 256)
    times = np.linspace(0, 1.0, 100)
    S = np.abs(np.random.randn(256, 100)) ** 2

    # Should run without error and create a mesh
    update_artists(plotter, freqs, times, S, f1=None, f2=None, vowel=None)

    # Mesh should now exist
    assert plotter._spec_mesh is not None
    assert plotter.ax_spec.get_title() == "Spectrogram"


def test_update_artists_with_vowel_scatter():
    plotter = FakePlotter()

    freqs = np.linspace(0, 5000, 256)
    times = np.linspace(0, 1.0, 100)
    S = np.abs(np.random.randn(256, 100)) ** 2

    # Provide valid formants + vowel
    update_artists(plotter, freqs, times, S, f1=500, f2=1500, vowel="i")

    # Scatter should be created
    assert "i" in plotter._vowel_scatters
    scatter = plotter._vowel_scatters["i"]
    assert scatter.get_offsets().shape == (1, 2)


def test_update_artists_updates_existing_mesh():
    plotter = FakePlotter()

    freqs = np.linspace(0, 5000, 256)
    times = np.linspace(0, 1.0, 100)
    S = np.abs(np.random.randn(256, 100)) ** 2

    # First call creates mesh
    update_artists(plotter, freqs, times, S, None, None, None)

    # Modify S to force update path
    S2 = np.abs(np.random.randn(256, 100)) ** 2
    update_artists(plotter, freqs, times, S2, None, None, None)

    # Mesh should still exist and be updated
    assert plotter._spec_mesh is not None
