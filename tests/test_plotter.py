import numpy as np
from unittest.mock import MagicMock
from calibration.plotter import update_artists
from calibration.plotter import safe_spectrogram


def test_update_artists_initial_mesh():
    win = MagicMock()
    win.state.phase = "capture"
    win._spec_mesh = None
    win._vowel_scatters = {}
    win._vowel_colors = {"a": "red"}
    win._last_draw = 0
    win._min_draw_interval = 0

    freqs = np.array([100, 200, 300])
    times = np.array([0.1, 0.2, 0.3])
    S = np.ones((3, 3))

    update_artists(win, freqs, times, S, 500, 1500, "a")

    win.ax_spec.clear.assert_called_once()
    win.ax_spec.pcolormesh.assert_called_once()
    win.canvas.draw_idle.assert_called()


def test_update_artists_updates_existing_mesh():
    win = MagicMock()
    win.state.phase = "capture"
    win._spec_mesh = MagicMock()
    win._spec_mesh.get_array.return_value.size = 9  # 3x3
    win._vowel_scatters = {"a": MagicMock()}
    win._vowel_colors = {"a": "red"}
    win._last_draw = 0
    win._min_draw_interval = 0

    freqs = np.array([100, 200, 300])
    times = np.array([0.1, 0.2, 0.3])
    S = np.ones((3, 3))

    update_artists(win, freqs, times, S, 500, 1500, "a")

    win._spec_mesh.set_array.assert_called_once()


def test_update_artists_recreates_mesh_on_size_change():
    win = MagicMock()
    win.state.phase = "capture"
    win._spec_mesh = MagicMock()
    win._spec_mesh.get_array.return_value.size = 4  # wrong size
    win._vowel_scatters = {"a": MagicMock()}
    win._vowel_colors = {"a": "red"}
    win._last_draw = 0
    win._min_draw_interval = 0

    freqs = np.array([100, 200, 300])
    times = np.array([0.1, 0.2, 0.3])
    S = np.ones((3, 3))

    update_artists(win, freqs, times, S, 500, 1500, "a")

    win.ax_spec.clear.assert_called_once()
    win.ax_spec.pcolormesh.assert_called_once()


def test_update_artists_creates_vowel_scatter():
    win = MagicMock()
    win.state.phase = "capture"
    win._spec_mesh = MagicMock()
    win._spec_mesh.get_array.return_value.size = 9
    win._vowel_scatters = {}
    win._vowel_colors = {"a": "red"}
    win._last_draw = 0
    win._min_draw_interval = 0

    freqs = np.array([100, 200, 300])
    times = np.array([0.1, 0.2, 0.3])
    S = np.ones((3, 3))

    update_artists(win, freqs, times, S, 500, 1500, "a")

    win.ax_vowel.scatter.assert_called_once()
    win.ax_vowel.legend.assert_called_once()


def test_update_artists_updates_vowel_scatter():
    scatter = MagicMock()

    win = MagicMock()
    win.state.phase = "capture"
    win._spec_mesh = MagicMock()
    win._spec_mesh.get_array.return_value.size = 9
    win._vowel_scatters = {"a": scatter}
    win._vowel_colors = {"a": "red"}
    win._last_draw = 0
    win._min_draw_interval = 0

    freqs = np.array([100, 200, 300])
    times = np.array([0.1, 0.2, 0.3])
    S = np.ones((3, 3))

    update_artists(win, freqs, times, S, 500, 1500, "a")

    scatter.set_offsets.assert_called_once()


def test_update_artists_missing_formants():
    win = MagicMock()
    win.state.phase = "capture"
    win._spec_mesh = MagicMock()
    win._spec_mesh.get_array.return_value.size = 9
    win._vowel_scatters = {}
    win._vowel_colors = {}
    win._last_draw = 0
    win._min_draw_interval = 0

    freqs = np.array([100, 200, 300])
    times = np.array([0.1, 0.2, 0.3])
    S = np.ones((3, 3))

    update_artists(win, freqs, times, S, None, None, None)

    win.canvas.draw_idle.assert_called()


def test_safe_spectrogram_empty():
    f, t, S = safe_spectrogram([], 44100)
    assert S.shape[0] == 128
    assert S.shape[1] == 1


def test_safe_spectrogram_fallback_fft(monkeypatch):
    def fail_stft(*_args, **_kwargs):
        raise ValueError("fail")

    monkeypatch.setattr("librosa.stft", fail_stft)

    y = np.random.randn(4096)
    f, t, S = safe_spectrogram(y, 44100)

    assert S.shape[0] > 0
    assert S.shape[1] > 0


def test_update_spectrum_no_analyzer(qtbot):
    class Dummy:
        ax_chart = MagicMock()
        canvas = MagicMock()
        sample_rate = 16000
        analyzer = None

    from tuner.tuner_plotter import update_spectrum
    win = Dummy()

    update_spectrum(win, "i", (300, 2500, None), (320, 2400, None), 140, None)


def test_update_spectrum_analyzer_raises(qtbot):
    class BadAnalyzer:
        def get_latest_raw(self):
            raise RuntimeError("boom")

    class Dummy:
        ax_chart = MagicMock()
        canvas = MagicMock()
        sample_rate = 16000
        analyzer = BadAnalyzer()

    from tuner.tuner_plotter import update_spectrum
    win = Dummy()

    update_spectrum(win, "i", (300, 2500, None), (320, 2400, None), 140, None)


def test_update_vowel_chart_nan_safe(qtbot):
    class Dummy:
        ax_vowel = MagicMock()
        canvas = MagicMock()
        vowel_measured_artist = None
        vowel_line_artist = None

    from tuner.tuner_plotter import update_vowel_chart
    win = Dummy()

    update_vowel_chart(
        win,
        "i",
        (None, None, None),
        (np.nan, np.nan, None),
        vowel_score=0.5,
        resonance_score=0.5,
        overall=0.5,
    )


def test_update_vowel_chart_skips_nan(qtbot):
    class Dummy:
        ax_vowel = type("A", (), {"scatter": lambda *a, **k: None,
                                  "plot": lambda *a, **k: [None],
                                  "set_title": lambda *a, **k: None})()
        canvas = type("C", (), {"draw_idle": lambda *a, **k: None})()
        vowel_measured_artist = None
        vowel_line_artist = None

    from tuner.tuner_plotter import update_vowel_chart

    win = Dummy()
    update_vowel_chart(win, "i",
                       target_formants=(None, None, None),
                       measured_formants=(float("nan"), float("nan"), None),
                       vowel_score=0.1,
                       resonance_score=0.2,
                       overall=0.3)
