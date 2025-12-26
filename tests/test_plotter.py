import numpy as np
from unittest.mock import MagicMock
from tuner.tuner_plotter import update_vowel_chart


# ---------------------------------------------------------
# Helper window with all required attributes
# ---------------------------------------------------------
class FakeWindow:
    def __init__(self):
        # Axes + canvas
        self.ax_vowel = MagicMock()
        self.canvas = MagicMock()

        # Artists
        self.vowel_measured_artist = None
        self.vowel_line_artist = None

        # Analyzer + smoother (MISSING BEFORE)
        self.analyzer = MagicMock()
        self.analyzer.formant_smoother = MagicMock()
        self.analyzer.formant_smoother.formants_stable = True
        self.analyzer.formant_smoother._stability_score = 0.0
        # Latest raw (confidence + method)
        self.latest_raw = {"confidence": 1.0, "method": "lpc"}

        # Internal state
        self._last_draw = 0
        self._min_draw_interval = 0
        self._vowel_scatters = {}
        self._vowel_colors = {"i": "blue", "a": "red"}
# ---------------------------------------------------------
# NAN-safe behavior
# ---------------------------------------------------------


def test_update_vowel_chart_nan_safe():
    win = FakeWindow()

    update_vowel_chart(
        win,
        "i",
        (None, None, None),
        (np.nan, np.nan, None),
        vowel_score=0.5,
        resonance_score=0.5,
        overall=0.5,
    )

    # No scatter should be created because measured formants are NaN
    assert win.vowel_measured_artist is None
    assert win.vowel_line_artist is None


# ---------------------------------------------------------
# Scatter creation when stable + confident
# ---------------------------------------------------------
def test_update_vowel_chart_creates_scatter():
    win = FakeWindow()

    update_vowel_chart(
        win,
        "i",
        (500, 1500, None),
        (520, 1480, None),
        vowel_score=0.8,
        resonance_score=0.7,
        overall=0.75,
    )

    assert win.vowel_measured_artist is not None
    assert win.vowel_line_artist is not None


# ---------------------------------------------------------
# Scatter suppressed when unstable
# ---------------------------------------------------------
def test_update_vowel_chart_suppresses_when_unstable():
    win = FakeWindow()
    win.analyzer.formant_smoother.formants_stable = False

    update_vowel_chart(
        win,
        "i",
        (500, 1500, None),
        (520, 1480, None),
        vowel_score=0.8,
        resonance_score=0.7,
        overall=0.75,
    )

    # No artists should be created when unstable
    assert win.vowel_measured_artist is None
    assert win.vowel_line_artist is None

# ---------------------------------------------------------
# Scatter suppressed when low confidence
# ---------------------------------------------------------


def test_update_vowel_chart_suppresses_when_low_confidence():
    win = FakeWindow()
    win.latest_raw["confidence"] = 0.1  # below threshold

    update_vowel_chart(
        win,
        "i",
        (500, 1500, None),
        (520, 1480, None),
        vowel_score=0.8,
        resonance_score=0.7,
        overall=0.75,
    )

    assert win.vowel_measured_artist is None
    assert win.vowel_line_artist is None
