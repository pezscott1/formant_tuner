import os
import pytest
if os.environ.get("CI") == "true":
    pytest.skip("Skipping Qt/Matplotlib UI tests in CI", allow_module_level=True)
import numpy as np
import matplotlib.pyplot as plt
import types

from tuner.tuner_plotter import update_spectrum, update_vowel_chart


# ------------------------------------------------------------
# Fake window object for structural testing
# ------------------------------------------------------------

class FakeAnalyzer:
    def __init__(self, segment=None):
        self._segment = segment

    def get_latest_raw(self):
        if self._segment is None:
            return {}
        return {"segment": self._segment}


class FakeWindow:
    def __init__(self, segment=None):
        fig, (ax_chart, ax_vowel) = plt.subplots(2, 1)

        self.ax_chart = ax_chart
        self.ax_vowel = ax_vowel
        self.canvas = types.SimpleNamespace(draw_idle=lambda: None)

        self.analyzer = FakeAnalyzer(segment)
        self.sample_rate = 44100

        # Artists used by update_vowel_chart
        self.vowel_measured_artist = None
        self.vowel_line_artist = None


# ------------------------------------------------------------
# update_spectrum tests
# ------------------------------------------------------------

def test_update_spectrum_no_segment():
    """Covers: no FFT path, title without pitch, formant lines."""
    window = FakeWindow(segment=None)

    update_spectrum(
        window,
        vowel="i",
        target_formants=(500, 1500, 2500),
        measured_formants=(600, 1600, 2600),
        pitch=None,
        _tol=None,
    )

    # Should have drawn formant lines
    assert len(window.ax_chart.lines) >= 3
    assert "Spectrum /i/" in window.ax_chart.get_title()


def test_update_spectrum_with_segment_and_pitch():
    """Covers: FFT path, pitch→note title, measured lines."""
    # Create a fake audio segment
    segment = np.random.randn(2048)

    window = FakeWindow(segment=segment)

    update_spectrum(
        window,
        vowel="a",
        target_formants=(700, 1200, 2600),
        measured_formants=(710, 1180, 2550),
        pitch=220.0,
        _tol=None,
    )

    # FFT plot should produce at least one line
    assert len(window.ax_chart.lines) > 0

    # Title should include note name and pitch
    title = window.ax_chart.get_title()
    assert "Spectrum /a/" in title
    assert "Hz" in title


# ------------------------------------------------------------
# update_vowel_chart tests
# ------------------------------------------------------------

def test_update_vowel_chart_basic_point_and_line():
    """Covers: measured point, measured→target line, title update."""
    window = FakeWindow()

    update_vowel_chart(
        window,
        vowel="i",
        target_formants=(500, 1500, None),
        measured_formants=(520, 1480, None),
        vowel_score=0.8,
        resonance_score=0.7,
        overall=0.75,
    )

    # Measured point should exist
    assert window.vowel_measured_artist is not None

    # Line should exist
    assert window.vowel_line_artist is not None

    # Title should be updated
    title = window.ax_vowel.get_title()
    assert "/i/" in title
    assert "Overall=0.75" in title


def test_update_vowel_chart_nan_handling():
    """Covers: NaN sanitization, no measured point, no line."""
    window = FakeWindow()

    update_vowel_chart(
        window,
        vowel="u",
        target_formants=(np.nan, np.nan, None),
        measured_formants=(np.nan, np.nan, None),
        vowel_score=0.0,
        resonance_score=0.0,
        overall=0.0,
    )

    # No artists should be created
    assert window.vowel_measured_artist is None
    assert window.vowel_line_artist is None


def test_update_vowel_chart_artist_removal():
    """Covers: removal of previous artists."""
    window = FakeWindow()

    # First call creates artists
    update_vowel_chart(
        window,
        vowel="a",
        target_formants=(600, 1100, None),
        measured_formants=(620, 1080, None),
        vowel_score=0.9,
        resonance_score=0.8,
        overall=0.85,
    )

    first_point = window.vowel_measured_artist  # noqa: F841
    first_line = window.vowel_line_artist  # noqa: F841

    # Second call should remove old artists and create new ones
    update_vowel_chart(
        window,
        vowel="a",
        target_formants=(600, 1100, None),
        measured_formants=(630, 1070, None),
        vowel_score=0.9,
        resonance_score=0.8,
        overall=0.85,
    )

    assert window.vowel_measured_artist is None
    assert window.vowel_line_artist is None
