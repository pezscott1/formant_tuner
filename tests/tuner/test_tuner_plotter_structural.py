import os
import pytest
if os.environ.get("CI") == "true":
    pytest.skip("Skipping Qt/Matplotlib UI tests in CI", allow_module_level=True)

import numpy as np
import matplotlib.pyplot as plt
import types
from tuner.tuner_plotter import update_spectrum, update_vowel_chart


# ------------------------------------------------------------
# Fake analyzer
# ------------------------------------------------------------

class FakeAnalyzer:
    def __init__(self, segment=None, confidence=1.0, method="hybrid", stable=True):
        self._segment = segment
        self._confidence = confidence
        self._method = method
        self.formant_smoother = types.SimpleNamespace(formants_stable=stable)

    def get_latest_raw(self):
        raw = {}
        if self._segment is not None:
            raw["segment"] = self._segment
        raw["confidence"] = self._confidence
        raw["method"] = self._method
        return raw


# ------------------------------------------------------------
# Fake text overlay object
# ------------------------------------------------------------

class FakeText:
    def __init__(self, ax=None):
        self._text = ""
        self.axes = ax

    def set_text(self, txt):
        self._text = txt

    def get_text(self):
        return self._text


# ------------------------------------------------------------
# Fake window
# ------------------------------------------------------------

class FakeWindow:
    def __init__(self, segment=None, confidence=1.0, method="hybrid", stable=True):
        fig, (ax_chart, ax_vowel) = plt.subplots(2, 1)

        self.ax_chart = ax_chart
        self.ax_vowel = ax_vowel
        self.canvas = types.SimpleNamespace(draw_idle=lambda: None)

        self.analyzer = FakeAnalyzer(
            segment=segment,
            confidence=confidence,
            method=method,
            stable=stable,
        )

        self.sample_rate = 44100

        # Artists used by update_vowel_chart
        self.vowel_measured_artist = None
        self.vowel_line_artist = None

        self.spec_status_text = FakeText(self.ax_chart)
        self.vowel_status_text = FakeText(self.ax_vowel)


# ------------------------------------------------------------
# update_spectrum tests
# ------------------------------------------------------------

def test_update_spectrum_no_segment():
    """No FFT path, but target formant lines + overlay text appear."""
    window = FakeWindow(segment=None)

    update_spectrum(
        window,
        vowel="i",
        target_formants={"f1": 500, "f2": 1500, "f3": 2500},
        measured_formants={"f1": 600, "f2": 1600, "f3": 2600},
        pitch=None,
        _tolerance=None,
    )

    # Should have drawn at least the 3 target formant lines
    assert len(window.ax_chart.lines) >= 3

    # Overlay text now contains vowel info
    assert "/i/" in window.spec_status_text.get_text()


def test_update_spectrum_with_segment_and_pitch():
    """FFT path + pitch overlay + measured lines."""
    segment = np.random.randn(2048)
    window = FakeWindow(segment=segment, confidence=1.0)

    update_spectrum(
        window,
        vowel="a",
        target_formants={"f1": 700, "f2": 1200, "f3": 2600},
        measured_formants={"f1": 710, "f2": 1180, "f3": 2550},
        pitch=220.0,
        _tolerance=None,
    )

    # FFT plotted
    assert len(window.ax_chart.lines) > 0

    # Overlay text includes pitch
    txt = window.spec_status_text.get_text()
    assert "/a/" in txt
    assert "220.0 Hz" in txt


# ------------------------------------------------------------
# update_vowel_chart tests
# ------------------------------------------------------------

def test_update_vowel_chart_basic_point_and_line():
    """Measured point + line + overlay text."""
    window = FakeWindow(confidence=1.0, stable=True)

    update_vowel_chart(
        window,
        vowel="i",
        target_formants={"f1": 500, "f2": 1500},
        measured_formants={"f1": 520, "f2": 1480},
        vowel_score=0.8,
        resonance_score=0.7,
        overall=0.75,
    )

    assert window.vowel_measured_artist is not None

    txt = window.vowel_status_text.get_text()
    assert "/i/" in txt
    assert "Overall=0.75" in txt


def test_update_vowel_chart_nan_handling():
    """Invalid formants → no artists."""
    window = FakeWindow(confidence=1.0, stable=True)

    update_vowel_chart(
        window,
        vowel="u",
        target_formants={"f1": np.nan, "f2": np.nan},
        measured_formants={"f1": np.nan, "f2": np.nan},
        vowel_score=0.0,
        resonance_score=0.0,
        overall=0.0,
    )

    assert window.vowel_measured_artist is None
    assert window.vowel_line_artist is None


def test_update_vowel_chart_artist_removal():
    """Second call removes previous artists and does NOT store new ones."""
    window = FakeWindow(confidence=1.0, stable=True)

    # First call → stores artists
    update_vowel_chart(
        window,
        vowel="a",
        target_formants={"f1": 600, "f2": 1100},
        measured_formants={"f1": 620, "f2": 1080},
        vowel_score=0.9,
        resonance_score=0.8,
        overall=0.85,
    )
    assert window.vowel_measured_artist is not None
    first_point = window.vowel_measured_artist
    first_line = window.vowel_line_artist

    # Second call → removes old artists, does NOT store new ones
    update_vowel_chart(
        window,
        vowel="a",
        target_formants={"f1": 600, "f2": 1100},
        measured_formants={"f1": 630, "f2": 1070},
        vowel_score=0.9,
        resonance_score=0.8,
        overall=0.85,
    )

    assert getattr(first_point, "removed", True)
    if first_line is not None:
        assert first_line.removed
    assert window.vowel_measured_artist is None
    assert window.vowel_line_artist is None
