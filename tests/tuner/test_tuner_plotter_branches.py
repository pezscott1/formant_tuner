import os
import pytest

if os.environ.get("CI") == "true":
    pytest.skip("Skipping Qt/Matplotlib UI tests in CI", allow_module_level=True)

import numpy as np
from tuner.tuner_plotter import update_spectrum, update_vowel_chart


# ----------------------------------------------------------------------
# Dummy window + axes
# ----------------------------------------------------------------------

class DummyText:
    def __init__(self, ax=None):
        self._text = ""
        self.axes = ax

    def set_text(self, txt):
        self._text = txt

    def get_text(self):
        return self._text


class DummyAxis:
    def __init__(self):
        self.cleared = False
        self.lines = []
        self.title = ""

    def clear(self):
        self.cleared = True

    def plot(self, x, y, **kwargs):
        self.lines.append(("plot", x, y, kwargs))
        return [DummyArtist()]

    def axvline(self, x, **kwargs):
        self.lines.append(("vline", x, kwargs))

    def scatter(self, x, y, **kwargs):
        artist = DummyArtist()
        self.lines.append(("scatter", x, y, kwargs))
        return artist

    def set_title(self, title):
        self.title = title

    def set_xlabel(self, x):
        pass

    def set_ylabel(self, y):
        pass


class DummyArtist:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class DummyCanvas:
    def __init__(self):
        self.drawn = False

    def draw_idle(self):
        self.drawn = True


class DummyAnalyzer:
    def __init__(self, raw, smoother=None):
        self._raw = raw
        self.formant_smoother = smoother

    def get_latest_raw(self):
        return self._raw


class DummyWindow:
    def __init__(self):
        self.ax_chart = DummyAxis()
        self.ax_vowel = DummyAxis()
        self.canvas = DummyCanvas()
        self.sample_rate = 48000

        # Analyzer is injected per test
        self.analyzer = None

        # Artists
        self.vowel_measured_artist = None
        self.vowel_line_artist = None

        self.spec_status_text = DummyText(self.ax_chart)
        self.vowel_status_text = DummyText(self.ax_vowel)


# ----------------------------------------------------------------------
# update_spectrum tests
# ----------------------------------------------------------------------

def test_spectrum_no_analyzer_no_segment():
    w = DummyWindow()
    target = {"f1": 500, "f2": 1500, "f3": 2500}
    measured = {"f1": 600, "f2": 1600, "f3": 2600}

    update_spectrum(w, "a", target, measured, pitch=None, _tolerance=50)

    assert w.ax_chart.cleared
    assert "/a/" in w.spec_status_text.get_text()
    assert not any(t[0] == "plot" for t in w.ax_chart.lines)


def test_spectrum_with_segment_and_confidence():
    seg = np.sin(2 * np.pi * 200 * np.linspace(0, 0.01, 480))
    raw = {"segment": seg, "confidence": 0.9, "method": "lpc"}
    w = DummyWindow()
    w.analyzer = DummyAnalyzer(raw)

    target = {"f1": 500, "f2": 1500, "f3": 2500}
    measured = {"f1": 600, "f2": 1600, "f3": 2600}

    update_spectrum(w, "a", target, measured, pitch=200, _tolerance=50)

    assert any(t[0] == "plot" for t in w.ax_chart.lines)
    assert any(t[0] == "vline" for t in w.ax_chart.lines)

    txt = w.spec_status_text.get_text()
    assert "200.0 Hz" in txt
    assert "conf=" in txt


def test_spectrum_low_confidence_suppresses_measured():
    seg = np.ones(1024)
    raw = {"segment": seg, "confidence": 0.1, "method": "lpc"}
    w = DummyWindow()
    w.analyzer = DummyAnalyzer(raw)

    target = {"f1": 500, "f2": 1500, "f3": 2500}
    measured = {"f1": 600, "f2": 1600, "f3": 2600}

    update_spectrum(w, "a", target, measured, pitch=200, _tolerance=50)

    vlines = [t for t in w.ax_chart.lines if t[0] == "vline"]
    assert len(vlines) >= 3


# ----------------------------------------------------------------------
# update_vowel_chart tests
# ----------------------------------------------------------------------

def test_vowel_chart_suppressed_invalid_formants():
    w = DummyWindow()
    raw = {"confidence": 1.0}
    w.analyzer = DummyAnalyzer(raw=raw)

    target = {"f1": 500, "f2": 1500, "f3": 2500}
    measured = {"f1": None, "f2": 1500, "f3": 2500}

    update_vowel_chart(
        w, "a", target, measured,
        vowel_score=0.5, resonance_score=0.4, overall=0.45
    )

    assert w.vowel_measured_artist is not None
    assert w.vowel_line_artist is None
    assert "/a/" in w.vowel_status_text.get_text()


def test_vowel_chart_suppressed_low_confidence():
    w = DummyWindow()
    raw = {"confidence": 0.1}
    w.analyzer = DummyAnalyzer(raw=raw)

    target = {"f1": 500, "f2": 1500, "f3": 2500}
    measured = {"f1": 500, "f2": 1500, "f3": 2500}

    update_vowel_chart(
        w, "a", target, measured,
        vowel_score=0.5, resonance_score=0.4, overall=0.45
    )

    assert w.vowel_measured_artist is None
    assert w.vowel_line_artist is None
    assert "/a/" in w.vowel_status_text.get_text()


def test_vowel_chart_first_success_creates_artists():
    w = DummyWindow()
    raw = {"confidence": 1.0}
    smoother = type("X", (), {"formants_stable": True})()
    w.analyzer = DummyAnalyzer(raw=raw, smoother=smoother)

    target = {"f1": 500, "f2": 1500, "f3": 2500}
    measured = {"f1": 500, "f2": 1500, "f3": 2500}

    update_vowel_chart(
        w, "a", target, measured,
        vowel_score=0.5, resonance_score=0.4, overall=0.45
    )

    assert w.vowel_measured_artist is not None


def test_vowel_chart_second_success_removes_and_does_not_store():
    w = DummyWindow()
    raw = {"confidence": 1.0}
    smoother = type("X", (), {"formants_stable": True})()
    w.analyzer = DummyAnalyzer(raw=raw, smoother=smoother)

    target = {"f1": 500, "f2": 1500, "f3": 2500}
    measured = {"f1": 500, "f2": 1500, "f3": 2500}

    update_vowel_chart(
        w, "a", target, measured,
        vowel_score=0.5, resonance_score=0.4, overall=0.45
    )

    first_point = w.vowel_measured_artist

    update_vowel_chart(
        w, "a", target, measured,
        vowel_score=0.5, resonance_score=0.4, overall=0.45
    )

    assert first_point.removed
    assert w.vowel_measured_artist is None
    assert w.vowel_line_artist is None
