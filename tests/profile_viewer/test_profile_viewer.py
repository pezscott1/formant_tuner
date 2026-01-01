from unittest.mock import patch
from profile_viewer.profile_viewer import ProfileViewerWindow
import pytest
from PyQt5.QtWidgets import QApplication
import sys


@pytest.fixture(scope="session")
def app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture
def sample_profile():
    return {
        "i": {"f1": 300, "f2": 2200, "f0": 200},
        "ɛ": {"f1": 600, "f2": 1800, "f0": 150},
        "ɑ": {"f1": 700, "f2": 1100, "f0": 140},
        "ɔ": {"f1": 500, "f2": 900, "f0": 130},
        "u": {"f1": 350, "f2": 800, "f0": 120},
        "e": {"f1": 500, "f2": 2000, "f0": 180},   # interpolated
        "ɪ": {"f1": 400, "f2": 2100, "f0": 190},   # interpolated
    }


def test_window_initializes(app, sample_profile):
    win = ProfileViewerWindow(sample_profile, headless=True)
    assert win.profile == sample_profile
    assert win.calibrated_vowels == {"i", "ɛ", "ɑ", "ɔ", "u"}
    assert "e" in win.interpolated_vowels
    assert "ɪ" in win.interpolated_vowels


def test_values_panel_contains_expected_text(app, sample_profile):
    win = ProfileViewerWindow(sample_profile, headless=True)
    text = win.values_panel.toHtml()

    # Check vowel labels appear
    assert "/i/" in text
    assert "/e/" in text

    # Check formant numbers appear
    assert "F1=300.0" in text
    assert "F2=2200.0" in text


def test_values_panel_uses_color_coding(app, sample_profile):
    win = ProfileViewerWindow(sample_profile, headless=True)
    html = win.values_panel.toHtml()

    # Check that HTML color styling is present
    assert "color:" in html
    assert "font-weight:bold" in html or "font-weight:600" in html


def test_legend_present(app, sample_profile):
    win = ProfileViewerWindow(sample_profile, headless=True)
    legend_html = win.legend_panel.toHtml()

    assert "Calibrated vowels" in legend_html
    assert "Interpolated vowels" in legend_html


def test_plot_profile_runs_without_error(app, sample_profile):
    win = ProfileViewerWindow(sample_profile, headless=True)

    # Mock scatter so we don't depend on matplotlib internals
    with patch.object(win.ax, "scatter", return_value=None) as mock_scatter:
        win._plot_profile()
        # Should be called once per vowel
        assert mock_scatter.call_count == len(sample_profile)


def test_window_resizes_to_expected_dimensions(app, sample_profile):
    win = ProfileViewerWindow(sample_profile, headless=True)
    win.resize(800, 600)
    app.processEvents()

    assert win.width() == 800
    assert win.height() == 600

    win.close()   # <-- critical
