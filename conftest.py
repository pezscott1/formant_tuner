# tests/conftest.py
import os
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock
import matplotlib
import gc
os.environ.setdefault("COVERAGE_DISABLE_C_EXTENSION", "1")
# Prevent Qt from creating real windows (critical on Windows)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_configure():
    matplotlib.use('QtAgg', force=True)


@pytest.fixture(scope="session", autouse=True)
def disable_gc_for_pytest():
    gc.disable()
    yield
    gc.enable()


def pytest_sessionstart(session):
    # Force a pure headless backend (belt-and-suspenders with MPLBACKEND=Agg)
    matplotlib.use("Agg")

    # Disable cyclic GC to avoid Windows 3.12 + coverage access violations
    gc.disable()


def pytest_sessionfinish(session, exitstatus):
    # Re-enable GC after tests; not strictly needed, but explicit.
    gc.enable()

# ---------------------------------------------------------
# Global QApplication fixture (session-scoped)
# ---------------------------------------------------------


@pytest.fixture(scope="session")
def qapp():
    """Provide a single QApplication for all Qt tests."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ---------------------------------------------------------
# Optional: allow tests to skip Qt entirely
# ---------------------------------------------------------
def pytest_runtest_call(item):
    if item.get_closest_marker("noqt"):
        return item.runtest()


# ---------------------------------------------------------
# FakeWindow for tuner_plotter tests
# ---------------------------------------------------------
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
