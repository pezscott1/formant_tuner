# calibration/window.py
import traceback
import time

import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QFrame,
)
from PyQt5.QtCore import QTimer, pyqtSignal

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas

from calibration.session import CalibrationSession
from calibration.state_machine import CalibrationStateMachine
from calibration.plotter import update_artists, safe_spectrogram

from analysis.engine import FormantAnalysisEngine


class CalibrationWindow(QMainWindow):
    """
    Thin PyQt wrapper for the calibration workflow.

    All DSP is done by the shared FormantAnalysisEngine instance, which
    is already being fed by the main mic pipeline. This window only:

      - runs the prep/sing/capture state machine
      - polls engine.get_latest_raw()
      - accumulates formant captures
      - saves a profile
      - notifies the main tuner via profile_calibrated(signal)
    """

    profile_calibrated = pyqtSignal(str)

    def __init__(self, profile_name, voice_type="bass",
                 analyzer=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self._last_frame = None
        # -----------------------------------------------------
        # Core objects
        # -----------------------------------------------------
        self.vowels = ["i", "e", "a", "o", "u"]
        self.analyzer = analyzer
        self.session = CalibrationSession(
            profile_name=profile_name,
            voice_type=voice_type,
            vowels=self.vowels,
        )
        self.state = CalibrationStateMachine(self.vowels)
        # Shared engine: MUST be the same instance the tuner mic feeds
        self.engine = analyzer or FormantAnalysisEngine(voice_type=voice_type)
        # Capture behavior
        self.capture_timeout = 3.0  # seconds

        # -----------------------------------------------------
        # UI Setup
        # -----------------------------------------------------
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left panel: status + captures
        left = self._build_left_panel()
        layout.addWidget(left)

        # Right panel: spectrogram + vowel scatter
        right = self._build_right_panel()
        layout.addWidget(right, stretch=1)

        # -----------------------------------------------------
        # Timers
        # -----------------------------------------------------
        self.phase_timer = QTimer(self)
        self.phase_timer.timeout.connect(self._tick_phase)  # type:ignore
        self.phase_timer.start(1000)

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_audio)  # type:ignore
        self.poll_timer.start(80)

        self.show()

    # ---------------------------------------------------------
    # UI Construction
    # ---------------------------------------------------------
    def _build_left_panel(self):
        frame = QFrame()
        layout = QVBoxLayout(frame)

        self.status_panel = QPlainTextEdit()
        self.status_panel.setReadOnly(True)
        layout.addWidget(QLabel("Status"))
        layout.addWidget(self.status_panel)

        self.capture_panel = QPlainTextEdit()
        self.capture_panel.setReadOnly(True)
        layout.addWidget(QLabel("Captures"))
        layout.addWidget(self.capture_panel)

        return frame

    def _build_right_panel(self):
        frame = QFrame()
        layout = QVBoxLayout(frame)

        self.fig, (self.ax_spec, self.ax_vowel) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = Canvas(self.fig)
        layout.addWidget(self.canvas)

        # Plotter state
        self._spec_mesh = None
        self._vowel_scatters = {}
        self._vowel_colors = {
            "i": "red",
            "e": "green",
            "a": "blue",
            "o": "purple",
            "u": "orange",
        }
        self._last_draw = 0.0
        self._min_draw_interval = 0.05

        return frame

    # ---------------------------------------------------------
    # Phase machine tick
    # ---------------------------------------------------------
    def _tick_phase(self):
        event = self.state.tick()

        if event["event"] == "prep_countdown":
            self.status_panel.appendPlainText(
                f"Prepare: /{self.state.current_vowel}/ in {event['secs']}…"
            )

        elif event["event"] == "start_sing":
            self.status_panel.appendPlainText(
                f"Sing /{event['vowel']}/…"
            )

        elif event["event"] == "sing_countdown":
            self.status_panel.appendPlainText(
                f"Sing /{self.state.current_vowel}/ – {event['secs']}s"
            )

        elif event["event"] == "start_capture":
            self.status_panel.appendPlainText(
                f"Capturing /{self.state.current_vowel}/…"
            )

        elif event["event"] == "capture_ready":
            self._process_capture()

        elif event["event"] == "finished":
            self.status_panel.appendPlainText("Calibration complete!")
            self._finish()
            return

        # Timeout check during capture
        if self.state.check_timeout(self.capture_timeout):
            vowel = self.state.current_vowel
            self.status_panel.appendPlainText(
                f"/{vowel}/ capture timed out after {self.capture_timeout:.1f}s"
            )
            ev = self.state.advance()
            if ev["event"] == "finished":
                self.status_panel.appendPlainText("Calibration complete!")
                self._finish()

    # ---------------------------------------------------------
    # Poll shared engine
    # ---------------------------------------------------------
    def _poll_audio(self):
        """
        Poll the shared engine for the latest raw frame.
        """
        raw = self.engine.get_latest_raw()
        if raw is None:
            return

        # Extract formants and pitch
        f0 = raw.get("f0")
        f1, f2, f3 = raw.get("formants", (None, None, None))
        self._last_frame = (f1, f2, f0)  # store (f1, f2, f0) for session

        # Extract audio segment for spectrogram
        segment = raw.get("segment")
        if segment is None:
            return

        try:
            seg_arr = np.asarray(segment, dtype=float).flatten()
            if seg_arr.size == 0:
                return

            freqs, times, S = safe_spectrogram(seg_arr, getattr(self.engine, "sample_rate", 44100))
            update_artists(
                self,
                freqs,
                times,
                S,
                f1,
                f2,
                self.state.current_vowel,
            )
        except Exception:
            traceback.print_exc()

        # Throttle extra draws if update_artists didn't already draw
        now = time.time()
        if now - self._last_draw >= self._min_draw_interval:
            try:
                self.canvas.draw_idle()
            except Exception:
                pass
            self._last_draw = now

    # ---------------------------------------------------------
    # Capture processing
    # ---------------------------------------------------------
    def _process_capture(self):
        last_frame = getattr(self, "_last_frame", None)
        if last_frame is None:
            self.status_panel.appendPlainText("No audio captured")
            return

        if self._last_frame is None:
            self.status_panel.appendPlainText("No audio captured")
            return
        f0 = self._last_result.get("f0") if hasattr(self, "_last_result") else None
        f1, f2 = self._last_frame
        accepted, skipped, msg = self.session.handle_result(f1, f2, f0)
        self.status_panel.appendPlainText(msg)

        # ✅ Successful capture
        if accepted:
            idx = self.session.current_index - 1
            vowel = self.session.vowels[idx]
            self.capture_panel.appendPlainText(
                f"/{vowel}/ F1={f1:.1f}, F2={f2:.1f}, F0={f0 or 0:.1f}"
            )
            self.state.advance()  # advance UI
            return

        # ✅ Skip
        if skipped:
            self.state.advance()
            return

        # ✅ Retry (do NOT advance UI)
        # Just update the status panel; stay on same vowel
        return

    # ---------------------------------------------------------
    # Finish and save profile
    # ---------------------------------------------------------
    def _finish(self):
        """
        Save profile via CalibrationSession and notify parent.
        """
        try:
            base_name = self.session.save_profile()
            self.status_panel.appendPlainText(
                f"Profile saved for {base_name}"
            )
        except Exception:
            traceback.print_exc()
            self.status_panel.appendPlainText("Failed to save profile.")
            base_name = f"{self.session.profile_name}_{self.session.voice_type}"

        # Notify parent (tuner window)
        try:
            self.profile_calibrated.emit(base_name)  # type: ignore[arg-type]
        except Exception:
            traceback.print_exc()

        self.close()

    # ---------------------------------------------------------
    # Close handling
    # ---------------------------------------------------------
    def closeEvent(self, event):
        try:
            self.phase_timer.stop()
            self.poll_timer.stop()
        except Exception:
            traceback.print_exc()
        event.accept()
