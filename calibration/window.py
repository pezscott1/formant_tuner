# calibration/window.py
import time
import traceback
from analysis.smoothing import LabelSmoother, PitchSmoother, MedianSmoother
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
    profile_calibrated = pyqtSignal(str)

    def __init__(self, profile_name, voice_type="bass",
                 analyzer=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self._last_frame = None
        self._last_result = None
        self._capture_buffer = []
        self._spec_buffer = np.array([], dtype=float)
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

        # Smoothers
        self.label_smoother = LabelSmoother(hold_frames=4)
        self.pitch_smoother = PitchSmoother(alpha=0.25)
        self.formant_smoother = MedianSmoother(window=5)
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

        # -----------------------------------------------------
        # Match size and position to parent (TunerWindow)
        # -----------------------------------------------------
        if parent is not None:
            # Match size exactly
            self.resize(parent.size())

            # Center calibration window over the parent
            geo = parent.geometry()
            self.move(
                geo.center().x() - self.width() // 2,
                geo.center().y() - self.height() // 2,
            )
        else:
            # Fallback default size if launched standalone
            self.resize(900, 700)

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
        # Prevent axis label overlap
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.12)
        self.canvas = Canvas(self.fig)
        layout.addWidget(self.canvas)

        # Plotter state
        self.ax_spec.set_ylabel("Frequency (Hz)")
        self.ax_spec.set_xlabel("Time (s)")
        self.ax_vowel.set_xlabel("F2 (Hz)")
        self.ax_vowel.set_ylabel("F1 (Hz)")
        self.ax_vowel.invert_xaxis()
        self.ax_vowel.invert_yaxis()
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
            return

        elif event["event"] == "finished":
            self.status_panel.appendPlainText("Calibration complete!")
            self._finish()
            return

        # Timeout check during capture
        if (event["event"] == "capture_tick" and
                self.state.check_timeout(self.capture_timeout)):
            vowel = self.state.current_vowel
            self.status_panel.appendPlainText(
                f"/{vowel}/ capture timed out after {self.capture_timeout:.1f}s"
            )
            ev = self.state.advance()
            if ev["event"] == "finished":
                self.status_panel.appendPlainText("Calibration complete!")
                self._finish()
                return

    # ---------------------------------------------------------
    # Poll shared engine
    # ---------------------------------------------------------
    def _poll_audio(self):
        """
        Poll the shared engine for the latest raw frame.
        Build a rolling audio buffer for spectrogram display.
        """

        # ---------------------------------------------------------
        # 1. ALWAYS get raw and update _last_frame FIRST
        # ---------------------------------------------------------
        raw = self.engine.get_latest_raw()
        if raw is None:
            return

        self._last_result = raw
        f0 = raw.get("f0")
        f1, f2, f3 = raw.get("formants", (None, None, None))
        self._last_frame = (f1, f2, f0)
        # ---------------------------------------------------------
        # CAPTURE BUFFER: collect frames during capture phase
        # ---------------------------------------------------------
        if self.state.phase == "capture":
            # Basic sanity check
            if f1 is not None and f2 is not None and f0 is not None:
                # Optional pitch plausibility (you can relax this)
                if 40 <= f0 <= 800:
                    self._capture_buffer.append((float(f1), float(f2), float(f0)))
                    if self.state.phase == "capture":
                        print(f"APPEND: vowel="
                              f"{self.state.current_vowel} f1={f1} f2={f2} f0={f0}")
        # ---------------------------------------------------------
        # 2. Extract segment (but DO NOT return early if missing)
        # ---------------------------------------------------------
        segment = raw.get("segment")
        if segment is None:
            return  # safe: _last_frame already updated

        # ---------------------------------------------------------
        # 3. Spectrogram: fully isolated so it cannot break capture
        # ---------------------------------------------------------
        try:
            seg_arr = np.asarray(segment, dtype=float).flatten()
            if seg_arr.size == 0:
                return

            # Rolling buffer
            if not hasattr(self, "_spec_buffer"):
                self._spec_buffer = np.array([], dtype=float)

            self._spec_buffer = np.concatenate((self._spec_buffer, seg_arr))

            sr = getattr(self.engine, "sample_rate", 44100)
            max_seconds = 0.75
            max_samples = int(sr * max_seconds)

            # Trim buffer
            if self._spec_buffer.size > max_samples:
                self._spec_buffer = self._spec_buffer[-max_samples:]

            # Only compute spectrogram if buffer is long enough
            if self._spec_buffer.size > 2048:
                freqs, times, S = safe_spectrogram(self._spec_buffer, sr)

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
            # DO NOT return — calibration must continue

        # ---------------------------------------------------------
        # 4. Throttled draw
        # ---------------------------------------------------------
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
        # ---------------------------------------------------------
        # 1. Must have frames
        # ---------------------------------------------------------
        if not self._capture_buffer:
            self.status_panel.appendPlainText("No audio captured")
            return

        vowel = self.state.current_vowel

        # Unpack buffer
        f1_vals, f2_vals, f0_vals = zip(*self._capture_buffer)

        f1_arr = np.asarray(f1_vals, dtype=float)
        f2_arr = np.asarray(f2_vals, dtype=float)
        f0_arr = np.asarray(f0_vals, dtype=float)

        # Remove NaN/inf
        f1_arr = f1_arr[np.isfinite(f1_arr)]
        f2_arr = f2_arr[np.isfinite(f2_arr)]
        f0_arr = f0_arr[np.isfinite(f0_arr)]

        # ---------------------------------------------------------
        # 2. Retry if truly empty
        # ---------------------------------------------------------
        if f1_arr.size == 0 or f2_arr.size == 0:
            self.status_panel.appendPlainText("Retrying: no usable frames")
            self._capture_buffer.clear()
            self.state.retry_current_vowel()
            return

        # ---------------------------------------------------------
        # 3. Low-confidence handling (avoid infinite retries)
        # ---------------------------------------------------------
        # For most vowels, 2 usable frames is enough.
        # For /o/ and /u/, formants are often sparse — allow 1.
        min_frames = 2
        if vowel in ("o", "u"):
            min_frames = 1

        if f1_arr.size < min_frames or f2_arr.size < min_frames:
            self.status_panel.appendPlainText(
                f"Low confidence for /{vowel}/ — using available frames"
            )

        # ---------------------------------------------------------
        # 4. Compute medians
        # ---------------------------------------------------------
        f1_med = float(np.median(f1_arr))
        f2_med = float(np.median(f2_arr))
        f0_med = float(np.median(f0_arr)) if f0_arr.size > 0 else None

        # ---------------------------------------------------------
        # 5. Vowel-specific plausibility fixes
        # ---------------------------------------------------------

        # /o/ is the hardest vowel — clamp to realistic region
        if vowel == "o":
            # F1 should be ~300–700 Hz
            if not (200 <= f1_med <= 800):
                good = f1_arr[(f1_arr >= 200) & (f1_arr <= 800)]
                if good.size > 0:
                    f1_med = float(np.median(good))

            # F2 should be ~700–1500 Hz
            if not (400 <= f2_med <= 1800):
                good = f2_arr[(f2_arr >= 400) & (f2_arr <= 1800)]
                if good.size > 0:
                    f2_med = float(np.median(good))

        # /u/ also has weak F2 — allow fallback
        if vowel == "u":
            if f2_med > 2000:  # clearly wrong
                good = f2_arr[(f2_arr >= 300) & (f2_arr <= 1500)]
                if good.size > 0:
                    f2_med = float(np.median(good))

        # ---------------------------------------------------------
        # 6. Submit to session
        # ---------------------------------------------------------
        accepted, skipped, msg = self.session.handle_result(f1_med, f2_med, f0_med)
        self.status_panel.appendPlainText(msg)

        # Clear buffer for next vowel
        self._capture_buffer.clear()

        # ---------------------------------------------------------
        # 7. Handle outcomes
        # ---------------------------------------------------------
        if accepted:
            idx = self.session.current_index - 1
            vowel = self.session.vowels[idx]
            self.capture_panel.appendPlainText(
                f"/{vowel}/ F1={f1_med:.1f}, F2={f2_med:.1f}, F0={(f0_med or 0):.1f}"
            )
            self.state.advance()
            return

        if skipped:
            self.state.advance()
            return

        # Retry case from handle_result — stay on same vowel
        # (state machine will re-enter prep/sing/capture)
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
