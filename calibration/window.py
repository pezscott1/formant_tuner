# calibration/window.py
import time
import traceback
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

from tuner.live_analyzer import LiveAnalyzer
from analysis.smoothing import LabelSmoother, PitchSmoother, MedianSmoother
from analysis.plausibility import is_plausible_formants
from calibration.session import CalibrationSession
from calibration.state_machine import CalibrationStateMachine
from calibration.plotter import safe_spectrogram, update_spectrogram


class CalibrationWindow(QMainWindow):
    profile_calibrated = pyqtSignal(str)
    vowel_capture_started = pyqtSignal()
    vowel_capture_finished = pyqtSignal()

    def __init__(
        self,
        profile_name,
        voice_type="bass",
        engine=None,
        analyzer=None,
        profile_manager=None,
        existing_profile=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Calibration")

        # Shared engine
        self.engine = engine
        self.engine.use_hybrid = True
        self.engine.calibrating = True
        self.profile_manager = profile_manager

        # Rolling state
        self._capture_buffer = []
        self._spec_buffer = np.array([], dtype=float)

        # Mic/render state
        self._mic_active = False

        # Core vowel set
        self.vowels = ["i", "ɛ", "ɑ", "ɔ", "u"]
        # f0 locking state
        self._f0_locked = None
        self._f0_candidates = []

        # Calibration session
        self.session = CalibrationSession(
            profile_name=profile_name,
            voice_type=voice_type,
            vowels=self.vowels,
            profile_manager=self.profile_manager,
            existing_profile=existing_profile,
        )
        self.state = CalibrationStateMachine(self.vowels)

        # Smoothers
        self.pitch_smoother = PitchSmoother(sr=48000, min_confidence=0.25)
        self.formant_smoother = MedianSmoother(min_confidence=0.25)
        self.label_smoother = LabelSmoother(min_confidence=0.25)

        # Analyzer (shared with tuner)
        self.analyzer = analyzer if analyzer is not None else LiveAnalyzer(
            engine=self.engine,
            pitch_smoother=self.pitch_smoother,
            formant_smoother=self.formant_smoother,
            label_smoother=self.label_smoother,
        )

        self.capture_timeout = 3.0

        # Store durable vowel anchors
        self._vowel_anchors = {}
        self._vowel_colors = {
            "i": "red",
            "ɛ": "green",
            "ɑ": "blue",
            "ɔ": "purple",
            "u": "orange",
        }

        # -----------------------------------------------------
        # UI Setup
        # -----------------------------------------------------
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        left = self._build_left_panel()
        layout.addWidget(left)

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

        # Window sizing
        if parent is not None:
            self.resize(parent.size())
            geo = parent.geometry()
            self.move(
                geo.center().x() - self.width() // 2,
                geo.center().y() - self.height() // 2,
            )
        else:
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
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.12)

        self.canvas = Canvas(self.fig)
        layout.addWidget(self.canvas)

        # Spectrogram axes
        self.ax_spec.set_ylabel("Frequency (Hz)")
        self.ax_spec.set_xlabel("Time (s)")

        # Vowel axes
        self.ax_vowel.set_xlabel("F2 (Hz)")
        self.ax_vowel.set_ylabel("F1 (Hz)")
        self.ax_vowel.invert_xaxis()
        self.ax_vowel.invert_yaxis()

        self._last_draw = 0.0
        self._min_draw_interval = 0.05

        return frame

    # ---------------------------------------------------------
    # Phase machine tick
    # ---------------------------------------------------------
    def _reset_f0_lock(self):
        self._f0_locked = None
        self._f0_candidates = []

    def _tick_phase(self):
        event = self.state.tick()
        evt = event["event"]

        if evt == "prep_countdown":
            self._mic_active = False
            self.analyzer.pause()
            self.status_panel.appendPlainText(
                f"Prepare: /{self.state.current_vowel}/ in {event['secs']}…"
            )

        elif evt == "start_sing":
            self._mic_active = True
            self.analyzer.resume()
            self.vowel_capture_started.emit()  # type:ignore
            self.status_panel.appendPlainText(f"Sing /{event['vowel']}/…")

        elif evt == "sing_countdown":
            self.status_panel.appendPlainText(
                f"Sing /{self.state.current_vowel}/ – {event['secs']}s"
            )

        elif evt == "start_capture":
            self._mic_active = True
            self.analyzer.resume()
            self.status_panel.appendPlainText(
                f"Capturing /{self.state.current_vowel}/…"
            )

        elif evt == "retry":
            self._reset_f0_lock()
            self._mic_active = False
            self.analyzer.pause()
            self.vowel_capture_finished.emit()  # type:ignore
            self.status_panel.appendPlainText(
                f"Retrying /{self.state.current_vowel}/…"
            )

        elif evt == "next_vowel":
            self._reset_f0_lock()
            self._mic_active = False
            self.analyzer.pause()
            self.vowel_capture_finished.emit()  # type:ignore
            self.status_panel.appendPlainText(
                f"Next vowel: /{event['vowel']}/"
            )

        elif evt == "capture_ready":
            self._mic_active = False
            self.analyzer.pause()
            self.vowel_capture_finished.emit()  # type:ignore
            self._process_capture()
            return

        elif evt == "finished":
            self._mic_active = False
            self.analyzer.pause()
            self.vowel_capture_finished.emit()  # type:ignore
            self.status_panel.appendPlainText("Calibration complete!")
            self._finish()
            return

        # Timeout check
        if evt == "capture_tick" and self.state.check_timeout(self.capture_timeout):
            vowel = self.state.current_vowel
            self.status_panel.appendPlainText(
                f"/{vowel}/ capture timed out after {self.capture_timeout:.1f}s"
            )
            self._mic_active = False
            self.analyzer.pause()
            self.vowel_capture_finished.emit()  # type:ignore

            ev = self.state.advance()
            if ev["event"] == "finished":
                self.status_panel.appendPlainText("Calibration complete!")
                self._finish()
                return

    # ---------------------------------------------------------
    # Poll shared engine
    # ---------------------------------------------------------
    def _poll_audio(self):
        raw = self.analyzer.get_latest_raw()
        if raw is None:
            return

        confidence = raw.get("confidence", 0.0)
        target = self.state.current_vowel
        self.engine.vowel_hint = target

        # Pitch
        pitch_raw = raw.get("f0")
        try:
            pitch_val = float(pitch_raw)
        except Exception:
            pitch_val = None
        if pitch_val is not None and pitch_val <= 0:
            pitch_val = None

        if hasattr(pitch_raw, "f0"):
            pitch_raw = pitch_raw.f0

        self.pitch_smoother.update(pitch_raw, confidence)

        # F0 locking
        f0_cal = pitch_val
        if f0_cal is not None:
            if self.session.voice_type in ("bass", "baritone"):
                low, high = 60, 260
            else:
                low, high = 70, 320
            if not (low <= f0_cal <= high):
                f0_cal = None

        if self._f0_locked is None:
            if f0_cal is not None:
                self._f0_candidates.append(f0_cal)
                if len(self._f0_candidates) >= 3:
                    self._f0_locked = float(np.median(self._f0_candidates))
        else:
            f0_cal = self._f0_locked

        # Raw formants
        raw_f1 = raw.get("f1") or raw.get("fb_f1")
        raw_f2 = raw.get("f2") or raw.get("fb_f2")

        if target is not None:
            f1, f2 = raw_f1, raw_f2
        else:
            f1, f2, _ = self.formant_smoother.update(raw_f1, raw_f2, None, confidence)

        # Capture gating
        if target is not None and raw_f1 is not None and raw_f2 is not None:
            ok, reason = is_plausible_formants(
                f1, f2,
                voice_type=self.session.voice_type,
                vowel=target,
                calibrated=self.session.data,
            )
            if ok and f0_cal is not None:
                self._capture_buffer.append((raw_f1, raw_f2, f0_cal))

        # Spectrogram
        if not self._mic_active:
            return

        segment = raw.get("segment")
        if segment is None:
            return

        seg_arr = np.asarray(segment, dtype=float).flatten()
        if seg_arr.size == 0:
            return

        self._spec_buffer = np.concatenate((self._spec_buffer, seg_arr))

        # Rolling 1s buffer
        sr = 48000
        max_samples = int(sr * 1.0)
        if self._spec_buffer.size > max_samples:
            self._spec_buffer = self._spec_buffer[-max_samples:]

        # Draw spectrogram
        freqs, times, S = safe_spectrogram(self._spec_buffer, sr)
        update_spectrogram(self, freqs, times, S)

        # Draw vowel anchors
        self._redraw_vowel_anchors()

        # Throttled draw
        now = time.time()
        if now - self._last_draw >= self._min_draw_interval:
            self.canvas.draw_idle()
            self._last_draw = now

    # ---------------------------------------------------------
    # Vowel anchor redraw
    # ---------------------------------------------------------
    def _redraw_vowel_anchors(self):
        ax = self.ax_vowel
        ax.cla()
        ax.set_xlabel("F2 (Hz)")
        ax.set_ylabel("F1 (Hz)")
        ax.invert_xaxis()
        ax.invert_yaxis()

        for vowel, (f1, f2) in self._vowel_anchors.items():
            color = self._vowel_colors.get(vowel, "black")
            ax.scatter(f2, f1, s=180, c=color, marker="x", linewidths=3)
            ax.text(f2, f1, f" /{vowel}/", fontsize=9, color=color)

    # ---------------------------------------------------------
    # Capture processing
    # ---------------------------------------------------------
    def _process_capture(self):
        self.phase_timer.stop()

        vowel = self.state.current_vowel
        buffer_snapshot = list(self._capture_buffer)

        if not buffer_snapshot:
            self.status_panel.appendPlainText(
                f"No valid frames for /{vowel}/ — please sing again"
            )
            self.session.increment_retry(vowel)
            self._capture_buffer.clear()
            self.state.retry_current_vowel()
            self.phase_timer.start(1000)
            return

        f1_vals, f2_vals, f0_vals = zip(*buffer_snapshot)
        f1_arr = np.asarray(f1_vals, dtype=float)
        f2_arr = np.asarray(f2_vals, dtype=float)
        f0_arr = np.asarray([v for v in f0_vals if v is not None], dtype=float)

        if f0_arr.size == 0:
            self.status_panel.appendPlainText(
                f"No valid pitch for /{vowel}/ — retrying"
            )
            self.session.increment_retry(vowel)
            self._capture_buffer.clear()
            self.phase_timer.start(1000)
            return

        f1_med = float(np.median(f1_arr))
        f2_med = float(np.median(f2_arr))
        f0_med = float(np.median(f0_arr))

        self._capture_buffer.clear()
        self._apply_capture_result(vowel, f1_med, f2_med, f0_med)

        self.phase_timer.start(1000)

    def _apply_capture_result(self, vowel, f1, f2, f0):
        accepted, should_retry, msg = self.session.handle_result(
            vowel,
            f1, f2, f0,
            confidence=1.0,
            stability=0.0,
        )

        self.status_panel.appendPlainText(msg)

        if accepted:
            self.status_panel.appendPlainText(
                f"/{vowel}/ F1={f1:.1f}, F2={f2:.1f}, F0={f0:.1f}"
            )

            # Store durable anchor
            self._vowel_anchors[vowel] = (f1, f2)
            self._redraw_vowel_anchors()
            self.canvas.draw_idle()

            self._reset_f0_lock()
            self.state.advance()
            return True

        if should_retry:
            self.status_panel.appendPlainText(
                f"Retrying /{vowel}/ — please sing clearly"
            )
            return False

        return False

    # ---------------------------------------------------------
    # Finish and save profile
    # ---------------------------------------------------------
    def _finish(self):
        try:
            self.phase_timer.stop()
            self.poll_timer.stop()
        except Exception:
            pass

        try:
            self.engine.use_hybrid = True
            self.engine.calibrating = False
            self.engine.vowel_hint = None
        except Exception:
            pass

        if hasattr(self.engine, "stop_stream"):
            try:
                self.engine.stop_stream()
            except Exception:
                pass

        try:
            self.label_smoother.reset()
            self.pitch_smoother.reset()
            self.formant_smoother.reset()
        except Exception:
            pass

        self._capture_buffer.clear()
        self._spec_buffer = np.array([], dtype=float)

        try:
            base_name = self.session.save_profile()
            self.status_panel.appendPlainText(f"Profile saved for {base_name}")
        except Exception:
            traceback.print_exc()
            self.status_panel.appendPlainText("Failed to save profile.")
            base_name = f"{self.session.profile_name}_{self.session.voice_type}"

        try:
            self.profile_calibrated.emit(base_name)  # type:ignore
        except Exception:
            traceback.print_exc()

        self.close()

    def closeEvent(self, event):
        # Stop timers
        try:
            self.phase_timer.stop()
            self.poll_timer.stop()
        except Exception:
            pass

        # Always stop the mic if calibration is aborted or force-closed
        try:
            if hasattr(self.engine, "stop_stream"):
                self.engine.stop_stream()
        except Exception:
            pass

        # Reset analyzer state
        try:
            self.analyzer.pause()
            self.engine.calibrating = False
            self.engine.vowel_hint = None
            self.label_smoother.reset()
            self.pitch_smoother.reset()
            self.formant_smoother.reset()
        except Exception:
            pass

        super().closeEvent(event)
