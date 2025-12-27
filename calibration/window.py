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
from calibration.session import CalibrationSession
from calibration.state_machine import CalibrationStateMachine
from calibration.plotter import update_artists, safe_spectrogram


VOWEL_ALIASES = {
    "i": {"i", "ɪ"},
    "ɛ": {"ɛ", "e", "æ"},
    "ɑ": {"a", "æ", "ʌ"},
    "ɔ": {"o", "ɔ", "ʊ"},
    "u": {"u", "ʊ"},
}


class CalibrationWindow(QMainWindow):
    profile_calibrated = pyqtSignal(str)

    def __init__(self, profile_name, voice_type="bass",
                 engine=None, analyzer=None, profile_manager=None,
                 existing_profile=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        # Shared engine
        self.engine = engine
        self.profile_manager = profile_manager

        # Rolling state
        self._last_result = None
        self._capture_buffer = []
        self._spec_buffer = np.array([], dtype=float)

        # Core vowel set
        self.vowels = ["i", "ɛ", "ɑ", "ɔ", "u"]

        self.session = CalibrationSession(
            profile_name=profile_name,
            voice_type=voice_type,
            vowels=self.vowels,
            profile_manager=self.profile_manager,
            existing_profile=existing_profile,
        )
        self.state = CalibrationStateMachine(self.vowels)

        # Build smoothers (confidence-aware)
        self.pitch_smoother = PitchSmoother(sr=48000, min_confidence=0.25)
        self.formant_smoother = MedianSmoother(min_confidence=0.25)
        self.label_smoother = LabelSmoother(min_confidence=0.25)
        # Analyzer (shared with tuner)
        self.analyzer = analyzer if analyzer is not None else (
            LiveAnalyzer(engine=self.engine,
                         pitch_smoother=self.pitch_smoother,
                         formant_smoother=self.formant_smoother,
                         label_smoother=self.label_smoother))
        self.capture_timeout = 3.0

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
            "ɛ": "green",
            "ɑ": "blue",
            "ɔ": "purple",
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
            self.status_panel.appendPlainText(f"Sing /{event['vowel']}/…")
            self.state.force_capture_mode()

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

        # Timeout check
        if (event["event"] == "capture_tick"
                and self.state.check_timeout(self.capture_timeout)):
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
        # Pull latest analyzer frame (NOT engine)
        raw = self.analyzer.get_latest_raw()
        if raw is None:
            return

        # Extract pitch (unwrap PitchResult)
        pitch = raw.get("f0")

        if hasattr(pitch, "f0"):
            f0 = pitch.f0
        else:
            f0 = pitch
        print("DEBUG PITCH:", pitch, "→ f0:", f0)
        # Sanitize pitch
        if f0 is None or not np.isfinite(f0):
            f0 = None
        # Extract formants
        form = raw.get("formants")
        if isinstance(form, (tuple, list)) and len(form) >= 2:
            f1, f2 = form[0], form[1]
        else:
            f1, f2 = None, None

        confidence = raw.get("confidence", 0.0)

        # -----------------------------
        # Capture gating (calibration)
        # -----------------------------
        target = self.state.current_vowel
        if (
                target is not None
                and f1 is not None
                and f2 is not None
                and confidence >= 0.25
        ):
            self._capture_buffer.append((f1, f2, f0))

        # -----------------------------
        # Spectrogram
        # -----------------------------
        segment = raw.get("segment")
        if segment is None:
            return

        seg_arr = np.asarray(segment, dtype=float).flatten()
        if seg_arr.size == 0:
            return

        self._spec_buffer = np.concatenate((self._spec_buffer, seg_arr))

        sr = 48000
        max_seconds = 1.0
        max_samples = int(sr * max_seconds)

        if self._spec_buffer.size > max_samples:
            self._spec_buffer = self._spec_buffer[-max_samples:]

        if self._spec_buffer.size > 1024:
            freqs, times, S = safe_spectrogram(
                self._spec_buffer,
                sr,
                n_fft=1024,
                hop_length=256,
            )
            update_artists(
                self,
                freqs,
                times,
                S,
                f1,
                f2,
                target,
            )

        # Draw throttled
        now = time.time()
        if now - self._last_draw >= self._min_draw_interval:
            self.canvas.draw_idle()
            self._last_draw = now

    # ---------------------------------------------------------
    # Capture processing
    # ---------------------------------------------------------
    def _process_capture(self):
        self.phase_timer.stop()

        vowel = self.state.current_vowel
        buffer_snapshot = list(self._capture_buffer)

        # No audio captured
        if not buffer_snapshot:
            self.status_panel.appendPlainText(
                f"No audio captured for /{vowel}/ — retrying"
            )
            self.session.increment_retry(vowel)
            self.state.retry_current_vowel()
            self.phase_timer.start(1000)
            return

        # Extract arrays
        f1_vals, f2_vals, f0_vals = zip(*buffer_snapshot)
        f1_arr = np.asarray(f1_vals, dtype=float)
        f2_arr = np.asarray(f2_vals, dtype=float)
        # Clean f0: keep only finite floats, drop None
        f0_clean = [v for v in f0_vals if v is not None and np.isfinite(v)]
        f0_arr = (
            np.asarray(f0_clean, dtype=float)
            if f0_clean
            else np.array([], dtype=float)
        )

        # No plausible frames
        if f1_arr.size == 0 or f2_arr.size == 0:
            self.status_panel.appendPlainText(
                f"Retrying: no plausible frames for /{vowel}/"
            )
            self.session.increment_retry(vowel)
            ev = self.state.retry_current_vowel()
            if ev["event"] == "max_retries":
                f1_best, f2_best, f0_best = self._pick_best_attempt(buffer_snapshot)
                accepted, skipped, msg = self.session.handle_result(
                    vowel,
                    f1_best, f2_best, f0_best,
                    confidence=0.0,
                    stability=float("inf"),
                )
                self.status_panel.appendPlainText(msg)
                self.state.advance()
            self.phase_timer.start(1000)
            return

        # Medians
        f1_med = float(np.median(f1_arr))
        f2_med = float(np.median(f2_arr))
        f0_med = float(np.median(f0_arr)) if f0_arr.size > 0 else None

        # Submit to session
        accepted, skipped, msg = self.session.handle_result(
            vowel,
            f1_med, f2_med, f0_med,
            confidence=1.0,
            stability=0.0,
        )
        self.status_panel.appendPlainText(msg)
        self._capture_buffer.clear()

        if accepted or skipped:
            self.capture_panel.appendPlainText(
                f"/{vowel}/ F1={f1_med:.1f}, F2={f2_med:.1f}, F0={(f0_med or 0):.1f}"
            )
            self.state.advance()
            self.phase_timer.start(1000)
            return

        # Retry case
        ev = self.state.retry_current_vowel()
        if ev["event"] == "max_retries":
            f1_best, f2_best, f0_best = self._pick_best_attempt()
            accepted, skipped, msg = self.session.handle_result(
                vowel,
                f1_best, f2_best, f0_best,
                confidence=1.0,
                stability=0.0,
            )
            self.capture_panel.appendPlainText(
                f"/{vowel}/ F1={f1_best:.1f}, F2={f2_best:.1f}, F0={(f0_best or 0):.1f}"
            )
            self.status_panel.appendPlainText(msg)
            self.state.advance()

        self.phase_timer.start(1000)

    def _pick_best_attempt(self, buffer=None):
        buf = buffer if buffer is not None else self._capture_buffer
        if not buf:
            return None, None, None

        f1_vals, f2_vals, f0_vals = zip(*buf)
        f1_med = float(np.median(f1_vals))
        f2_med = float(np.median(f2_vals))
        f0_med = float(np.median(f0_vals))
        return f1_med, f2_med, f0_med

    def closeEvent(self, event):
        try:
            self.phase_timer.stop()
            self.poll_timer.stop()
        except Exception:
            pass
        super().closeEvent(event)

    # ---------------------------------------------------------
    # Finish and save profile
    # ---------------------------------------------------------
    def _finish(self):
        try:
            self.phase_timer.stop()
            self.poll_timer.stop()
        except Exception:
            pass

        # Stop engine stream if present
        if hasattr(self.engine, "stop_stream"):
            try:
                self.engine.stop_stream()
            except Exception:
                pass

        # Reset smoothers
        try:
            self.label_smoother.reset()
            self.pitch_smoother.reset()
            self.formant_smoother.reset()
        except Exception:
            pass

        self._capture_buffer.clear()
        self._spec_buffer = np.array([], dtype=float)

        # Save profile
        try:
            base_name = self.session.save_profile()
            self.status_panel.appendPlainText(f"Profile saved for {base_name}")
        except Exception:
            traceback.print_exc()
            self.status_panel.appendPlainText("Failed to save profile.")
            base_name = f"{self.session.profile_name}_{self.session.voice_type}"

        # Notify tuner
        try:
            self.profile_calibrated.emit(base_name)  # type:ignore
        except Exception:
            traceback.print_exc()

        self.close()
