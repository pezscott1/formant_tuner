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

from analysis.smoothing import LabelSmoother, PitchSmoother, MedianSmoother
from analysis.vowel import is_plausible_formants
from calibration.session import CalibrationSession
from calibration.state_machine import CalibrationStateMachine
from calibration.plotter import update_artists, safe_spectrogram
from analysis.engine import FormantAnalysisEngine


# Alias table still useful for debugging / live display, even if
# we no longer *gate* calibration on classifier labels.
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
                 analyzer=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")

        # Rolling state
        self._last_frame = None
        self._last_result = None
        self._capture_buffer = []
        self._spec_buffer = np.array([], dtype=float)

        # Stability tracking (for debugging only now)
        self._stable_vowel = None
        self._stable_count = 0
        self._stable_required = 5

        # Core vowel set
        self.vowels = ["i", "ɛ", "ɑ", "ɔ", "u"]

        # Core objects
        self.analyzer = analyzer
        self.session = CalibrationSession(
            profile_name=profile_name,
            voice_type=voice_type,
            vowels=self.vowels,
        )
        self.state = CalibrationStateMachine(self.vowels)

        # Shared engine
        self.engine = analyzer or FormantAnalysisEngine(voice_type=voice_type)

        # Capture behavior
        self.capture_timeout = 3.0

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

        left = self._build_left_panel()
        layout.addWidget(left)

        right = self._build_right_panel()
        layout.addWidget(right, stretch=1)

        # -----------------------------------------------------
        # Timers
        # -----------------------------------------------------
        self.phase_timer = QTimer(self)
        self.phase_timer.timeout.connect(self._tick_phase)
        self.phase_timer.start(1000)

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_audio)
        self.poll_timer.start(80)

        # -----------------------------------------------------
        # Window sizing
        # -----------------------------------------------------
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
        """
        Poll the shared engine for the latest raw frame.
        Build a rolling audio buffer for spectrogram display.
        """

        raw = self.engine.get_latest_raw()
        if raw is None:
            return
        print("RAW:", raw.get("f0"), raw.get("formants"))
        self._last_result = raw
        f0 = raw.get("f0")
        f1, f2, f3 = raw.get("formants", (None, None, None))
        self._last_frame = (f1, f2, f0)

        # -----------------------------------------------------
        # Debug vowel guess (but do NOT use it for gating)
        # -----------------------------------------------------
        vowel_guess_raw = raw.get("vowel_guess")
        vowel_guess = self.label_smoother.update(vowel_guess_raw)

        target = self.state.current_vowel

        if target == "ɛ":
            print(
                f"[DEBUG /ɛ/] guess_raw={vowel_guess_raw!r}, "
                f"guess_smooth={vowel_guess!r}, "
                f"f1={f1}, f2={f2}, f0={f0}"
            )

        if target == "ɑ":
            print(
                f"[DEBUG /ɑ/] guess_raw={vowel_guess_raw!r}, "
                f"guess_smooth={vowel_guess!r}, "
                f"f1={f1}, f2={f2}, f0={f0}"
            )

        # -----------------------------------------------------
        # Capture gating (NO classifier, NO stability)
        # -----------------------------------------------------
        if (
            target is not None
            and f1 is not None and f2 is not None and f0 is not None
            and 40 <= f0 <= 800
        ):
            ok, _ = is_plausible_formants(
                f1,
                f2,
                voice_type=self.session.voice_type,
                vowel=target,
                calibrated=getattr(self.session, "calibrated_profile", None),
            )
            if ok:
                self._capture_buffer.append((float(f1), float(f2), float(f0)))

        # -----------------------------------------------------
        # Spectrogram
        # -----------------------------------------------------
        segment = raw.get("segment")
        if segment is None:
            return

        try:
            seg_arr = np.asarray(segment, dtype=float).flatten()
            if seg_arr.size == 0:
                return

            self._spec_buffer = np.concatenate((self._spec_buffer, seg_arr))

            sr = getattr(self.engine, "sample_rate", 44100)
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
                    self.state.current_vowel,
                )

        except Exception:
            traceback.print_exc()

        # Throttled draw
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
        if not self._capture_buffer:
            vowel = self.state.current_vowel
            self.status_panel.appendPlainText(
                f"No audio captured for /{vowel}/ — retrying")
            self.state.retry_current_vowel()
            return

        vowel = self.state.current_vowel

        f1_vals, f2_vals, f0_vals = zip(*self._capture_buffer)
        f1_arr = np.asarray(f1_vals, dtype=float)
        f2_arr = np.asarray(f2_vals, dtype=float)
        f0_arr = np.asarray(f0_vals, dtype=float)

        # Remove NaN/inf
        f1_arr = f1_arr[np.isfinite(f1_arr)]
        f2_arr = f2_arr[np.isfinite(f2_arr)]
        f0_arr = f0_arr[np.isfinite(f0_arr)]

        # Per-frame plausibility
        mask = []
        for f1_val, f2_val in zip(f1_arr, f2_arr):
            ok, _ = is_plausible_formants(
                f1_val,
                f2_val,
                voice_type=self.session.voice_type,
                vowel=vowel,
                calibrated=getattr(self.session, "calibrated_profile", None),
            )
            mask.append(ok)

        mask = np.asarray(mask, dtype=bool)
        f1_arr = f1_arr[mask]
        f2_arr = f2_arr[mask]
        f0_arr = f0_arr[mask] if f0_arr.size == mask.size else f0_arr

        if f1_arr.size == 0 or f2_arr.size == 0:
            self.status_panel.appendPlainText("Retrying: no plausible frames")
            self._capture_buffer.clear()
            self.state.retry_current_vowel()
            return

        # Minimum frames
        min_frames = 1 if vowel in ("ɔ", "u") else 2
        if f1_arr.size < min_frames or f2_arr.size < min_frames:
            self.status_panel.appendPlainText(
                f"Low confidence for /{vowel}/ — using available frames"
            )

        # Medians
        f1_med = float(np.median(f1_arr))
        f2_med = float(np.median(f2_arr))
        f0_med = float(np.median(f0_arr)) if f0_arr.size > 0 else None

        # Vowel-specific refinements
        if vowel == "i":
            good_f1 = f1_arr[(150 <= f1_arr) & (f1_arr <= 450)]
            good_f2 = f2_arr[(1800 <= f2_arr) & (f2_arr <= 3200)]
            if good_f1.size > 0:
                f1_med = float(np.median(good_f1))
            if good_f2.size > 0:
                f2_med = float(np.median(good_f2))

        elif vowel == "ɛ":
            good_f1 = f1_arr[(300 <= f1_arr) & (f1_arr <= 800)]
            good_f2 = f2_arr[(1500 <= f2_arr) & (f2_arr <= 3000)]
            if good_f1.size > 0:
                f1_med = float(np.median(good_f1))
            if good_f2.size > 0:
                f2_med = float(np.median(good_f2))

        elif vowel == "ɑ":
            good_f1 = f1_arr[(400 <= f1_arr) & (f1_arr <= 1100)]
            good_f2 = f2_arr[(800 <= f2_arr) & (f2_arr <= 3000)]
            if good_f1.size > 0:
                f1_med = float(np.median(good_f1))
            if good_f2.size > 0:
                f2_med = float(np.median(good_f2))

        elif vowel == "ɔ":
            good_f1 = f1_arr[(200 <= f1_arr) & (f1_arr <= 800)]
            good_f2 = f2_arr[(400 <= f2_arr) & (f2_arr <= 1800)]
            if good_f1.size > 0:
                f1_med = float(np.median(good_f1))
            if good_f2.size > 0:
                f2_med = float(np.median(good_f2))

        elif vowel == "u":
            good_f2 = f2_arr[(300 <= f2_arr) & (f2_arr <= 1500)]
            if good_f2.size > 0:
                f2_med = float(np.median(good_f2))

        # Submit to session
        accepted, skipped, msg = self.session.handle_result(f1_med, f2_med, f0_med)
        self.status_panel.appendPlainText(msg)
        self._capture_buffer.clear()

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

        # Retry case
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

        try:
            self.profile_calibrated.emit(base_name)
        except Exception:
            traceback.print_exc()

        self.close()
