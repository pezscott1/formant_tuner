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
from analysis.vowel import is_plausible_formants
from calibration.session import CalibrationSession
from calibration.state_machine import CalibrationStateMachine
from calibration.plotter import update_artists, safe_spectrogram


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
                 engine=None, analyzer=None, parent=None):
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
        self.session = CalibrationSession(
            profile_name=profile_name,
            voice_type=voice_type,
            vowels=self.vowels,
        )
        self.state = CalibrationStateMachine(self.vowels)
        # Shared engine
        self.engine = engine
        # Build smoothers
        self.pitch_smoother = PitchSmoother(alpha=0.25, voice_type=voice_type)
        self.formant_smoother = MedianSmoother(window=5)
        self.label_smoother = LabelSmoother(hold_frames=4)
        if analyzer is None:
            self.analyzer = LiveAnalyzer(
                engine=self.engine,
                pitch_smoother=self.pitch_smoother,
                formant_smoother=self.formant_smoother,
                label_smoother=self.label_smoother,
            )
        else:
            self.analyzer = analyzer
        # Capture behavior
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
            # >>> NEW: enter capture immediately <<<
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
    def _poll_audio(self):  # noqa: C901
        """
        Poll the shared engine for the latest raw frame.
        Build a rolling audio buffer for spectrogram display.
        """

        raw = self.engine.get_latest_raw()
        print("RAW:", raw)
        if raw is None:
            return
        # Process through LiveAnalyzer (adds smoothing + stability)
        frame = self.analyzer.process_raw(raw)

        self._last_result = frame
        f0 = frame["f0"]
        f1, f2, f3 = frame["formants"]
        # Fallback to raw LPC formants if smoother output is None
        fb_f1 = frame.get("fb_f1")
        fb_f2 = frame.get("fb_f2")

        if f1 is None and fb_f1 is not None:
            f1 = float(fb_f1)
        if f2 is None and fb_f2 is not None:
            f2 = float(fb_f2)
        # -----------------------------------------------------
        # Debug vowel guess (but do NOT use it for gating)
        # -----------------------------------------------------
        vowel_guess_raw = raw.get("vowel_guess")
        self.label_smoother.update(vowel_guess_raw)

        target = self.state.current_vowel
        # inside _poll_audio, after frame = self.analyzer.process_raw(raw)
        print("DBG frame:", {
            "f0": frame.get("f0"),
            "f1": frame["formants"][0],
            "f2": frame["formants"][1],
            "vowel_guess": raw.get("vowel_guess"),
        })

        # right before gating
        print("DBG gating check:", {
            "target": target,
            "f0": f0, "f1": f1, "f2": f2,
        })
        # -----------------------------------------------------
        # Capture gating (NO classifier, NO stability)
        # Use voice-type-only plausibility to match LiveAnalyzer.
        # -----------------------------------------------------
        if not hasattr(self, "_partial_buffer"):
            self._partial_buffer = []

        if target is not None and f1 is not None and f2 is not None:
            ok, reason = is_plausible_formants(
                f1, f2,
                voice_type=self.session.voice_type,
            )
            print("DBG DECIDE", target, f1, f2, f0, "ok=", ok, "reason=", reason)
            if ok:
                self._capture_buffer.append((f1, f2, f0))
        elif target is not None and (f1 is not None or f2 is not None):
            # fallback: store partial frames for median-only fallback
            self._partial_buffer.append((f1, f2, f0))
            if len(self._partial_buffer) > 120:
                self._partial_buffer.pop(0)
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
    def _process_capture(self):  # noqa: C901
        # -----------------------------------------------------
        # Freeze phase timer so UI cannot advance mid-processing
        # -----------------------------------------------------
        self.phase_timer.stop()

        # Capture vowel NOW so all messages stay consistent
        vowel = self.state.current_vowel
        # Snapshot buffer BEFORE any clearing happens
        buffer_snapshot = list(self._capture_buffer)
        if not self._capture_buffer and getattr(self, "_partial_buffer", None):
            f1_vals = np.array([p[0] for p in
                                self._partial_buffer if p[0] is not None], dtype=float)
            f2_vals = np.array([p[1] for p in
                                self._partial_buffer if p[1] is not None], dtype=float)
            f0_vals = np.array([p[2] for p in
                                self._partial_buffer if p[2] is not None], dtype=float)

            if f1_vals.size > 0 and f2_vals.size > 0:
                f1_med = float(np.median(f1_vals))
                f2_med = float(np.median(f2_vals))
                f0_med = float(np.median(f0_vals)) if f0_vals.size > 0 else None
                accepted, skipped, msg = (
                    self.session.handle_result(f1_med, f2_med, f0_med))
                self.status_panel.appendPlainText(msg)
                print("DBG handle_result:",
                      {"accepted": accepted, "skipped":
                          skipped, "session_index": self.session.current_index})
                self.capture_panel.appendPlainText(
                    f"/{vowel}/ F1={f1_med:.1f}"
                    f", F2={f2_med:.1f}, F0={(f0_med or 0):.1f}"
                )
                self._partial_buffer.clear()
                self.state.advance()
                self.phase_timer.start(1000)
                return

        # -----------------------------------------------------
        # No audio captured
        # -----------------------------------------------------
        if not self._capture_buffer:
            self.status_panel.appendPlainText(
                f"No audio captured for /{vowel}/ — retrying"
            )
            # Track retries in session as well as state machine
            if hasattr(self.session, "increment_retry"):
                self.session.increment_retry(vowel)
            else:
                # fallback: direct map increment
                self.session.retries_map[vowel] = (
                        self.session.retries_map.get(vowel, 0) + 1)

            self.state.retry_current_vowel()
            self.phase_timer.start(1000)
            return

        # -----------------------------------------------------
        # Extract arrays
        # -----------------------------------------------------
        f1_vals, f2_vals, f0_vals = zip(*self._capture_buffer)
        f1_arr = np.asarray(f1_vals, dtype=float)
        f2_arr = np.asarray(f2_vals, dtype=float)
        f0_arr = np.asarray(f0_vals, dtype=float)

        # Remove NaN/inf
        f1_arr = f1_arr[np.isfinite(f1_arr)]
        f2_arr = f2_arr[np.isfinite(f2_arr)]
        f0_arr = f0_arr[np.isfinite(f0_arr)]

        # -----------------------------------------------------
        # Per-frame plausibility
        # -----------------------------------------------------
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
        # Always mask F0 to match F1/F2 frames
        if f0_arr.size == mask.size:
            f0_arr = f0_arr[mask]
        else:
            # Reconstruct F0 array aligned to F1/F2 frames
            f0_arr = np.array([p[2] for p, keep in
                               zip(buffer_snapshot, mask) if keep and p[2] is not None],
                              dtype=float)
        print(f"DBG capture snapshot len: {len(buffer_snapshot)}")
        print(f"DBG f1_arr size: {f1_arr.size}, "
              f"f2_arr size: {f2_arr.size}, f0_arr size: {f0_arr.size}")
        print(f"DBG mask sum: {mask.sum()} / {mask.size}")
        # -----------------------------------------------------
        # No plausible frames
        # -----------------------------------------------------
        if f1_arr.size == 0 or f2_arr.size == 0:
            self.status_panel.appendPlainText(
                f"Retrying: no plausible frames for /{vowel}/"
            )
            if hasattr(self.session, "increment_retry"):
                self.session.increment_retry(vowel)
            else:
                self.session.retries_map[vowel] = (
                        self.session.retries_map.get(vowel, 0) + 1)
            snapshot = buffer_snapshot
            self._capture_buffer.clear()
            ev = self.state.retry_current_vowel()

            # Max retries?
            if ev["event"] == "max_retries":
                f1_best, f2_best, f0_best = self._pick_best_attempt(snapshot)
                accepted, skipped, msg = self.session.handle_result(
                    f1_best, f2_best, f0_best
                )
                self.status_panel.appendPlainText(msg)
                self._capture_buffer.clear()
                self.state.advance()
                self.phase_timer.start(1000)
                return

            self.phase_timer.start(1000)
            return

        # -----------------------------------------------------
        # Minimum frames check
        # -----------------------------------------------------
        min_frames = 1 if vowel in ("ɔ", "u") else 2
        if f1_arr.size < min_frames or f2_arr.size < min_frames:
            self.status_panel.appendPlainText(
                f"Low confidence for /{vowel}/ — retrying"
            )
            if hasattr(self.session, "increment_retry"):
                self.session.increment_retry(vowel)
            else:
                self.session.retries_map[vowel] = (
                        self.session.retries_map.get(vowel, 0) + 1)

            self._capture_buffer.clear()
            self.state.retry_current_vowel()
            self.phase_timer.start(1000)
            return

        # -----------------------------------------------------
        # Medians
        # -----------------------------------------------------
        f1_med = float(np.median(f1_arr))
        f2_med = float(np.median(f2_arr))
        f0_med = float(np.median(f0_arr)) if f0_arr.size > 0 else None

        # -----------------------------------------------------
        # Vowel-specific refinements
        # -----------------------------------------------------
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

        # -----------------------------------------------------
        # Submit to session
        # -----------------------------------------------------
        accepted, skipped, msg = (
            self.session.handle_result(f1_med, f2_med, f0_med))
        self.status_panel.appendPlainText(msg)
        self._capture_buffer.clear()

        if accepted:
            vowel = self.state.current_vowel
            self.capture_panel.appendPlainText(
                f"/{vowel}/ F1={f1_med:.1f}, "
                f"F2={f2_med:.1f}, F0={(f0_med or 0):.1f}"
            )
            self.state.advance()
            self.phase_timer.start(1000)
            return

        if skipped:
            self.state.advance()
            self.phase_timer.start(1000)
            return

        # Retry case
        ev = self.state.retry_current_vowel()
        if ev["event"] == "max_retries":
            f1_best, f2_best, f0_best = self._pick_best_attempt()
            self.status_panel.appendPlainText(
                f"Max retries reached — accepting best /{vowel}/"
            )
            accepted, skipped, msg = self.session.handle_result(
                f1_best, f2_best, f0_best
            )
            self.capture_panel.appendPlainText(
                f"/{vowel}/ F1={f1_best:.1f}, "
                f"F2={f2_best:.1f}, F0={(f0_best or 0):.1f}"
            )
            self.status_panel.appendPlainText(msg)
            self._capture_buffer.clear()
            self.state.advance()
            self.phase_timer.start(1000)
            return

        self.phase_timer.start(1000)
        return

    def _pick_best_attempt(self, buffer=None):
        """
        Choose the best attempt from the capture buffer or a provided snapshot.
        """
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

        if hasattr(self.engine, "stop_stream"):
            try:
                self.engine.stop_stream()
            except Exception:
                pass

        super().closeEvent(event)

    # ---------------------------------------------------------
    # Finish and save profile
    # ---------------------------------------------------------
    def _finish(self):  # noqa: C901
        """
        Save profile, stop timers, stop audio, reset smoothers,
        and notify parent.
        """
        # Stop timers
        try:
            self.phase_timer.stop()
            self.poll_timer.stop()
        except Exception:
            pass

        # Stop audio stream if engine supports it
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

        # Clear buffers
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

        # Notify parent
        try:
            self.profile_calibrated.emit(base_name)  # type:ignore
        except Exception:
            traceback.print_exc()

        self.close()
