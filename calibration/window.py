# calibration/window.py
import time
import traceback
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QFrame, QPushButton,
    QApplication, QCheckBox,
)
from PyQt6.QtCore import QTimer, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from analysis.vowel_data import STANDARD_VOWELS, expanded_vowels_for_voice
from tuner.live_analyzer import LiveAnalyzer
from analysis.smoothing import LabelSmoother, PitchSmoother, MedianSmoother
from analysis.plausibility import is_plausible_formants
from calibration.session import CalibrationSession
from calibration.state_machine import CalibrationStateMachine
from calibration.plotter import safe_spectrogram, update_spectrogram
from matplotlib.lines import Line2D
from profile_viewer.vowel_colors import vowel_color_for


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
        expanded_mode=False,
        optional_vowels=None,
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
        self.optional_vowels = optional_vowels or []
        self.expanded_mode = expanded_mode

        if self.expanded_mode:
            # Use the full expanded list in the correct order
            self.vowels_to_calibrate = expanded_vowels_for_voice(voice_type)
        else:
            self.vowels_to_calibrate = STANDARD_VOWELS

        # Calibration session
        self.session = CalibrationSession(
            profile_name=profile_name,
            voice_type=voice_type,
            vowels=self.vowels_to_calibrate,
            profile_manager=self.profile_manager,
            existing_profile=existing_profile,
        )
        self._vowel_anchors = {
            v: (entry["f1"], entry["f2"])
            for v, entry in self.session.data.items()
            if entry.get("f1") is not None and entry.get("f2") is not None
        }

        self.state = CalibrationStateMachine(self.vowels_to_calibrate)
        print("Calibrating vowels:", self.vowels_to_calibrate)
        # Smoothers
        self.pitch_smoother = PitchSmoother(min_confidence=0.25)
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
        self._interpolated_vowels = {}
        all_vowels = self.vowels_to_calibrate

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
        self.show_interpolated = True
        self.chk_show_interpolated = QCheckBox("Show interpolated vowels")
        self.chk_show_interpolated.setChecked(True)
        self.chk_show_interpolated.stateChanged.connect(  # type: ignore
            self._on_toggle_interpolated)
        layout.addWidget(self.chk_show_interpolated)
        # -----------------------------------------------------
        # Timers
        # -----------------------------------------------------
        self.phase_timer = QTimer(self)
        self.phase_timer.timeout.connect(self._tick_phase)  # type:ignore
        self.phase_timer.start(1000)

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_audio)  # type:ignore
        self.poll_timer.start(80)

        screen = QApplication.primaryScreen().availableGeometry()
        w = int(screen.width() * 0.80)
        h = int(screen.height() * 0.80)
        self.setMaximumHeight(h)
        self.setMinimumHeight(h)
        self.resize(w, h)

        # Center horizontally and vertically
        self.move(
            screen.center().x() - w // 2,
            screen.center().y() - h // 2,
        )

        self.show()

    def _on_toggle_interpolated(self, state):
        self.show_interpolated = bool(state)
        self._redraw_vowel_anchors()

    def _compute_vowel_axis_limits(self):
        f1_vals = []
        f2_vals = []
        for f1, f2 in self._vowel_anchors.values():
            if f1:
                f1_vals.append(f1)
            if f2:
                f2_vals.append(f2)

        if f1_vals:
            self.f1_min = min(f1_vals) * 0.9
            self.f1_max = max(f1_vals) * 1.1
        if f2_vals:
            self.f2_min = min(f2_vals) * 0.9
            self.f2_max = max(f2_vals) * 1.1

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

        btn = QPushButton("Close")
        btn.clicked.connect(self.close)  # type: ignore
        layout.addWidget(btn)

        return frame

    def _build_right_panel(self):
        frame = QFrame()
        layout = QVBoxLayout(frame)

        self.fig, (self.ax_spec, self.ax_vowel) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.subplots_adjust(
            top=0.97,
            bottom=0.08,
            hspace=0.42
        )

        self.canvas = Canvas(self.fig)
        layout.addWidget(self.canvas)
        self._compute_vowel_axis_limits()
        # Spectrogram axes
        self.ax_spec.set_xlabel("Time (s)", labelpad=10)
        self.ax_spec.set_ylabel("Frequency (Hz)", labelpad=10)
        # Vowel axes
        self.ax_vowel.set_xlabel("F2 (Hz)")
        self.ax_vowel.set_ylabel("F1 (Hz)")

        self._last_draw = 0.0
        self._min_draw_interval = 0.05

        return frame

    # ---------------------------------------------------------
    # Phase machine tick
    # ---------------------------------------------------------

    def _tick_phase(self):
        event = self.state.tick()
        evt = event["event"]

        if evt == "prep_countdown":
            self._mic_active = False
            self.analyzer.pause()
            self.status_panel.appendPlainText(
                f"Prepare: /{self.state.current_vowel}/ in {event['secs']}…")
            self.status_panel.verticalScrollBar().setValue(
                self.status_panel.verticalScrollBar().maximum()
            )

        elif evt == "start_sing":
            self._mic_active = True
            self.pitch_smoother.reset()
            self.analyzer.resume()
            self.vowel_capture_started.emit()  # type:ignore
            self.status_panel.appendPlainText(f"Sing /{event['vowel']}/…")
            self.status_panel.verticalScrollBar().setValue(
                self.status_panel.verticalScrollBar().maximum()
            )

        elif evt == "sing_countdown":
            self.status_panel.appendPlainText(
                f"Sing /{self.state.current_vowel}/ – {event['secs']}s"
            )
            self.status_panel.verticalScrollBar().setValue(
                self.status_panel.verticalScrollBar().maximum()
            )

        elif evt == "start_capture":
            self._mic_active = True
            self.pitch_smoother.reset()
            self.analyzer.resume()
            self.status_panel.appendPlainText(
                f"Capturing /{self.state.current_vowel}/…"
            )
            self.status_panel.verticalScrollBar().setValue(
                self.status_panel.verticalScrollBar().maximum()
            )

        elif evt == "retry":
            self._mic_active = False
            self.pitch_smoother.reset()
            self.analyzer.pause()
            self.vowel_capture_finished.emit()  # type:ignore
            self.status_panel.appendPlainText(
                f"Retrying /{self.state.current_vowel}/…"
            )
            self.status_panel.verticalScrollBar().setValue(
                self.status_panel.verticalScrollBar().maximum()
            )

        elif evt == "max_retries":
            vowel = event["vowel"]
            self.status_panel.appendPlainText(
                f"Skipping /{vowel}/ after {self.state.MAX_RETRIES} failed attempts"
            )
            self.status_panel.verticalScrollBar().setValue(
                self.status_panel.verticalScrollBar().maximum()
            )
            # If recalibrating, keep old anchor and weight
            # (existing_profile is passed into CalibrationSession)
            # So we simply do nothing here.

            adv = event.get("advance")
            if adv and adv["event"] == "finished":
                self.status_panel.appendPlainText("Calibration complete!")
                self._finish()
                return

            # Otherwise continue to next vowel
            return

        elif evt == "next_vowel":
            self._mic_active = False
            self.pitch_smoother.reset()
            self.analyzer.pause()
            self.vowel_capture_finished.emit()  # type:ignore
            self.status_panel.appendPlainText(
                f"Next vowel: /{event['vowel']}/"
            )
            self.status_panel.verticalScrollBar().setValue(
                self.status_panel.verticalScrollBar().maximum()
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
            self.status_panel.verticalScrollBar().setValue(
                self.status_panel.verticalScrollBar().maximum()
            )
            self._finish()
            return

        # Timeout check
        if evt == "capture_tick" and self.state.check_timeout(self.capture_timeout):
            vowel = self.state.current_vowel
            self.status_panel.appendPlainText(
                f"/{vowel}/ capture timed out after {self.capture_timeout:.1f}s"
            )
            self.status_panel.verticalScrollBar().setValue(
                self.status_panel.verticalScrollBar().maximum()
            )
            self._mic_active = False
            self.analyzer.pause()
            self.vowel_capture_finished.emit()  # type:ignore

            ev = self.state.advance()
            if ev["event"] == "finished":
                self.status_panel.appendPlainText("Calibration complete!")
                self.status_panel.verticalScrollBar().setValue(
                    self.status_panel.verticalScrollBar().maximum()
                )
                self._finish()
                return

    # ---------------------------------------------------------
    # Poll shared engine
    # ---------------------------------------------------------

    def _extract_pitch(self, raw):
        pitch_raw = raw.get("f0")

        if hasattr(pitch_raw, "f0"):
            pitch_raw = pitch_raw.f0

        try:
            val = float(pitch_raw)
        except Exception:
            return pitch_raw, None

        if val <= 0:
            return pitch_raw, None

        return pitch_raw, val

    def _extract_formants(self, raw, target, confidence):
        raw_f1 = raw.get("f1") or raw.get("fb_f1")
        raw_f2 = raw.get("f2") or raw.get("fb_f2")

        if target is not None:
            return raw_f1, raw_f2

        f1, f2, _ = self.formant_smoother.update(raw_f1, raw_f2, None, confidence)
        return f1, f2

    def _maybe_capture(self, f1, f2, f0_cal, confidence, target, raw_f1, raw_f2):
        if target is None or raw_f1 is None or raw_f2 is None:
            return

        ok, reason = is_plausible_formants(
            f1, f2,
            voice_type=self.session.voice_type,
            vowel=target,
            calibrated=self.session.data,
        )

        if ok and f0_cal is not None and confidence > 0.5:
            smoothed = self.pitch_smoother.current
            if smoothed is not None and f0_cal > 1.6 * smoothed:
                f0_cal = None

        if f0_cal is not None:
            self._capture_buffer.append((raw_f1, raw_f2, f0_cal))

    def _update_spectrogram(self, raw):
        if not self._mic_active:
            return

        segment = raw.get("segment")
        if segment is None:
            return

        seg_arr = np.asarray(segment, dtype=float).flatten()
        if seg_arr.size == 0:
            return

        self._spec_buffer = np.concatenate((self._spec_buffer, seg_arr))

        sr = 48000
        max_samples = int(sr * 3.0)
        if self._spec_buffer.size > max_samples:
            self._spec_buffer = self._spec_buffer[-max_samples:]

        freqs, times, S = safe_spectrogram(self._spec_buffer, sr)
        update_spectrogram(self, freqs, times, S)

        self._redraw_vowel_anchors()
        self._interpolated_vowels = self.session.compute_interpolated_vowels()

        now = time.time()
        if now - self._last_draw >= self._min_draw_interval:
            self.canvas.draw_idle()
            self._last_draw = now

    def _apply_pitch_plausibility(self, pitch_val):
        """
        Normalize and plausibility‑check raw F0 for calibration.
        Returns a float or None.
        """

        # Must be numeric
        if not isinstance(pitch_val, (int, float)):
            return None

        # Reject non‑positive
        if pitch_val <= 0:
            return None

        # Voice‑type plausibility ranges
        if self.session.voice_type in ("bass", "baritone"):
            low, high = 60, 350
        else:
            low, high = 180, 500

        if not (low <= pitch_val <= high):
            return None

        return float(pitch_val)

    def _poll_audio(self):
        raw = self.analyzer.get_latest_raw()
        if raw is None:
            return

        confidence = raw.get("confidence", 0.0)
        target = self.state.current_vowel
        self.engine.vowel_hint = target

        pitch_raw, pitch_val = self._extract_pitch(raw)
        self.pitch_smoother.update(pitch_val, confidence)

        f0_cal = self._apply_pitch_plausibility(pitch_val)

        if f0_cal is not None:
            float(f0_cal)  # explicit narrowing
        else:
            f0_cal = None

        raw_f1 = raw.get("f1") or raw.get("fb_f1")
        raw_f2 = raw.get("f2") or raw.get("fb_f2")
        f1, f2 = self._extract_formants(raw, target, confidence)

        self._maybe_capture(f1, f2, f0_cal, confidence, target, raw_f1, raw_f2)

        self._update_spectrogram(raw)

    # ---------------------------------------------------------
    # Vowel anchor redraw
    # ---------------------------------------------------------
    def _redraw_vowel_anchors(self):
        # Ensure interpolated vowels never include calibrated ones
        clean_interp = {
            v: data for v, data in self._interpolated_vowels.items()
            if v not in self._vowel_anchors
        }

        ax = self.ax_vowel
        ax.cla()
        if hasattr(self, "f1_min"):
            ax.set_ylim(self.f1_max, self.f1_min)
        if hasattr(self, "f2_min"):
            ax.set_xlim(self.f2_max, self.f2_min)

        ax.set_xlabel("F2 (Hz)")
        ax.set_ylabel("F1 (Hz)")
        ax.invert_xaxis()
        ax.invert_yaxis()

        # Softer, thicker grid like Profile Viewer
        ax.grid(
            True,
            linewidth=1.5,
            color="#cccccc",
            alpha=0.8,
        )

        # Anti-aliasing
        for spine in ax.spines.values():
            spine.set_antialiased(True)

        # ---------------------------------------------------------
        # Draw vowels: calibrated take priority over interpolated
        # ---------------------------------------------------------

        # All vowels that might appear
        all_vowels = set(self._vowel_anchors.keys()) | set(clean_interp.keys())

        for vowel in all_vowels:
            if vowel in self._vowel_anchors:
                # --- Calibrated vowel (X marker) ---
                f1, f2 = self._vowel_anchors[vowel]
                color = vowel_color_for(vowel)

                ax.scatter(
                    f2, f1,
                    s=200,
                    c=color,
                    marker="x",
                    linewidths=3,
                    antialiased=True,
                )
                ax.text(
                    f2 + 20,
                    f1 - 20,
                    f"/{vowel}/",
                    fontsize=12,
                    fontweight="bold",
                    color=color,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5),
                    ha="left",
                    va="center",
                )
            else:
                if not self.show_interpolated:
                    continue
                # --- Interpolated vowel (circle) ---
                data = clean_interp[vowel]
                f1, f2 = data["f1"], data["f2"]

                ax.scatter(
                    f2, f1,
                    s=160,
                    edgecolors="gray",
                    facecolors="none",
                    marker="o",
                    linewidths=2,
                    antialiased=True,
                )
                ax.text(
                    f2 + 20,
                    f1 - 20,
                    f"/{vowel}/",
                    fontsize=11,
                    color="gray",
                    bbox=dict(facecolor="white", alpha=0.4, edgecolor="none", pad=1.0),
                    ha="left",
                    va="center",
                )

        legend_elements = [
            Line2D(
                [0], [0],
                marker='x',
                color='black',
                markersize=12,
                linewidth=0,
                markeredgewidth=3
            ),
            Line2D(
                [0], [0],
                marker='o',
                color='gray',
                markersize=12,
                fillstyle='none',
                linewidth=2
            ),
        ]

        ax.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            framealpha=0.8
        )

    # ---------------------------------------------------------
    # Capture processing
    # ---------------------------------------------------------
    def _process_capture(self):
        self.phase_timer.stop()

        vowel = self.state.current_vowel
        buffer_snapshot = list(self._capture_buffer)

        # Timestamp for logging
        ts = time.strftime("%H:%M:%S")

        # No frames captured
        if not buffer_snapshot:
            self.status_panel.appendPlainText(
                f"[{ts}] No valid frames for /{vowel}/ — please sing again"
            )
            self.session.increment_retry(vowel)
            self._capture_buffer.clear()
            event = self.state.retry_current_vowel()
            # If max retries reached, handle it immediately
            if event["event"] == "max_retries":
                self.status_panel.appendPlainText(
                    f"[{ts}] Skipping /{vowel}/ after "
                    f"{self.state.MAX_RETRIES} failed attempts"
                )
                # Advance to next vowel (already done inside state machine)
                self.phase_timer.start(1000)
                return
            # Otherwise normal retry
            self.phase_timer.start(1000)
            return

        # Unpack arrays
        f1_vals, f2_vals, f0_vals = zip(*buffer_snapshot)
        f1_arr = np.asarray(f1_vals, dtype=float)
        f2_arr = np.asarray(f2_vals, dtype=float)
        f0_arr = np.asarray([v for v in f0_vals if v is not None], dtype=float)

        # No pitch frames
        if f0_arr.size == 0:
            self.status_panel.appendPlainText(
                f"[{ts}] No valid pitch for /{vowel}/ — retrying"
            )
            self.session.increment_retry(vowel)
            self._capture_buffer.clear()
            self.phase_timer.start(1000)
            return

        # Compute medians
        f1_med = float(np.median(f1_arr))
        f2_med = float(np.median(f2_arr))
        f0_med = float(np.median(f0_arr))

        # Frame count
        frame_count = len(buffer_snapshot)

        # Clear buffer
        self._capture_buffer.clear()

        # Apply result
        self._apply_capture_result(
            vowel,
            f1_med,
            f2_med,
            f0_med,
            frame_count=frame_count,
            timestamp=ts,
        )

        self.phase_timer.start(1000)

    def _apply_capture_result(self, vowel, f1, f2, f0, frame_count=0, timestamp=""):
        accepted, should_retry, msg = self.session.handle_result(
            vowel,
            f1, f2, f0,
            confidence=1.0,
            stability=0.0,
        )

        # Status panel always gets the session message
        self.status_panel.appendPlainText(f"[{timestamp}] {msg}")
        color = vowel_color_for(vowel)

        # Format aligned monospace block
        html_line = (
            f'<span style="color:{color}; font-family:monospace;">'
            f'[{timestamp}] /{vowel}/  '
            f'F1={f1:6.1f}   F2={f2:6.1f}   F0={f0:6.1f}   '
            f'({frame_count} frames)'
            f'</span>'
        )

        if accepted:
            # Append to captures panel
            self.capture_panel.appendHtml(html_line)

            # Store durable anchor
            self._vowel_anchors[vowel] = (f1, f2)
            # NEW: recompute limits dynamically
            self._compute_vowel_axis_limits()
            self._redraw_vowel_anchors()
            self.canvas.draw_idle()

            self.canvas.draw_idle()

            # Reset F0 lock and advance
            self.state.advance()
            return True

        if should_retry:
            self.status_panel.appendPlainText(
                f"[{timestamp}] Retrying /{vowel}/ — please sing clearly"
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
