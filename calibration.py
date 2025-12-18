#!/usr/bin/env python3
import os
import json
import time
import traceback
from time import monotonic
from concurrent.futures import ThreadPoolExecutor
import logging
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QApplication,
    QDialog,
    QLineEdit,
    QComboBox,
    QPushButton,
    QPlainTextEdit,
    QMessageBox,
)
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtGui import QTextOption

from formant_utils import (
    safe_spectrogram,
    estimate_formants_lpc,
    normalize_profile_for_save,
    PROFILES_DIR,
)
from mic_analyzer import MicAnalyzer

mpl.rcParams["axes.formatter.useoffset"] = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _extract_audio_array(item):
    """Return a 1D float numpy array from ndarray, list, or dict wrappers."""
    if item is None:
        return None
    try:
        if isinstance(item, np.ndarray):
            return item.astype(float).flatten()
        if isinstance(item, (list, tuple)):
            return np.atleast_1d(item).astype(float).flatten()
        if isinstance(item, dict):
            for key in (
                "data",
                "audio",
                "frame",
                "samples",
                "chunk",
                "segment",
            ):
                if key in item:
                    return np.atleast_1d(item[key]).astype(float).flatten()
            for v in item.values():
                if isinstance(v, np.ndarray):
                    return v.astype(float).flatten()
                if isinstance(v, (list, tuple)):
                    return np.atleast_1d(v).astype(float).flatten()
        return np.atleast_1d(item).astype(float).flatten()
    except Exception:  # noqa: E722
        return None


class ProfileDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setup Profile")
        self.resize(600, 300)

        layout = QVBoxLayout(self)
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("Enter profile name")
        layout.addWidget(self.name_edit)

        self.voice_combo = QComboBox(self)
        self.voice_combo.addItems(
            ["bass", "baritone", "tenor", "mezzo", "soprano"]
        )
        layout.addWidget(self.voice_combo)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        ok_btn.clicked.connect(self.accept)  # type: ignore
        cancel_btn.clicked.connect(self.reject)  # type: ignore

    def get_values(self):
        return self.name_edit.text().strip(), self.voice_combo.currentText()


class CalibrationWindow(QMainWindow):
    """Calibration workflow window."""
    result_ready = pyqtSignal(object)
    profile_calibrated = pyqtSignal(str)

    def __init__(self, analyzer, profile_name, voice_type, mic=None):
        super().__init__()
        self.analyzer = analyzer
        self.profile_name = profile_name
        self.voice_type = voice_type

        # State
        self._compute_in_flight = False
        self._submitted_index = None
        self._finished = False
        self._logger = logger
        self._vowel_scatters = {}
        self._spec_mesh = None
        self._vowel_colors = {
            "i": "red",
            "e": "blue",
            "a": "green",
            "o": "purple",
            "u": "orange",
        }
        self.vowels = ["i", "e", "a", "o", "u"]
        self.retries_map = {v: 0 for v in self.vowels}
        self.current_index = 0
        self.results = {}
        self.capture_buffer = []
        self._pending_frames = []
        self._last_draw = 0.0
        self._min_draw_interval = 0.18
        self._hb_last = time.time()
        self.capture_start_time = None
        self.phase_deadline = 0.0
        self.sing_secs = 0
        self.capture_secs = 0
        self._closing = False
        self.result_ready.connect(self._on_result_ready)  # type: ignore
        # UI setup
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        text_row = QHBoxLayout()
        self.status_panel = QPlainTextEdit()
        self.status_panel.setReadOnly(True)
        self.status_panel.setFixedHeight(50)
        self.status_panel.setStyleSheet(
            "QFrame { background-color: #e6f0ff; "
            "border: 1px solid #99c; border-radius: 6px; }"
        )

        self.status_panel.setStyleSheet(
            "color: darkblue; font-size: 11pt; "
            "font-family: Consolas; font-weight: bold;")
        text_row.addWidget(self.status_panel, stretch=1)

        self.capture_panel = QPlainTextEdit()
        self.capture_panel.setWordWrapMode(QTextOption.NoWrap)
        self.capture_panel.setReadOnly(True)
        self.capture_panel.setStyleSheet(
            "QFrame { background-color: #f2f2f2; "
            "border: 1px solid #ccc; border-radius: 6px; }"
        )
        self.capture_panel.setStyleSheet(
            "color: darkgreen; font-size: 11pt; "
            "font-family: Consolas; font-weight: bold;")
        layout.addLayout(text_row)
        layout.addWidget(self.capture_panel)

        self.fig, (self.ax_spec, self.ax_vowel) = plt.subplots(
            1, 2, figsize=(12, 6)
        )
        for ax in (self.ax_spec, self.ax_vowel):
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, stretch=1)

        screen = QApplication.primaryScreen().availableGeometry()
        w = int(screen.width() * 0.75)
        h = int(screen.height() * 0.75)
        self.resize(w, h)
        self.move(screen.center().x() - w // 2, screen.center().y() - h // 2)

        # Mic setup
        self._own_mic = False
        if mic is not None:
            self.mic = mic
        else:
            self.mic = MicAnalyzer(
                vowel_provider=lambda: self.current_vowel_name,
                tol_provider=lambda: 50,
                pitch_provider=lambda: 261,
                sample_rate=44100,
                frame_ms=40,
                analyzer=self.analyzer,
            )
            self._own_mic = True

        try:
            self.mic.start()
            self.mic.is_running = True
        except Exception:  # noqa: E722
            self.mic.is_running = False
            traceback.print_exc()

        # Timers
        self.phase = "prep"
        self.prep_secs = 3
        self.sing_secs_total = 3
        self.capture_timeout = 8.0

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)  # type: ignore
        self.timer.start(1000)

        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self.poll_queue)  # type: ignore
        self.poll_timer.start(80)

        self.start_phase("prep", 3)
        self.show()

    @property
    def current_vowel_name(self):
        if 0 <= self.current_index < len(self.vowels):
            return self.vowels[self.current_index]
        return None

    def start_phase(self, phase, seconds):
        self.phase = phase
        self.phase_deadline = monotonic() + seconds
        if self.timer is not None:
            self.timer.setInterval(
                1000 if phase in ("prep", "sing", "capture") else 250
            )

    def submit_compute(self, segment):
        sr = getattr(self.mic, "sample_rate", 44100)

        def _compute(segment_local, sr_local):
            try:
                freqs, times, S = safe_spectrogram(segment_local, sr_local)
            except Exception:  # noqa: E722
                freqs, times, S = (
                    np.array([0.0]),
                    np.array([0.0]),
                    np.zeros((1, 1)),
                )
            try:
                f1, f2, f3 = estimate_formants_lpc(segment_local, sr_local)
            except Exception:  # noqa: E722
                f1 = f2 = f3 = None
            return freqs, times, S, f1, f2, f3

        future = _EXECUTOR.submit(_compute, segment, sr)

        def _on_done(fut):
            try:
                res = fut.result()
            except Exception:  # noqa: E722
                traceback.print_exc()
                self._compute_in_flight = False
                return

            # Emit result to UI thread
            try:
                self.result_ready.emit(res)  # type: ignore
            except Exception:  # noqa: E722
                traceback.print_exc()

            # Clear in-flight flag so new segments can be submitted
            self._compute_in_flight = False

        future.add_done_callback(_on_done)

    # -------------------------
    # Result handling (UI thread)
    # -------------------------

    def _on_result_ready(self, result):
        self._compute_in_flight = False
        try:
            freqs, times, S, f1, f2, f3 = result
            self._apply_compute_result(freqs, times, S, f1, f2, f3)
        except Exception:  # noqa: E722
            print("[_on_result_ready] malformed result:", type(result))
            traceback.print_exc()

    def _apply_compute_result(self, freqs, times, s, f1, f2, f0):
        self._compute_in_flight = False

        vowel = self.current_vowel_name
        try:
            self.update_artists(freqs, times, s, f1, f2, vowel)
        except Exception:  # noqa: E722
            traceback.print_exc()

        try:
            ok_formants = (
                (f1 is not None)
                and (f2 is not None)
                and (not np.isnan(f1))
                and (not np.isnan(f2))
            )
            ok_pitch = (f0 is not None) and (not np.isnan(f0))
            max_retries = 3
            retries = int(self.retries_map.get(vowel, 0) or 0)

            if ok_formants and vowel is not None:
                self.results[vowel] = (
                    float(f1),
                    float(f2),
                    float(f0) if ok_pitch else None,
                )
                color = self._vowel_colors.get(vowel, "black")
                html_line = (
                    f'<span style="color:{color}">/{vowel}/ '
                    f"F1={f1:.1f} Hz, F2={f2:.1f} Hz, F3={f0 or 0:.1f} Hz</span>"
                )
                self.capture_panel.appendHtml(html_line)
                self.current_index += 1
                self._submitted_index = None
            else:
                if retries < max_retries:
                    vowel = self.current_vowel_name
                    if vowel is None:
                        return  # or skip retries update

                    retries = int(self.retries_map.get(vowel, 0) or 0)
                    self.retries_map[vowel] = retries + 1
                    self.status_panel.appendPlainText(
                        f"/{vowel}/ retry {self.retries_map[vowel]}"
                        f" (formants/pitch missing)"
                    )
                else:
                    self.status_panel.appendPlainText(
                        f"/{vowel}/ skipped after {max_retries} attempts"
                    )
                    self.current_index += 1
                    self._submitted_index = None
        except Exception:  # noqa: E722
            traceback.print_exc()

        if self.current_index >= len(self.vowels):
            self.status_panel.appendPlainText("Calibration complete!")
            if self.phase != "finished":
                self.phase = "finished"
                QTimer.singleShot(1000, self.finish)
            return

        self.phase = "prep"
        self.prep_secs = 3
        self.timer.setInterval(1000)

    # -------------------------
    # Artist update
    # -------------------------
    def update_artists(self, freqs, times, s, f1, f2, vowel):
        if freqs is None or times is None or s is None:
            return

        # Spectrogram (<= 4 kHz)
        mask = freqs <= 4000
        if isinstance(mask, np.ndarray) and mask.sum() < 2:
            # fallback if too few bins
            mask = np.arange(len(freqs))

        max_time_bins = 200
        step = max(1, s.shape[1] // max_time_bins) if s.shape[1] > 0 else 1
        S_small = s[mask, ::step]
        times_small = times[::step]
        arr_db = 10 * np.log10(S_small + 1e-12)

        ny, nx = arr_db.shape

        if self._spec_mesh is None:
            self.ax_spec.clear()
            try:
                # Use shading="auto" to avoid off-by-one issues
                self._spec_mesh = self.ax_spec.pcolormesh(
                    times_small, freqs[mask], arr_db, shading="auto"
                )
            except Exception:
                mean_spec = np.mean(S_small, axis=1)\
                    if S_small.size else np.zeros_like(freqs[mask])
                self.ax_spec.plot(freqs[mask], 10 * np.log10(mean_spec + 1e-12))
        else:
            try:
                expected_size = ny * nx
                current_size = self._spec_mesh.get_array().size
                if current_size != expected_size:
                    # Recreate mesh if dimensions changed
                    self.ax_spec.clear()
                    self._spec_mesh = self.ax_spec.pcolormesh(
                        times_small, freqs[mask], arr_db, shading="auto"
                    )
                else:
                    # Update with full array, no slicing
                    self._spec_mesh.set_array(arr_db.ravel())
            except Exception:
                traceback.print_exc()

        self.ax_spec.set_title("Spectrogram")
        self.ax_spec.set_xlabel("Time (s)")
        self.ax_spec.set_ylabel("Frequency (Hz)")
        self.ax_spec.set_ylim(0, 4000)
        self.canvas.draw_idle()

        # Vowel scatter
        if f1 is None or f2 is None or vowel is None:
            now = time.time()
            if now - self._last_draw >= self._min_draw_interval:
                try:
                    self.canvas.draw_idle()
                except Exception:
                    pass
                self._last_draw = now
            return

        if vowel not in self._vowel_scatters:
            scatter = self.ax_vowel.scatter(
                [f2],
                [f1],
                c=self._vowel_colors.get(vowel, "black"),
                s=70,
                zorder=4,
                label=f"/{vowel}/",
            )
            self._vowel_scatters[vowel] = scatter
            self.ax_vowel.set_title("Vowel Space")
            self.ax_vowel.set_xlabel("F2 (Hz)")
            self.ax_vowel.set_ylabel("F1 (Hz)")
            try:
                self.ax_vowel.set_xlim(4000, 0)
                self.ax_vowel.set_ylim(1200, 0)
            except Exception:
                pass
            self.ax_vowel.legend(loc="best")
        else:
            try:
                self._vowel_scatters[vowel].set_offsets(np.column_stack(([f2], [f1])))
            except Exception:
                try:
                    self._vowel_scatters[vowel].remove()
                except Exception:
                    pass
                scatter = self.ax_vowel.scatter(
                    [f2],
                    [f1],
                    c=self._vowel_colors.get(vowel, "black"),
                    s=70,
                    zorder=4,
                    label=f"/{vowel}/",
                )
                self._vowel_scatters[vowel] = scatter
                self.ax_vowel.legend(loc="best")

        try:
            self.canvas.draw_idle()
        except Exception:
            pass

        now = time.time()
        if now - self._last_draw >= self._min_draw_interval:
            try:
                self.canvas.draw_idle()
            except Exception:
                pass
            self._last_draw = now

    # -------------------------
    # Tick (prep/sing/capture)
    # -------------------------
    def tick(self):
        if getattr(self, "_closing", False):
            return

        if self.current_index >= len(self.vowels):
            if self.phase != "finished":
                self.phase = "finished"
                self.status_panel.appendPlainText("Calibration complete!")
            return

        now = time.time()
        if now - self._hb_last > 1.0:
            self._logger.debug(
                "heartbeat phase=%s idx=%d time=%s",
                self.phase,
                self.current_index,
                time.strftime("%H:%M:%S"),
            )
            self._hb_last = now

        if self.phase == "prep":
            if self.prep_secs == 3:
                try:
                    self.status_panel.clear()
                except Exception:  # noqa: E722
                    pass

            if self.prep_secs > 0:
                try:
                    self.status_panel.appendPlainText(
                        f"Prepare: Sing /{self.vowels[self.current_index]}/ "
                        f"in {self.prep_secs}…"
                    )
                except Exception:  # noqa: E722
                    pass
                self.prep_secs -= 1
            else:
                self.phase = "sing"
                self.sing_secs = 2
                try:
                    self.status_panel.appendPlainText(
                        f"Sing /{self.vowels[self.current_index]}"
                        f"/ – remaining {self.sing_secs}s"
                    )
                except Exception:  # noqa: E722
                    pass

        elif self.phase == "sing":
            if self.sing_secs > 0:
                try:
                    self.status_panel.appendPlainText(
                        f"Sing /{self.vowels[self.current_index]}"
                        f"/ – remaining {self.sing_secs}s"
                    )
                except Exception:  # noqa: E722
                    pass
                self.sing_secs -= 1
            else:
                self.phase = "capture"
                self.capture_secs = 1
                self.capture_start_time = monotonic()
                try:
                    self.status_panel.appendPlainText(
                        f"Capturing /{self.vowels[self.current_index]}/…"
                    )
                except Exception:  # noqa: E722
                    pass

        elif self.phase == "capture":
            if self.capture_secs > 0:
                self.capture_secs -= 1
            else:
                self.process_capture()

    # -------------------------
    # Queue polling
    # -------------------------
    def poll_queue(self):
        queue_to_poll = getattr(self.mic, "raw_queue", None)
        if queue_to_poll is None:
            queue_to_poll = getattr(self.mic, "results_queue", None)

        MAX_ITEMS_PER_TICK = 20
        drained = 0
        any_drained = False

        while drained < MAX_ITEMS_PER_TICK:
            try:
                item = queue_to_poll.get(timeout=0)
            except Exception:  # noqa: E722
                break

            try:
                self._pending_frames.append(item)
                if len(self._pending_frames) > 200:
                    self._pending_frames = self._pending_frames[-200:]
            except Exception:  # noqa: E722
                pass
            drained += 1
            any_drained = True

        if not any_drained and not self._pending_frames:
            return

        segment = None
        try:
            chunks = self._pending_frames[-8:]
            arrays = []
            for x in chunks:
                arr = _extract_audio_array(x)
                if arr is not None and arr.size > 0:
                    arrays.append(arr)
            if arrays:
                segment = np.concatenate(arrays)
                if len(self._pending_frames) > len(chunks):
                    self._pending_frames = self._pending_frames[: -len(chunks)]
                else:
                    self._pending_frames = []
        except Exception:  # noqa: E722
            traceback.print_exc()
            segment = None

        if segment is not None and self.phase == "capture":
            if (not getattr(self, "_compute_in_flight", False)) and (
                self._submitted_index != self.current_index
            ):
                try:
                    self._compute_in_flight = True
                    self._submitted_index = self.current_index
                    self.submit_compute(segment)
                except Exception:  # noqa: E722
                    traceback.print_exc()
                    self._compute_in_flight = False
                    self._submitted_index = None

        now = time.time()
        if now - self._last_draw >= self._min_draw_interval:
            try:
                self.canvas.draw_idle()
            except Exception:  # noqa: E722
                pass
            self._last_draw = now

    # -------------------------
    # Capture processing
    # -------------------------
    def process_capture(self):
        if self._pending_frames:
            try:
                self.status_panel.appendPlainText(
                    "[process_capture] audio queued for analysis"
                )
            except Exception:  # noqa: E722
                pass
            return

        if self.current_index >= len(self.vowels):
            try:
                self.status_panel.appendPlainText("Calibration complete!")
            except Exception:  # noqa: E722
                pass
            QTimer.singleShot(1000, self.finish)
            return

        vowel = self.vowels[self.current_index]
        elapsed = monotonic() - (self.capture_start_time or monotonic())
        if elapsed > self.capture_timeout:
            try:
                self.status_panel.appendPlainText(
                    f"/{vowel}/ capture timed out after {self.capture_timeout}s"
                )
            except Exception:  # noqa: E722
                pass
            self.current_index += 1
            self._submitted_index = None
            if self.current_index >= len(self.vowels):
                try:
                    self.status_panel.appendPlainText("Calibration complete!")
                except Exception:  # noqa: E722
                    pass
                QTimer.singleShot(1000, self.finish)
            else:
                self.phase = "prep"
                self.prep_secs = 3
            return

        try:
            self.status_panel.appendPlainText(
                f"/{vowel}/ waiting for audio... ({elapsed:.1f}s)"
            )
        except Exception:  # noqa: E722
            pass

    # -------------------------
    # Finish and cleanup
    # -------------------------
    def finish(self):
        if getattr(self, "_finished", False):
            return
        self._finished = True

        try:
            if getattr(self, "_own_mic", False) and self.mic:
                self.mic.stop()
        except Exception:  # noqa: E722
            pass

        base_name = f"{self.profile_name}_{self.voice_type}"
        profile_path = os.path.join(PROFILES_DIR, f"{base_name}_profile.json")

        profile_dict = normalize_profile_for_save(
            self.results, retries_map=self.retries_map)
        try:
            with open(profile_path, "w", encoding="utf-8") as fh:
                json.dump(profile_dict, fh, indent=2)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to save profile: {e}")
            return

        # Tell parent to update active profile
        parent = self.parent()
        if parent:
            try:
                if hasattr(parent, "reload_profiles"):
                    parent.reload_profiles()
                if hasattr(parent, "set_active_profile"):
                    parent.set_active_profile(base_name)
            except Exception:  # noqa: E722
                traceback.print_exc()
        try:
            self.profile_calibrated.emit(base_name)   # type:ignore
        except Exception:  # noqa: E722
            pass
        self.close()

    def closeEvent(self, event):
        self._closing = True
        try:
            self.timer.stop()
        except Exception:  # noqa: E722
            pass
        try:
            self.poll_timer.stop()
        except Exception:  # noqa: E722
            pass
        try:
            if getattr(self, "_own_mic", False) and self.mic:
                self.mic.stop()
        except Exception:  # noqa: E722
            pass
        event.accept()


# Standalone run (for testing this file directly)
if __name__ == "__main__":
    app = QApplication([])
    win = CalibrationWindow(
        analyzer=None, profile_name="user1", voice_type="tenor", mic=None
    )
    sys.exit(app.exec_())
