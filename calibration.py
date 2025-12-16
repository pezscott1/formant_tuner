#!/usr/bin/env python3
import sys
import os
import json
import numpy as np
import time
from time import monotonic
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QApplication,
    QDialog, QLineEdit, QComboBox, QPushButton, QPlainTextEdit, QLabel, QMessageBox
)
from PyQt5.QtCore import QTimer,  pyqtSignal
from PyQt5.QtGui import QTextOption
from formant_utils import (
    safe_spectrogram, get_expected_formants, is_plausible_formants,
    set_active_profile, estimate_formants_lpc, normalize_profile_for_save, is_plausible_pitch, PROFILES_DIR
)
from mic_analyzer import MicAnalyzer, results_queue
import librosa
import traceback
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import logging
mpl.rcParams['axes.formatter.useoffset'] = False

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Single-thread executor for heavy computations (spectrogram, LPC)
_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _extract_audio_array(item):
    """Return a 1D float numpy array from various item shapes (ndarray, list, dict wrappers)."""
    if item is None:
        return None
    if isinstance(item, np.ndarray):
        try:
            return item.astype(float).flatten()
        except Exception:
            return None
    if isinstance(item, (list, tuple)):
        try:
            return np.atleast_1d(item).astype(float).flatten()
        except Exception:
            return None
    if isinstance(item, dict):
        # common keys used by producers
        for key in ("data", "audio", "frame", "samples", "chunk", "segment"):
            if key in item:
                try:
                    return np.atleast_1d(item[key]).astype(float).flatten()
                except Exception:
                    return None
        # fallback: first ndarray-like value
        for v in item.values():
            if isinstance(v, np.ndarray):
                return v.astype(float).flatten()
            if isinstance(v, (list, tuple)):
                try:
                    return np.atleast_1d(v).astype(float).flatten()
                except Exception:
                    continue
        return None
    try:
        return np.atleast_1d(item).astype(float).flatten()
    except Exception:
        return None


class ProfileDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setup Profile")
        layout = QVBoxLayout(self)
        self.resize(600, 300)
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("Enter profile name")
        layout.addWidget(self.name_edit)

        self.voice_combo = QComboBox(self)
        self.voice_combo.addItems(["bass", "baritone", "tenor", "mezzo", "soprano"])
        layout.addWidget(self.voice_combo)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

    def get_values(self):
        return self.name_edit.text().strip(), self.voice_combo.currentText()


class CalibrationWindow(QMainWindow):
    # Signal to deliver worker results safely to the UI thread
    result_ready = pyqtSignal(object)

    def __init__(self, analyzer, profile_name, voice_type, mic=None):
        super().__init__()

        # State
        # Connect worker -> UI signal
        self.result_ready.connect(lambda res: self._on_result_ready(res))
        self._compute_in_flight = False
        self.analyzer = analyzer
        self.profile_name = profile_name
        self.voice_type = voice_type
        self.capture_text = ""
        self._vowel_scatters = {} 
        self._vowel_colors = {
            "i": "red", "e": "blue", "a": "green", "o": "purple", "u": "orange"
        }
        self.vowels = ["i", "e", "a", "o", "u"]
        self.retries_map = {v: 0 for v in self.vowels}
        self.current_index = 0
        self.capture_buffer = []
        self.results = {}
        self.phase_deadline = monotonic()
        self._submitted_index = None  # track which vowel index has an active submission

        # Polling / drawing throttles and buffers
        self._last_draw = 0.0
        self._min_draw_interval = 0.18  # seconds between redraws
        self._pending_frames = []  # small in-memory buffer for recent frames
        self._hb_last = time.time()
        self.capture_start_time = None
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Two text panels side by side
        text_row = QHBoxLayout()
        self.status_panel = QPlainTextEdit()
        self.status_panel.setReadOnly(True)
        self.status_panel.setFixedHeight(50)
        self.status_panel.setStyleSheet("color: darkblue; font-size: 11pt; font-family: Consolas; font-weight: bold;")
        text_row.addWidget(self.status_panel, stretch=1)

        self.capture_panel = QPlainTextEdit()
        self.capture_panel.setWordWrapMode(QTextOption.NoWrap)
        self.capture_panel.setReadOnly(True)
        self.capture_panel.setStyleSheet("color: darkgreen; font-size: 11pt; font-family: Consolas; font-weight: bold;")
        layout.addWidget(self.capture_panel)

        layout.addLayout(text_row)

        # Create figure and canvas
        self.fig, (self.ax_spec, self.ax_vowel) = plt.subplots(1, 2, figsize=(12, 6))
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

        # Artists placeholders for reuse
        self._spec_mesh = None
        self._vowel_scatter = None

        # Mic setup: use provided mic or create a dedicated MicAnalyzer for this window
        self._own_mic = False
        if mic is not None:
            self.mic = mic
            self._own_mic = False
        else:
            tol_provider = (lambda: self.tol_slider.value()) if hasattr(self, "tol_slider") else (lambda: 50)
            pitch_provider = (lambda: self.pitch_slider.value()) if hasattr(self, "pitch_slider") else (lambda: 261)

            # Create per-window MicAnalyzer (must accept these kwargs)
            self.mic = MicAnalyzer(
                vowel_provider=lambda: self.current_vowel_name,
                tol_provider=tol_provider,
                pitch_provider=pitch_provider,
                sample_rate=44100,
                frame_ms=40,
                analyzer=self.analyzer
            )
            self._own_mic = True

        # Start mic and expose a running flag for diagnostics
        if self.mic:
            try:
                self.mic.start()
                # ensure attribute exists for diagnostics
                setattr(self.mic, "is_running", getattr(self.mic, "is_running", True))
                print("[CAL INIT] mic started id:", id(self.mic), "results_queue id:", id(getattr(self.mic, "results_queue", None)))
            except Exception as e:
                setattr(self.mic, "is_running", False)
                print("[CAL INIT] mic.start() error:", e)
                traceback.print_exc()

        # Timer for UI ticks and polling (start AFTER mic is started)
        self.phase = "prep"
        self.prep_secs = 3
        self.sing_secs_total = 3
        self.capture_timeout = getattr(self, "capture_timeout", 8.0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000)  # 10 Hz tick for UI

        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self.poll_queue)
        self.poll_timer.start(80)  # poll frequently but non-blocking
        self.start_phase("prep", 3)
        self.show()

    # -------------------------
    # Phase control helpers
    # -------------------------
    def start_phase(self, phase, seconds):
        self.phase = phase
        self.phase_deadline = monotonic() + seconds
        # Use 1s ticks for user-facing countdowns; shorter only for background/finished phases if needed
        if hasattr(self, "timer") and self.timer is not None:
            self.timer.setInterval(1000 if phase in ("prep", "sing", "capture") else 250)

    def remaining_secs(self):
        return max(0, int(round(self.phase_deadline - monotonic())))

    @property
    def current_vowel_name(self):
        if 0 <= self.current_index < len(self.vowels):
            return self.vowels[self.current_index]
        return None

    def _on_result_ready(self, res):
        """Slot called on the UI thread when worker finishes."""
        try:
            # res is expected to be (freqs, times, S, f1, f2, f0)
            freqs, times, S, f1, f2, f0 = res
        except Exception:
            # defensive: if res is malformed, ignore and log
            print("[_on_result_ready] malformed result:", type(res))
            return

        # debug print to confirm UI thread received the result
        print(f"[apply_compute_result] called idx={self.current_index} f1={f1} f2={f2} f0={f0} "
              f"freqs_len={len(freqs) if freqs is not None else 'None'} S_shape={getattr(S,'shape',None)}")

        # call the existing UI update
        try:
            self._apply_compute_result(freqs, times, S, f1, f2, f0)
        except Exception as e:
            print("[_on_result_ready] _apply_compute_result error:", e)
            traceback.print_exc()

    # -------------------------
    # Main UI tick
    # -------------------------
    def tick(self):
        if getattr(self, "_closing", False):
            return

        # If we've finished all vowels, ensure we finish cleanly
        if self.current_index >= len(self.vowels):
            if self.phase != "finished":
                self.phase = "finished"
                self.status_panel.appendPlainText("Calibration complete!")
            return
        # Heartbeat
        now = time.time()
        if now - self._hb_last > 1.0:
            # quieter heartbeat; use debug logging
            logger = getattr(self, "_logger", None)
            if logger is None:
                import logging as _logging
                logger = _logging.getLogger(__name__)
                self._logger = logger
            logger.debug("heartbeat phase=%s idx=%d time=%s", self.phase, self.current_index, time.strftime('%H:%M:%S'))
            self._hb_last = now

        # Prep
        if self.phase == "prep":
            if self.prep_secs == 3:
                try:
                    self.status_panel.clear()
                except Exception:
                    pass

            if self.prep_secs > 0:
                try:
                    self.status_panel.appendPlainText(
                        f"Prepare: Sing /{self.vowels[self.current_index]}/ in {self.prep_secs}…"
                    )
                except Exception:
                    pass
                self.prep_secs -= 1
            else:
                self.phase = "sing"
                self.sing_secs = 2
                try:
                    self.status_panel.appendPlainText(
                        f"Sing /{self.vowels[self.current_index]}/ – remaining {self.sing_secs}s"
                    )
                except Exception:
                    pass

        # Sing
        elif self.phase == "sing":
            if self.sing_secs > 0:
                try:
                    self.status_panel.appendPlainText(
                        f"Sing /{self.vowels[self.current_index]}/ – remaining {self.sing_secs}s"
                    )
                except Exception:
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
                except Exception:
                    pass

        # Capture
        elif self.phase == "capture":
            if self.capture_secs > 0:
                self.capture_secs -= 1
            else:
                self.process_capture()

    # -------------------------
    # Polling results_queue (non-blocking)
    # -------------------------
    def poll_queue(self):
        """
        Drain from the mic's results_queue into a small in-memory buffer.
        """
        # Prefer the mic instance raw audio queue for calibration segments.
        # Fallback to mic.results_queue only if raw_queue is not present (backwards compatibility).
        queue_to_poll = getattr(self.mic, "raw_queue", None)
        if queue_to_poll is None:
            queue_to_poll = getattr(self.mic, "results_queue", None)
        if queue_to_poll is None:
            from mic_analyzer import results_queue as queue_to_poll

        # debug probe (uncomment if needed)
        # try: print("[poll_queue] qsize:", getattr(queue_to_poll, "qsize", lambda: 'n/a')())
        # except Exception: pass

        MAX_ITEMS_PER_TICK = 20
        drained = 0
        any_drained = False

        while drained < MAX_ITEMS_PER_TICK:
            try:
                item = queue_to_poll.get_nowait()
            except Empty:
                break
            except Exception as e:
                try:
                    item = queue_to_poll.get(timeout=0)
                except Exception:
                    print("[poll_queue] queue get error:", e)
                    break

            try:
                self._pending_frames.append(item)
                if len(self._pending_frames) > 200:
                    self._pending_frames = self._pending_frames[-200:]
            except Exception:
                pass
            drained += 1
            any_drained = True

        if not any_drained and not self._pending_frames:
            return

        # Combine recent chunks
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
                    self._pending_frames = self._pending_frames[:-len(chunks)]
                else:
                    self._pending_frames = []
        except Exception as e:
            print("[poll_queue] combine error:", e)
            traceback.print_exc()
            segment = None

        if segment is not None and self.phase == "capture":
            if not getattr(self, "_compute_in_flight", False) and self._submitted_index != self.current_index:
                try:
                    self._compute_in_flight = True
                    self._submitted_index = self.current_index
                    logger = getattr(self, "_logger", None) or __import__("logging").getLogger(__name__)
                    logger.info("poll_queue: submitting compute for vowel_idx=%d segment_len=%d", self.current_index, len(segment))
                    self.submit_compute(segment)
                except Exception as e:
                    print("[poll_queue] submit_compute error:", e)
                    traceback.print_exc()
                    self._compute_in_flight = False
                    self._submitted_index = None

        now = time.time()
        if now - self._last_draw >= self._min_draw_interval:
            try:
                self.canvas.draw_idle()
            except Exception as e:
                print("[poll_queue] draw_idle error:", e)
            self._last_draw = now

    # -------------------------
    # Capture processing (UI thread only; lightweight)
    # -------------------------
    def process_capture(self):
        # If we have pending frames, wait for worker to be submitted
        if self._pending_frames:
            try:
                self.status_panel.appendPlainText("[process_capture] audio queued for analysis")
            except Exception:
                pass
            return

        if self.current_index >= len(self.vowels):
            try:
                self.status_panel.appendPlainText("Calibration complete!")
            except Exception:
                pass
            QTimer.singleShot(1000, self.finish)
            return

        vowel = self.vowels[self.current_index]
        elapsed = monotonic() - (self.capture_start_time or monotonic())
        if elapsed > self.capture_timeout:
            try:
                self.status_panel.appendPlainText(f"/{vowel}/ capture timed out after {self.capture_timeout}s")
            except Exception:
                pass
            self.current_index += 1
            self._submitted_index = None
            if self.current_index >= len(self.vowels):
                try:
                    self.status_panel.appendPlainText("Calibration complete!")
                except Exception:
                    pass
                QTimer.singleShot(1000, self.finish)
            else:
                self.phase = "prep"
                self.prep_secs = 3
            return
        else:
            try:
                self.status_panel.appendPlainText(f"/{vowel}/ waiting for audio... ({elapsed:.1f}s)")
            except Exception:
                pass
            return

    # -------------------------
    # Offloaded compute: spectrogram + formant estimation
    # -------------------------
    def submit_compute(self, segment):
        sr = getattr(self.mic, "sample_rate", 44100)

        def _compute(segment_local, sr_local):
            try:
                freqs, times, S = safe_spectrogram(segment_local, sr_local)
            except Exception:
                try:
                    S = np.abs(librosa.stft(segment_local, n_fft=1024, hop_length=256)) ** 2
                    freqs = librosa.fft_frequencies(sr=sr_local, n_fft=1024)
                    times = np.arange(S.shape[1]) * (256.0 / sr_local)
                except Exception:
                    freqs, times, S = np.array([0.0]), np.array([0.0]), np.zeros((1, 1))
            try:
                f1, f2, f0 = estimate_formants_lpc(segment_local, sr_local)
            except Exception:
                f1 = f2 = f0 = None
            return freqs, times, S, f1, f2, f0

        future = _EXECUTOR.submit(_compute, segment, sr)

        def _on_done(fut):
            try:
                res = fut.result()
                logger.debug("worker_done result type: %s", type(res))
            except Exception as e:
                print("[submit_compute] worker exception:", e)
                traceback.print_exc()
                self._compute_in_flight = False
                return

            # emit result to UI thread via signal
            try:
                self.result_ready.emit(res)
            except Exception as e:
                print("[submit_compute] emit failed:", e)
                traceback.print_exc()

            # clear in-flight flag so new segments can be submitted
            self._compute_in_flight = False

        future.add_done_callback(_on_done)
    # -------------------------
    # Apply compute result on UI thread (lightweight updates only)
    # -------------------------

    def _apply_compute_result(self, freqs, times, S, f1, f2, f0):
        # clear in-flight flag immediately so new segments can be submitted
        logger.info("apply_compute_result idx=%d f1=%.1f f2=%.1f f0=%s freqs_len=%s S_shape=%s",
            self.current_index,
            (f1 if f1 is not None else float('nan')),
            (f2 if f2 is not None else float('nan')),
            (f0 if f0 is not None else "None"),
            (len(freqs) if freqs is not None else "None"),
            getattr(S, "shape", None))
        self._compute_in_flight = False

        try:
            vowel = self.vowels[self.current_index] if 0 <= self.current_index < len(self.vowels) else None
        except Exception:
            vowel = None

        try:
            self.update_artists(freqs, times, S, f1, f2, vowel)
        except Exception as e:
            print("[apply_compute_result] update_artists error:", e)
            traceback.print_exc()

        try:
            ok_formants = (f1 is not None) and (f2 is not None) and (not np.isnan(f1)) and (not np.isnan(f2))
            ok_pitch = (f0 is not None) and (not np.isnan(f0))
            max_retries = 3
            retries = int(self.retries_map.get(vowel, 0) or 0)

            if ok_formants and vowel is not None:
                self.results[vowel] = (float(f1), float(f2), float(f0) if ok_pitch else None)
                color = self._vowel_colors.get(vowel, "black")
                html_line = f'<span style="color:{color}">/{vowel}/ F1={f1:.1f} Hz, F2={f2:.1f} Hz, F0={f0:.1f} Hz</span>'
                self.capture_panel.appendHtml(html_line)
                self.current_index += 1
                # after you set self.current_index (on accept or after max_retries)
                self._submitted_index = None
            else:
                if retries < max_retries:
                    self.retries_map[vowel] = retries + 1
                    self.status_panel.appendPlainText(f"/{vowel}/ retry {self.retries_map[vowel]} (formants/pitch missing)")
                else:
                    self.status_panel.appendPlainText(f"/{vowel}/ skipped after {max_retries} attempts")
                    self.current_index += 1
                    # after you set self.current_index (on accept or after max_retries)
                    self._submitted_index = None
        except Exception as e:
            print("[apply_compute_result] acceptance logic error:", e)
            traceback.print_exc()

        if self.current_index >= len(self.vowels):
            self.status_panel.appendPlainText("Calibration complete!")
            # schedule finish once, 1 second later
            if self.phase != "finished":
                self.phase = "finished"
                QTimer.singleShot(1000, self.finish)
            return
        else:
            self.phase = "prep"
            self.prep_secs = 3
            self.timer.setInterval(1000)

    # -------------------------
    # Artist update (reuse existing artists)
    # -------------------------
    def update_artists(self, freqs, times, S, f1, f2, vowel):
        # defensive checks
        if freqs is None or times is None or S is None:
            return

        # --- Spectrogram update ---
        mask = freqs <= 4000
        if isinstance(mask, np.ndarray) and mask.sum() == 0:
            mask = np.arange(len(freqs))

        max_time_bins = 200
        step = max(1, S.shape[1] // max_time_bins) if S.shape[1] > 0 else 1
        S_small = S[mask, ::step]
        times_small = times[::step]

        arr_db = 10 * np.log10(S_small + 1e-12)

        if self._spec_mesh is None:
            self.ax_spec.clear()
            try:
                self._spec_mesh = self.ax_spec.pcolormesh(times_small, freqs[mask], arr_db, shading='auto')
            except Exception:
                mean_spec = np.mean(S_small, axis=1) if S_small.size else np.zeros_like(freqs[mask])
                self.ax_spec.plot(freqs[mask], 10 * np.log10(mean_spec + 1e-12))
            self.ax_spec.set_title("Spectrogram")
            self.ax_spec.set_xlabel("Time (s)")
            self.ax_spec.set_ylabel("Frequency (Hz)")
            self.ax_spec.set_ylim(0, 4000)
        else:
            try:
                if hasattr(self._spec_mesh, "set_array"):
                    self._spec_mesh.set_array(arr_db.ravel())
                else:
                    self.ax_spec.clear()
                    self._spec_mesh = self.ax_spec.pcolormesh(times_small, freqs[mask], arr_db, shading='auto')
                self.ax_spec.set_ylim(0, 4000)
            except Exception as e:
                print("[update_artists] spec update error:", e)
                traceback.print_exc()

        # --- Vowel scatter update ---
        if f1 is None or f2 is None or vowel is None:
            # still draw spectrogram but no vowel point
            now = time.time()
            if now - self._last_draw >= self._min_draw_interval:
                try:
                    self.canvas.draw_idle()
                except Exception:
                    pass
                self._last_draw = now
            return

        if vowel not in self._vowel_scatters:
            scatter = self.ax_vowel.scatter([f2], [f1],
                                            c=self._vowel_colors.get(vowel, "black"),
                                            s=70, zorder=4, label=f"/{vowel}/")
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
                scatter = self.ax_vowel.scatter([f2], [f1],
                                                c=self._vowel_colors.get(vowel, "black"),
                                                s=70, zorder=4, label=f"/{vowel}/")
                self._vowel_scatters[vowel] = scatter
                self.ax_vowel.legend(loc="best")

        # --- Redraw throttled ---
        try:
            self.canvas.draw_idle()
        except Exception:
            logger.debug("canvas.draw_idle failed", exc_info=True)

        now = time.time()
        if now - self._last_draw >= self._min_draw_interval:
            try:
                self.canvas.draw_idle()
            except Exception as e:
                print("[update_artists] draw_idle error:", e)
            self._last_draw = now

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
        except Exception:
            pass

        base = self.profile_name
        profile_path = os.path.join(PROFILES_DIR, f"{base}_profile.json")

        profile_dict = normalize_profile_for_save(self.results, retries_map=self.retries_map)

        try:
            with open(profile_path, "w", encoding="utf-8") as fh:
                json.dump(profile_dict, fh, indent=2)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to save profile: {e}")
            return

        set_active_profile(base)

        if hasattr(self, "update_ui"):
            try:
                self.update_ui({
                    "f0": None,
                    "formants": (None, None, None),
                    "vowel_guess": None,
                    "vowel_confidence": 0.0,
                    "vowel_score": 0,
                    "resonance_score": 0,
                    "overall": 0,
                    "fb_f1": None,
                    "fb_f2": None
                })
            except Exception:
                pass

        parent = self.parent()
        if parent:
            if hasattr(parent, "reload_profiles"):
                parent.reload_profiles()
            if hasattr(parent, "apply_selected_profile"):
                parent.apply_selected_profile(base)

        msg = QMessageBox(self)
        msg.setWindowTitle("Calibration Finished")
        msg.setText("Calibration complete! Click OK to close.")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

        self.close()

    def closeEvent(self, event):
        self._closing = True
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            self.poll_timer.stop()
        except Exception:
            pass
        try:
            if getattr(self, "_own_mic", False) and self.mic:
                self.mic.stop()
        except Exception:
            pass
        event.accept()


# Standalone run (for testing this file directly)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CalibrationWindow(analyzer=None, profile_name="user1", voice_type="tenor", mic=None)
    sys.exit(app.exec_())