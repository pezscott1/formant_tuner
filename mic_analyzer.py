# mic_analyzer.py
import numpy as np
import sounddevice as sd
import queue
import logging
import threading
from collections import deque

from formant_utils import (
    estimate_formants_lpc,
    is_plausible_formants,
    directional_feedback,
    robust_guess
)
from voice_analysis import MedianSmoother

# Shared queue for UI updates
results_queue = queue.Queue()

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Replace the existing __init__ body with this (keep the same def header but add results_queue param) ---


class MicAnalyzer:
    def __init__(self, vowel_provider, tol_provider, pitch_provider,
                 sample_rate=44100, frame_ms=40, analyzer=None,
                 processing_window_s=0.25, lpc_win_ms=30, processing_queue_max=6,
                 rms_gate=1e-6, debug=False, results_queue=None):
        self.smoother = MedianSmoother(size=5)
        self.vowel_provider = vowel_provider
        self.tol_provider = tol_provider
        self.pitch_provider = pitch_provider
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.analyzer = analyzer
        self.stream = None

        # Rolling buffer and processing queue
        self.buffer = deque(maxlen=int(self.sample_rate * 3))
        self.processing_queue = queue.Queue(maxsize=int(processing_queue_max))
        self._worker_stop = threading.Event()
        self._worker = None

        # Tunables
        self.processing_window_s = float(processing_window_s)
        self.lpc_win_ms = int(lpc_win_ms)
        self.rms_gate = float(rms_gate)
        self.debug = bool(debug)
        
        # Ensure we always have a queue to publish audio/analysis results to
        module_q = globals().get("results_queue", None)
        if results_queue is not None:
            self.results_queue = results_queue
        elif module_q is not None:
            self.results_queue = module_q
        else:
            self.results_queue = queue.Queue(maxsize=200)

        # Per-instance raw audio queue for consumers that need raw segments (e.g., calibration)
        self.raw_queue = queue.Queue(maxsize=200)

        # Worker thread will be started in start()
        self._worker = None

        # running flag for diagnostics
        self.is_running = False

    # -------------------------
    # Audio callback (fast)
    # -------------------------
    def audio_callback(self, indata, frames, time_info, status):
        try:
            mono = np.asarray(indata[:, 0], dtype=float).flatten()
            # append to deque
            self.buffer.extend(mono.tolist())

            # cheap energy gate
            if mono.size == 0:
                return
            if np.mean(mono ** 2) < self.rms_gate:
                return

            # prepare a short segment (tail of buffer)
            win_len = int(self.sample_rate * self.processing_window_s)
            if win_len < 1:
                return
            if len(self.buffer) < win_len:
                segment = np.array(list(self.buffer), dtype=float)
            else:
                # efficient tail extraction
                tail = []
                it = iter(self.buffer)
                skip = len(self.buffer) - win_len
                for _ in range(skip):
                    next(it, None)
                for v in it:
                    tail.append(v)
                segment = np.array(tail, dtype=float)

            # enqueue for processing without blocking
            try:
                # also publish raw audio for any consumer that wants it (calibrator)
                try:
                    self.raw_queue.put_nowait(segment)
                except queue.Full:
                    # drop oldest then try once to keep moving
                    try:
                        _ = self.raw_queue.get_nowait()
                        self.raw_queue.put_nowait(segment)
                    except Exception:
                        pass

                self.processing_queue.put_nowait(segment)

                # less noisy: use logger.debug so you can enable/disable via logging level
                try:
                    q = getattr(self, "results_queue", None) or globals().get("results_queue", None)
                    logger.debug("MicAnalyzer pushed segment len=%d proc_q=%s ui_q=%s raw_q=%s",
                                 segment.size, getattr(self.processing_queue, "qsize", lambda: 'n/a')(),
                                 getattr(q, "qsize", lambda: 'n/a')(), getattr(self, "raw_queue", None)
                                 and getattr(self.raw_queue, "qsize", lambda: 'n/a')())
                except Exception:
                    pass
            except queue.Full:
                # drop frame if worker is busy
                pass
        except Exception:
            logger.exception("MicAnalyzer audio callback failed")

    # -------------------------
    # Worker thread
    # -------------------------
    def _processing_worker(self):
        while not self._worker_stop.is_set():
            try:
                segment = self.processing_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                # Defensive call: estimate_formants_lpc may return variable shapes
                res = estimate_formants_lpc(segment, self.sample_rate, order=None,
                                            win_len_ms=self.lpc_win_ms, debug=self.debug)

                # Normalize result into f1, f2, f0 (f0 used as f3 in some callers)
                if res is None:
                    f1 = f2 = f0 = None
                elif isinstance(res, (tuple, list)):
                    if len(res) >= 3:
                        f1, f2, f0 = res[:3]
                    elif len(res) == 2:
                        f1, f2, f0 = res[0], res[1], None
                    else:
                        f1 = f2 = f0 = None
                else:
                    f1 = f2 = f0 = None

                # Smooth values (MedianSmoother returns None for missing)
                f1_s, f2_s, f0_s = self.smoother.update(f1, f2, f0)

                # Plausibility gating
                voice_type = getattr(self.analyzer, "voice_type", "bass")
                ok, reason = is_plausible_formants(f1_s, f2_s, voice_type)
                if not ok:
                    f1_s, f2_s = None, None

                # Build user_forms mapping
                user_forms = {}
                try:
                    for v, vals in (getattr(self.analyzer, "user_formants", {}) or {}).items():
                        if isinstance(vals, (tuple, list)) and len(vals) >= 2:
                            user_forms[v] = {"f1": vals[0], "f2": vals[1]}
                        elif isinstance(vals, dict):
                            user_forms[v] = vals
                except Exception:
                    logger.exception("Failed to build user_forms mapping")

                # Directional feedback
                current_vowel = self.vowel_provider() if callable(self.vowel_provider) else None
                tolerance = self.tol_provider() if callable(self.tol_provider) else 50
                fb_f1, fb_f2 = directional_feedback((f1_s, f2_s, f0_s), user_forms, current_vowel, tolerance)

                # Vowel guess uses only formants
                try:
                    guessed, conf, second = robust_guess((f1_s, f2_s), voice_type=voice_type)
                except Exception:
                    guessed, conf, second = None, 0.0, None

                # Compose status dict and post to UI queue
                status = {
                    "f0": float(f0_s) if f0_s is not None else None,
                    "formants": (float(f1_s) if f1_s is not None else None,
                                 float(f2_s) if f2_s is not None else None,
                                 float(f0_s) if f0_s is not None else None),
                    "vowel_guess": guessed,
                    "vowel_confidence": float(conf) if conf is not None else 0.0,
                    "vowel_score": 0,
                    "resonance_score": 0,
                    "overall": 0,
                    "fb_f1": fb_f1,
                    "fb_f2": fb_f2,
                }

                try:
                    results_queue.put(status, timeout=0.1)
                except queue.Full:
                    pass

            except Exception:
                logger.exception("Processing worker failed while handling a segment")

            finally:
                try:
                    self.processing_queue.task_done()
                except Exception:
                    pass

    # -------------------------
    # Public control
    # -------------------------
    def start(self):
        if self._worker is None or not self._worker.is_alive():
            self._worker_stop.clear()
            self._worker = threading.Thread(target=self._processing_worker, daemon=True)
            self._worker.start()
            logger.info("MicAnalyzer worker started")

        if self.stream is None:
            try:
                blocksize = max(64, int(self.sample_rate * self.frame_ms / 1000))
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    blocksize=blocksize,
                    channels=1,
                    dtype='float32',
                    callback=self.audio_callback
                )
                self.stream.start()
                self.is_running = True
                logger.info("MicAnalyzer audio stream started at %d Hz blocksize %d", self.sample_rate, blocksize)
            except Exception:
                logger.exception("Failed to start audio stream")
                # ensure worker is stopped if stream fails
                self.stop()

    def stop(self, event=None):
        try:
            if self.stream is not None:
                try:
                    if getattr(self.stream, "active", False):
                        self.stream.stop()
                except Exception:
                    pass
                try:
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
                logger.info("MicAnalyzer audio stream stopped")
        except Exception:
            logger.exception("Error stopping audio stream")

        try:
            if self._worker is not None:
                self._worker_stop.set()
                try:
                    self.processing_queue.put_nowait(np.zeros(1, dtype=float))
                except Exception:
                    pass
                self._worker.join(timeout=1.0)
                self._worker = None
                logger.info("MicAnalyzer worker stopped")
        except Exception:
            logger.exception("Error stopping worker")
        finally:
            self.is_running = False
