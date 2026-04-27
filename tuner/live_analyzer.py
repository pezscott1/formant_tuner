# tuner/live_analyzer.py
import logging
import queue
import threading
import numpy as np

logger = logging.getLogger(__name__)


class LiveAnalyzer:
    """
    Pass raw engine output through:
      - pitch smoothing (confidence-aware)
      - formant smoothing (confidence-aware)
      - vowel label smoothing (confidence-aware)
      - profile-based scoring
      - stability tracking

    Runtime responsibilities:
      - accept audio frames from audio callback
      - call engine on frames in a worker thread
      - expose processed frames via a queue for the UI
    """

    def __init__(self, engine, pitch_smoother,
                 formant_smoother, label_smoother, sample_rate=48000):
        self.engine = engine
        self.pitch_smoother = pitch_smoother
        self.pitch_smoother.current = None
        self.formant_smoother = formant_smoother
        self.label_smoother = label_smoother
        self.sample_rate = sample_rate
        self._paused = threading.Event()
        # Audio → engine queue (raw segments)
        self._audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)
        # Engine → UI queue (processed dicts)
        self.processed_queue: "queue.Queue[dict]" = queue.Queue(maxsize=8)
        self.user_formants = {}
        self._worker_thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._latest_raw = None
        self._latest_processed = None
        self._lock = threading.Lock()
    # ------------------------------------------------------------------
    # CORE: process_raw
    # ------------------------------------------------------------------

    def get_latest_raw(self):
        """Return the most recent raw frame (pitch+formants+segment)."""
        with self._lock:
            return self._latest_raw

    def get_latest_processed(self):
        """Return the most recent processed frame (smoothed, stable, etc)."""
        with self._lock:
            return self._latest_processed

    def process_raw(self, raw_dict):
        """
        Take a raw engine frame and return a fully processed frame with
        smoothed pitch/formants/vowel, stability info, scores, and debug data.
        """
        with self._lock:
            self._latest_raw = raw_dict

        f0_raw = raw_dict.get("f0")
        f1_raw, f2_raw, f3_raw = raw_dict.get("formants", (None, None, None))
        vowel_raw = raw_dict.get("vowel") or raw_dict.get("vowel_guess")
        lpc_conf = float(raw_dict.get("confidence", 0.0))

        f0_s = self._smooth_pitch(f0_raw, lpc_conf)
        f1_s, f2_s, f3_s = self._smooth_formants(
            raw_dict, f1_raw, f2_raw, f3_raw, lpc_conf
        )
        vowel_s = self.label_smoother.update(vowel_raw, confidence=lpc_conf)
        stable, stability_score = self._read_stability()

        processed = self._build_processed_frame(
            raw_dict, f0_raw, f0_s, f1_s, f2_s, f3_s,
            vowel_raw, vowel_s, lpc_conf, stable, stability_score,
        )

        with self._lock:
            self._latest_processed = processed

        return processed

    def _smooth_pitch(self, f0_raw, lpc_conf):
        return self.pitch_smoother.update(f0_raw, confidence=lpc_conf)

    def _smooth_formants(self, raw_dict, f1_raw, f2_raw, f3_raw, lpc_conf):
        hf = raw_dict.get("hybrid_formants")
        if isinstance(hf, (list, tuple)) and len(hf) == 3:
            f1_in, f2_in, f3_in = hf
        else:
            f1_in, f2_in, f3_in = f1_raw, f2_raw, f3_raw
        return self.formant_smoother.update(
            f1=f1_in, f2=f2_in, f3=f3_in, confidence=lpc_conf
        )

    def _read_stability(self):
        stable = getattr(self.formant_smoother, "formants_stable", False)
        stability_score = getattr(
            self.formant_smoother, "_stability_score", float("inf")
        )
        return stable, stability_score

    def _build_processed_frame(
        self, raw_dict, f0_raw, f0_s, f1_s, f2_s, f3_s,
        vowel_raw, vowel_s, lpc_conf, stable, stability_score,
    ):
        return {
            "f0_raw": f0_raw,
            "f0": f0_s,
            "formants": (f1_s, f2_s, f3_s),
            "hybrid_formants": raw_dict.get("hybrid_formants"),
            "smoothed_formants": {"f1": f1_s, "f2": f2_s, "f3": f3_s},
            "vowel": vowel_s,
            "vowel_guess": vowel_raw,
            "confidence": lpc_conf,
            "vowel_score": raw_dict.get("vowel_score"),
            "resonance_score": raw_dict.get("resonance_score"),
            "overall": raw_dict.get("overall"),
            "stable": stable,
            "stability_score": stability_score,
            "method": raw_dict.get("method"),
            "roots": raw_dict.get("roots"),
            "peaks": raw_dict.get("peaks"),
            "lpc_order": raw_dict.get("lpc_order"),
            "segment": raw_dict.get("segment"),
        }

    # ------------------------------------------------------------------
    # RUNTIME: accepting audio and driving the engine
    # ------------------------------------------------------------------

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def submit_audio_segment(self, segment: np.ndarray):
        """Called from the audio callback. Must be non-blocking."""
        if self._paused.is_set():
            return
        if segment is None:
            return
        try:
            self._audio_queue.put_nowait(segment)
        except queue.Full:
            logger.debug("Audio queue full; dropping frame")

    def _worker_loop(self):
        """Background thread: consume audio, run engine, push processed frames."""
        while not self._stop_flag.is_set():
            try:
                segment = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Engine call: match FormantAnalysisEngine.process_frame signature
            try:
                raw = self.engine.process_frame(segment, self.sample_rate)
            except Exception:
                logger.exception("Engine error processing frame")
                continue

            processed = self.process_raw(raw)

            # Keep only the most recent processed frame
            if self.processed_queue.full():
                try:
                    self.processed_queue.get_nowait()
                    logger.warning("Processed queue full; dropping oldest frame")
                except queue.Empty:
                    pass

            self.processed_queue.put_nowait(processed)

    def start_worker(self):
        """Start the background analyzer worker."""
        if self._worker_thread is not None:
            return
        # Reset smoothing state at start
        self.reset()
        self._stop_flag.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
        )
        self._worker_thread.start()

    def stop_worker(self):
        self._stop_flag.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=0.5)
            if self._worker_thread.is_alive():
                logger.warning("Worker thread did not stop within timeout")
            self._worker_thread = None

    # ------------------------------------------------------------------
    # Reset smoothing state
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all smoothing state, e.g., between calibrations or sessions."""
        # Pitch smoother
        if hasattr(self.pitch_smoother, "current"):
            self.pitch_smoother.current = None
        if hasattr(self.pitch_smoother, "audio_buffer"):
            self.pitch_smoother.audio_buffer.clear()

        # Formant smoother (MedianSmoother)
        fs = self.formant_smoother
        if hasattr(fs, "buf_f1"):
            fs.buf_f1.clear()
        if hasattr(fs, "buf_f2"):
            fs.buf_f2.clear()
        if hasattr(fs, "buf_f3"):
            fs.buf_f3.clear()
        if hasattr(fs, "stability") and hasattr(fs.stability, "reset"):
            fs.stability.reset()
        if hasattr(fs, "formants_stable"):
            fs.formants_stable = False
        if hasattr(fs, "_stability_score"):
            fs._stability_score = float("inf")

        # Label smoother
        if hasattr(self.label_smoother, "current"):
            self.label_smoother.current = None
        if hasattr(self.label_smoother, "last"):
            self.label_smoother.last = None
        if hasattr(self.label_smoother, "counter"):
            self.label_smoother.counter = 0
