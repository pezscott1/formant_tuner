# tuner/live_analyzer.py
import queue
import threading
import numpy as np


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
        self.paused = False
        # Audio → engine queue (raw segments)
        self._audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)
        # Engine → UI queue (processed dicts)
        self.processed_queue: "queue.Queue[dict]" = queue.Queue(maxsize=8)

        self._worker_thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._latest_raw = None
        self._latest_processed = None
        self._lock = threading.Lock()
    # ------------------------------------------------------------------
    # CORE: process_raw (unchanged)
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
        Take a raw engine frame and return a fully processed frame:
          - smoothed pitch
          - smoothed formants
          - smoothed vowel
          - stability info
          - confidence + scoring
          - raw vowel guess
          - fallback formants + LPC debug
        """

        # =========================================================
        # 1. Store RAW frame for calibration / spectrogram
        # =========================================================
        with self._lock:
            self._latest_raw = raw_dict

        # Extract raw values
        f0_raw = raw_dict.get("f0")
        f1_raw, f2_raw, f3_raw = raw_dict.get("formants", (None, None, None))
        hybrid = raw_dict.get("hybrid_formants")
        vowel_raw = raw_dict.get("vowel") or raw_dict.get("vowel_guess")
        lpc_conf = float(raw_dict.get("confidence", 0.0))

        # =========================================================
        # 2. Pitch smoothing
        # =========================================================
        f0_s = self.pitch_smoother.update(f0_raw, confidence=lpc_conf)

        # =========================================================
        # 3. Formant smoothing
        # =========================================================
        f1_s, f2_s, f3_s = self.formant_smoother.update(
            f1=f1_raw,
            f2=f2_raw,
            f3=f3_raw,
            confidence=lpc_conf,
        )

        # =========================================================
        # 4. Vowel smoothing
        # =========================================================
        vowel_s = self.label_smoother.update(vowel_raw, confidence=lpc_conf)

        # =========================================================
        # 5. Scoring (optional but recommended)
        # =========================================================
        vowel_score = raw_dict.get("vowel_score")
        resonance_score = raw_dict.get("resonance_score")
        overall = raw_dict.get("overall")

        # =========================================================
        # 6. Stability
        # =========================================================
        stable = getattr(self.formant_smoother, "formants_stable", False)
        stability_score = getattr(
            self.formant_smoother, "_stability_score", float("inf")
        )

        # =========================================================
        # 7. Build processed frame
        # =========================================================
        processed = {
            # Pitch
            "f0_raw": f0_raw,
            "f0": f0_s,

            # Formants
            "formants": (f1_s, f2_s, f3_s),
            "hybrid_formants": hybrid,

            # Smoothed vowel
            "vowel": vowel_s,

            # Raw vowel guess
            "vowel_guess": vowel_raw,

            # Confidence from LPC
            "confidence": lpc_conf,
            # Scores
            "vowel_score": vowel_score,
            "resonance_score": resonance_score,
            "overall": overall,

            # Stability
            "stable": stable,
            "stability_score": stability_score,

            # Fallback formants
            "fb_f1": raw_dict.get("fb_f1"),
            "fb_f2": raw_dict.get("fb_f2"),

            # LPC / debug info
            "method": raw_dict.get("method"),
            "roots": raw_dict.get("roots"),
            "peaks": raw_dict.get("peaks"),
            "lpc_order": raw_dict.get("lpc_order"),
            "lpc_debug": raw_dict.get("lpc_debug"),

            # Raw segment for spectrogram
            "segment": raw_dict.get("segment"),
        }

        # =========================================================
        # 8. Store processed frame for tuner UI
        # =========================================================
        with self._lock:
            self._latest_processed = processed

        return processed

    # ------------------------------------------------------------------
    # RUNTIME: accepting audio and driving the engine
    # ------------------------------------------------------------------

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def submit_audio_segment(self, segment: np.ndarray):
        """Called from the audio callback. Must be non-blocking."""
        if self.paused:
            return
        if segment is None:
            return
        try:
            self._audio_queue.put_nowait(segment)
        except queue.Full:
            # Drop frames instead of blocking
            pass

    def _worker_loop(self):
        import traceback
        """Background thread: consume audio, run engine, push processed frames."""
        while not self._stop_flag.is_set():
            try:
                segment = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Engine call: match FormantAnalysisEngine.process_frame signature
            try:
                raw = self.engine.process_frame(segment, self.sample_rate)
            except Exception as e:
                print("[ENGINE ERROR]", e)
                traceback.print_exc()
                continue

            processed = self.process_raw(raw)

            # Keep only the most recent processed frame
            if self.processed_queue.full():
                try:
                    _ = self.processed_queue.get_nowait()
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
            self._worker_thread = None

    # ------------------------------------------------------------------
    # Reset smoothing state
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all smoothing state, e.g., between calibrations or sessions."""
        if hasattr(self.pitch_smoother, "current"):
            self.pitch_smoother.current = None
        if hasattr(self.pitch_smoother, "audio_buffer"):
            self.pitch_smoother.audio_buffer.clear()

        if hasattr(self.formant_smoother, "buffer"):
            self.formant_smoother.buffer.clear()

        if hasattr(self.label_smoother, "current"):
            self.label_smoother.current = None
        if hasattr(self.label_smoother, "last"):
            self.label_smoother.last = None
        if hasattr(self.label_smoother, "counter"):
            self.label_smoother.counter = 0
