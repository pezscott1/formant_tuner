import numpy as np
import sounddevice as sd
import queue, logging
from formant_utils import (
    estimate_formants_lpc,
    estimate_pitch,
    overall_rating,
    directional_feedback
)
from voice_analysis import Analyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S")
    
logger = logging.getLogger(__name__)

results_queue = queue.Queue()

class MicAnalyzer:
    def __init__(self, vowel_provider, tol_provider, pitch_provider, sample_rate, frame_ms, analyzer=None):
        self.vowel_provider = vowel_provider
        self.tol_provider = tol_provider
        self.pitch_provider = pitch_provider
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.analyzer = analyzer
        self.stream = None 
        self.buffer = []

    def audio_callback(self, indata, frames, time, status):
        frame = np.asarray(indata).flatten()
        self.buffer.extend(frame.tolist()) 
        f1, f2 = estimate_formants_lpc(frame, self.sample_rate)
        measured_formants = (f1, f2, None)   # <-- always return 3 values

        tolerance = self.tol_provider()
        current_vowel = self.vowel_provider()
        f0 = estimate_pitch(frame, self.sample_rate)

        vowel_score, resonance_score, overall = overall_rating(
            measured_formants,
            self.analyzer.user_formants.get(current_vowel, (None, None, None)) if self.analyzer else (None, None, None),
            f0,
            tolerance
        )

        fb_f1, fb_f2 = directional_feedback(
            measured_formants,
            self.analyzer.user_formants if self.analyzer else {},
            current_vowel,
            tolerance
        )

        status = {
            "f0": f0,
            "formants": measured_formants,
            "vowel_score": vowel_score,
            "resonance_score": resonance_score,
            "overall": overall,
            "fb_f1": fb_f1,
            "fb_f2": fb_f2,
        }
        results_queue.put(status)
        

    def start(self):
        if self.stream is None:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * self.frame_ms / 1000),
                channels=1,
                callback=lambda indata, frames, time, status: self.audio_callback(indata, frames, time, status)
            )
            self.stream.start()
            logger.info("Mic started")
            
    def stop(self, event=None):
        if self.stream is not None:
            if self.stream.active:
                self.stream.stop()
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None
            logger.info("Mic stopped")
        else:
            print("Mic already stopped")