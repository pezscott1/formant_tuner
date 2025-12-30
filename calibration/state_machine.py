import logging
from time import monotonic

logger = logging.getLogger(__name__)


class CalibrationStateMachine:
    """
    Simple phase state machine for calibration:

      phases: "prep" → "sing" → "capture" → next vowel → ... → "finished"
    """

    MAX_RETRIES = 3

    def __init__(
        self,
        vowels,
        prep_seconds: int = 3,
        sing_seconds: int = 2,
        capture_seconds: int = 2,
    ):
        self.vowels = list(vowels)
        self.prep_seconds_default = prep_seconds
        self.sing_seconds_default = sing_seconds
        self.capture_seconds_default = capture_seconds

        self.index = 0
        self.phase = "prep"

        self.prep_secs = self.prep_seconds_default
        self.sing_secs = self.sing_seconds_default
        self.capture_secs = self.capture_seconds_default

        self.retry_count = 0
        self.capture_start_time = None
        self._last_heartbeat = monotonic()

    @property
    def current_vowel(self):
        if 0 <= self.index < len(self.vowels):
            return self.vowels[self.index]
        return None

    def tick(self):
        if self.phase == "finished" or self.current_vowel is None:
            return {"event": "finished"}

        now = monotonic()
        if now - self._last_heartbeat > 1.0:
            logger.debug("calib heartbeat phase=%s idx=%d", self.phase, self.index)
            self._last_heartbeat = now

        if self.phase == "prep":
            if self.prep_secs > 0:
                secs = self.prep_secs
                self.prep_secs -= 1
                return {"event": "prep_countdown", "secs": secs}
            else:
                self.phase = "sing"
                self.sing_secs = self.sing_seconds_default
                return {"event": "start_sing", "vowel": self.current_vowel}

        elif self.phase == "sing":
            if self.sing_secs > 0:
                secs = self.sing_secs
                self.sing_secs -= 1
                return {"event": "sing_countdown", "secs": secs}
            else:
                self.phase = "capture"
                self.capture_secs = self.capture_seconds_default
                self.capture_start_time = monotonic()
                return {"event": "start_capture"}

        elif self.phase == "capture":
            if self.capture_secs > 0:
                self.capture_secs -= 1
                return {"event": "capture_tick"}
            else:
                return {"event": "capture_ready"}

        return {"event": "noop"}

    def advance(self):
        self.index += 1
        self.retry_count = 0  # reset retries for next vowel

        if self.index >= len(self.vowels):
            self.phase = "finished"
            return {"event": "finished"}

        self.phase = "prep"
        self.prep_secs = self.prep_seconds_default
        self.sing_secs = self.sing_seconds_default
        self.capture_secs = self.capture_seconds_default
        self.capture_start_time = None

        return {"event": "next_vowel", "vowel": self.current_vowel}

    def check_timeout(self, capture_timeout: float) -> bool:
        if self.phase != "capture":
            return False
        if self.capture_start_time is None:
            return False

        now = monotonic()
        return (now - self.capture_start_time) > capture_timeout

    def retry_current_vowel(self):
        self.retry_count += 1
        if self.retry_count >= self.MAX_RETRIES:
            return {"event": "max_retries", "vowel": self.current_vowel}

        self.phase = "prep"
        self.prep_secs = self.prep_seconds_default
        self.sing_secs = self.sing_seconds_default
        self.capture_secs = self.capture_seconds_default
        self.capture_start_time = None

        return {"event": "retry", "vowel": self.current_vowel}

    def force_capture_mode(self):
        self.phase = "capture"
        self.capture_start_time = monotonic()

    def is_done(self):
        return self.phase == "finished"
