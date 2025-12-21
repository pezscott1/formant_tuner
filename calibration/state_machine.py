# calibration/state_machine.py
import logging
from time import monotonic

logger = logging.getLogger(__name__)


class CalibrationStateMachine:
    """
    Simple phase state machine for calibration:

      phases: "prep" → "sing" → "capture" → next vowel → ... → "finished"

    Exposes:
      - current_vowel
      - tick() -> event dict
      - advance() -> event dict
      - check_timeout(timeout) -> bool
    """

    def __init__(
        self,
        vowels,
        prep_seconds: int = 3,
        sing_seconds: int = 2,
        capture_seconds: int = 1,
    ):
        self.vowels = list(vowels)
        self.prep_seconds_default = prep_seconds
        self.sing_seconds_default = sing_seconds
        self.capture_seconds_default = capture_seconds

        self.index = 0
        self.phase = "prep"  # "prep", "sing", "capture", "finished"

        self.prep_secs = self.prep_seconds_default
        self.sing_secs = self.sing_seconds_default
        self.capture_secs = self.capture_seconds_default

        self.capture_start_time = None
        self._last_heartbeat = monotonic()

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    @property
    def current_vowel(self):
        if 0 <= self.index < len(self.vowels):
            return self.vowels[self.index]
        return None

    # ---------------------------------------------------------
    # Tick
    # ---------------------------------------------------------
    def tick(self):
        """
        Advance timers and emit a high-level event describing what happened.

        Returns an event dict, e.g.:

        {"event": "prep_countdown", "secs": 2}
        {"event": "start_sing", "vowel": "a"}
        {"event": "sing_countdown", "secs": 1}
        {"event": "start_capture"}
        {"event": "capture_ready"}
        {"event": "finished"}
        """
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
                # Transition to sing
                self.phase = "sing"
                self.sing_secs = self.sing_seconds_default
                return {
                    "event": "start_sing",
                    "vowel": self.current_vowel,
                }

        elif self.phase == "sing":
            if self.sing_secs > 0:
                secs = self.sing_secs
                self.sing_secs -= 1
                return {"event": "sing_countdown", "secs": secs}
            else:
                # Transition to capture
                self.phase = "capture"
                self.capture_secs = self.capture_seconds_default
                self.capture_start_time = monotonic()
                return {"event": "start_capture"}

        elif self.phase == "capture":
            if self.capture_secs > 0:
                self.capture_secs -= 1
                # Let UI show "capturing..." without extra event spam
                return {"event": "capture_tick"}
            else:
                # Ready to process capture
                return {"event": "capture_ready"}

        return {"event": "noop"}

    # ---------------------------------------------------------
    # Advance to next vowel
    # ---------------------------------------------------------
    def advance(self):
        """
        Move to the next vowel or finish.

        Returns an event dict. If finished:

          {"event": "finished"}
        """
        self.index += 1

        if self.index >= len(self.vowels):
            self.phase = "finished"
            return {"event": "finished"}

        # Reset timers for next vowel
        self.phase = "prep"
        self.prep_secs = self.prep_seconds_default
        self.sing_secs = self.sing_seconds_default
        self.capture_secs = self.capture_seconds_default
        self.capture_start_time = None

        return {"event": "next_vowel", "vowel": self.current_vowel}

    # ---------------------------------------------------------
    # Timeout check
    # ---------------------------------------------------------
    def check_timeout(self, capture_timeout: float) -> bool:
        """
        Return True if current capture phase has exceeded given timeout.
        """
        if self.phase != "capture":
            return False
        if self.capture_start_time is None:
            return False

        now = monotonic()
        return (now - self.capture_start_time) > capture_timeout

    # Add this at the bottom of the class
    def is_done(self):
        return self.phase == "finished"
