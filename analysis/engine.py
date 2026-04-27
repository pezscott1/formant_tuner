from __future__ import annotations
import logging
import numpy as np
from typing import Optional
from analysis.pitch import estimate_pitch
import analysis.lpc as lpc_mod
import analysis.vowel_classifier as vowel_mod
import analysis.scoring as scoring_mod
from analysis.hybrid_formants import estimate_formants_hybrid

logger = logging.getLogger(__name__)
VOWEL_GUESS_CONF_MIN = 0.1
MIN_FRAME_SIZE = 256  # ~5 ms at 48 kHz; below this estimators are unreliable


class FormantAnalysisEngine:
    def __init__(
            self,
            voice_type="bass",
            pitch_tracker=None,
            vowel_classifier=None,
            use_hybrid=False,
            vowel_hint=None,
    ):
        self.profile_classifier = None
        self.voice_type = voice_type
        self.calibrating = False
        # Safe defaults
        self.pitch_tracker = pitch_tracker or estimate_pitch
        self.vowel_classifier = vowel_classifier

        # Hybrid toggle
        self.use_hybrid = use_hybrid
        self.vowel_hint = vowel_hint

        self.user_formants = {}
        self._latest_raw = None

    def set_user_formants(self, fmts: dict[str, dict[str, float]]) -> None:
        self.user_formants = fmts

    def get_latest_raw(self):
        return self._latest_raw

    def process_frame(self, frame: np.ndarray, sr: int) -> dict:
        if frame is None or frame.size < MIN_FRAME_SIZE:
            return self._null_result("frame_too_short")
        try:
            f0, voiced = self._compute_pitch_and_voicing(frame, sr)
            f1, f2, f3, conf, method, hybrid = self._compute_formants(frame, sr)
            vowel, vowel_guess, vowel_confidence, vowel_score, resonance_score = \
                self._compute_vowel_and_scores(f1, f2, f3, f0, conf)

            overall = float(min(vowel_score, resonance_score))

            out = self._build_result_dict(
                f1=f1, f2=f2, f3=f3, f0=f0, conf=conf, method=method,
                vowel=vowel, vowel_guess=vowel_guess, vowel_confidence=vowel_confidence,
                vowel_score=vowel_score, resonance_score=resonance_score,
                overall=overall, voiced=voiced, segment=frame, hybrid_formants=hybrid,
            )
            self._latest_raw = out
            return out
        except Exception:
            logger.exception("Engine error in process_frame")
            return self._null_result("exception")

    @staticmethod
    def _null_result(reason: str = "") -> dict:
        if reason:
            logger.debug("Returning null result: %s", reason)
        return {
            "f1": None, "f2": None, "f3": None,
            "formants": (None, None, None),
            "hybrid_formants": None,
            "f0": None,
            "confidence": 0.0,
            "method": None,
            "vowel": None,
            "vowel_guess": None,
            "vowel_confidence": 0.0,
            "vowel_score": 0.0,
            "resonance_score": 0.0,
            "overall": 0.0,
            "segment": None,
            "voiced": False,
        }

    def _compute_vowel_and_scores(
            self,
            f1: Optional[float],
            f2: Optional[float],
            f3: Optional[float],
            f0: Optional[float],
            conf: Optional[float],
    ) -> tuple[Optional[str], Optional[str], float, float, float]:

        if self.calibrating:
            return None, None, 0.0, 0.0, 0.0
        if f2 is None:
            return None, None, 0.0, 0.0, 0.0

        # ---------------------------------------------------------
        # Unified vowel classification
        # ---------------------------------------------------------

        vowel = None
        vowel_guess = None
        vowel_confidence = 0.0

        # Prefer calibrated classifier if provided
        if self.profile_classifier is not None:
            try:
                vowel_guess, vowel_confidence = self.profile_classifier(f1, f2)
                vowel = vowel_guess if (vowel_confidence >=
                                        VOWEL_GUESS_CONF_MIN) else None
            except Exception as e:
                logger.warning("Profile classifier error: %s", e)
                vowel = None
                vowel_guess = None
                vowel_confidence = 0.0

        # Fallback to legacy classifier only if no profile classifier
        elif conf is not None and conf >= VOWEL_GUESS_CONF_MIN:
            try:
                res = vowel_mod.classify_vowel(f1, f2, voice_type=self.voice_type)
                if isinstance(res, tuple) and len(res) >= 2:
                    vowel_guess = res[0]
                    vowel_confidence = float(res[1])
                    vowel = vowel_guess if (vowel_confidence >=
                                            VOWEL_GUESS_CONF_MIN) else None
            except Exception as e:
                logger.warning("Classifier error: %s", e)
                vowel = None
                vowel_guess = None
                vowel_confidence = 0.0

        # Build target tuple from user_formants
        if vowel in self.user_formants:
            vf = self.user_formants[vowel]
            target = (vf.get("f1"), vf.get("f2"), vf.get("f3"))
        else:
            target = (None, None, None)

        measured = (f1, f2, f3)
        raw_vowel_score = scoring_mod.live_score_formants(
            target, measured, tolerance=50)
        resonance_score = scoring_mod.resonance_tuning_score(
            measured, f0, tolerance=50)

        # Normalize vowel_score to 100 when user targets exist and score > 0
        if vowel in self.user_formants and raw_vowel_score > 0.0:
            vowel_score = 100.0
        else:
            vowel_score = 0.0

        return vowel, vowel_guess, vowel_confidence, vowel_score, resonance_score

    def _compute_pitch_and_voicing(
            self, frame: np.ndarray, sr: int
    ) -> tuple[Optional[float], bool]:
        result = self.pitch_tracker(frame, sr)

        if result is None or result.f0 is None:
            return None, False

        f0 = result.f0
        # Normalize non-finite to None
        if not np.isfinite(f0):
            return None, False

        voiced = f0 > 50
        return f0, voiced

    def _compute_formants(self, frame: np.ndarray, sr: int):
        if self.use_hybrid or self.calibrating:
            hres = estimate_formants_hybrid(frame, sr, vowel_hint=self.vowel_hint)
            f1 = hres.f1
            f2 = hres.f2
            f3 = hres.f3
            conf = hres.confidence
            method = hres.method  # "lpc", "te", or "hybrid_front"
            hybrid = (f1, f2, f3)
        else:
            lres = lpc_mod.estimate_formants(frame, sr)
            f1 = lres.f1
            f2 = lres.f2
            f3 = lres.f3
            conf = lres.confidence
            method = getattr(lres, "method", "lpc")
            hybrid = None

        return f1, f2, f3, conf, method, hybrid

    @staticmethod
    def _build_result_dict(
            f1, f2, f3, f0, conf, method,
            vowel, vowel_guess, vowel_confidence,
            vowel_score, resonance_score, overall,
            voiced, segment,
            hybrid_formants=None,
    ) -> dict:
        return {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "formants": (f1, f2, f3),
            "hybrid_formants": hybrid_formants,
            "f0": f0,
            "confidence": conf,
            "method": method,
            "vowel": vowel,
            "vowel_guess": vowel_guess,
            "vowel_confidence": vowel_confidence,
            "vowel_score": vowel_score,
            "resonance_score": resonance_score,
            "overall": overall,
            "segment": segment,
            "voiced": voiced,
        }
