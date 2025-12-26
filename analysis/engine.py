import numpy as np
from typing import Dict, Tuple, Any, Optional

from analysis.pitch import estimate_pitch
from analysis.lpc import estimate_formants, LPCConfig
from analysis.vowel import robust_guess, guess_vowel
from analysis.scoring import live_score_formants, resonance_tuning_score


class FormantAnalysisEngine:
    """
    Core analysis engine for the tuner.

    Responsibilities:
      - Estimate pitch (f0)
      - Estimate formants (F1, F2, F3)
      - Guess vowel (robust_guess + fallback)
      - Compute vowel/resonance/overall scores
      - Expose latest raw result for LiveAnalyzer / UI
    """

    def __init__(self, voice_type: str = "bass") -> None:
        self.voice_type = voice_type

        # Map: vowel -> (f1, f2, f3)
        self.user_formants: Dict[
            str, Tuple[Optional[float], Optional[float], Optional[float]]
        ] = {}

        self._latest_raw: Optional[Dict[str, Any]] = None
        self.calibrated_profile = {}

        # New LPC configuration (mic-aware)
        self.lpc_config = LPCConfig(
            win_len_ms=30,
            min_order=10,
            max_order=14,
            pre_emph=None,          # auto for 48 kHz
            mic_profile="bright_48k"
        )

    # ---------------------------------------------------------
    # User targets
    # ---------------------------------------------------------
    def set_user_formants(
        self,
        formant_map: Dict[str, Tuple[Optional[float],
                                     Optional[float], Optional[float]]],) -> None:
        """Set user-specific target formants for each vowel."""
        self.user_formants = formant_map or {}

    # ---------------------------------------------------------
    # Core processing
    # ---------------------------------------------------------
    def process_frame(self, signal: np.ndarray, sr: int) -> Dict[str, Any]:  # noqa: C901
        """
        Process a single audio frame.

        Returns a raw dict consumed by LiveAnalyzer:
          {
            "f0": float or None,
            "formants": (f1, f2, f3),
            "vowel": str or None,
            "vowel_guess": str or None,
            "vowel_confidence": float,
            "vowel_score": float,
            "resonance_score": float,
            "overall": float,
            "fb_f1": float or None,
            "fb_f2": float or None,
            "segment": signal.copy(),
            "confidence": float,
            "method": str,
            "lpc_debug": dict,
          }
        """
        if signal is None or len(signal) == 0:
            result = {
                "f0": None,
                "formants": (None, None, None),
                "vowel": None,
                "vowel_guess": None,
                "vowel_confidence": 0.0,
                "vowel_score": 0.0,
                "resonance_score": 0.0,
                "overall": 0.0,
                "fb_f1": None,
                "fb_f2": None,
                "confidence": 0.0,
                "method": "none",
                "lpc_debug": {},
            }
            self._latest_raw = result
            return result

        # ---------------- Pitch ----------------
        f0 = estimate_pitch(signal, sr)

        # ---------------- Formants (new LPC API) ----------------
        lpc_result = estimate_formants(
            signal,
            sr,
            config=self.lpc_config,
            debug=False,
        )

        f1 = lpc_result.f1
        f2 = lpc_result.f2
        f3 = lpc_result.f3

        fb_f1, fb_f2 = f1, f2  # feedback copies

        # ---------------- Vowel guess + confidence ----------------
        # ---------------- Vowel guess + confidence ----------------
        def _is_valid_scalar(x):
            # Accept plain Python numbers and numpy scalar types
            if x is None:
                return False
            if isinstance(x, (int, float, np.floating)):
                # Guard against NaN explicitly
                return not np.isnan(x)
            # Anything else (arrays, objects, etc.) is treated as invalid here
            return False

        if (
            lpc_result.confidence > 0.3
            and _is_valid_scalar(f1)
            and _is_valid_scalar(f2)
        ):
            try:
                best, conf, _second = robust_guess(
                    (float(f1), float(f2)),
                    voice_type=self.voice_type,
                )
                vowel = best
                vowel_conf = float(conf) if conf is not None else 0.0
            except Exception:
                try:
                    vowel = guess_vowel(float(f1), float(f2), self.voice_type)
                except Exception:
                    vowel = None
                vowel_conf = 0.0
        else:
            vowel = None
            vowel_conf = 0.0

        # ---------------- Scoring ----------------
        if not self.user_formants:
            vowel_score = 0.0
            resonance_score = 0.0
            overall = 0.0
        else:
            target_formants = self.user_formants.get(vowel, (None, None, None))
            vowel_score = live_score_formants(
                target_formants, (f1, f2, f3), tolerance=50
            )
            resonance_score = resonance_tuning_score(
                (f1, f2, f3), f0, tolerance=50
            )
            overall = 0.5 * vowel_score + 0.5 * resonance_score

        # ---------------- Build result ----------------
        result = {
            "f0": f0,
            "formants": (f1, f2, f3),
            "vowel": vowel,
            "vowel_guess": vowel,
            "vowel_confidence": vowel_conf,
            "vowel_score": vowel_score,
            "resonance_score": resonance_score,
            "overall": overall,
            "fb_f1": fb_f1,
            "fb_f2": fb_f2,
            "segment": signal.copy(),

            # New fields from FormantResult
            "confidence": lpc_result.confidence,
            "method": lpc_result.method,
            "lpc_order": lpc_result.lpc_order,
            "peaks": lpc_result.peaks,
            "roots": lpc_result.roots,
            "bandwidths": lpc_result.bandwidths,
            "lpc_debug": lpc_result.debug,
        }

        self._latest_raw = result
        return result

    # ---------------------------------------------------------
    # Latest frame for UI
    # ---------------------------------------------------------
    def get_latest_raw(self) -> Optional[Dict[str, Any]]:
        """Return the latest raw result dict produced by process_frame, or None."""
        return self._latest_raw
