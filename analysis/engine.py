from __future__ import annotations
import numpy as np

from analysis.lpc import estimate_formants
from analysis.pitch import estimate_pitch
import analysis.vowel_classifier as vowel_classifier
from analysis.scoring import live_score_formants, resonance_tuning_score

VOWEL_GUESS_CONF_MIN = 0.1  # tests treat 0.05 as "too low"


class FormantAnalysisEngine:
    def __init__(self, voice_type="bass", debug=False):
        self.voice_type = voice_type
        self.debug = debug

        # Ensure user_formants always exists
        self.user_formants = {}
        self._latest_raw = None

        # Hybrid toggle (tests assume it exists)
        self.use_hybrid = False

    # called by ProfileManager
    def set_user_formants(self, fmts: dict[str, dict[str, float]]) -> None:
        self.user_formants = fmts

    def get_latest_raw(self):
        return self._latest_raw

    def process_frame(self, signal, sr: int):
        # ---------------------------------------------------------
        # Empty frame
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # Pitch
        # ---------------------------------------------------------
        pitch_res = estimate_pitch(signal, sr)
        f0 = getattr(pitch_res, "f0", None)
        try:
            f0 = float(f0)
        except Exception:
            f0 = None
        if f0 is None or not np.isfinite(f0):
            f0 = None

        # ---------------------------------------------------------
        # LPC formants
        # ---------------------------------------------------------
        lpc_result = estimate_formants(signal, sr, debug=False)
        f1_lpc = lpc_result.f1
        f2_lpc = lpc_result.f2
        f3_lpc = lpc_result.f3

        fb_f1, fb_f2 = f1_lpc, f2_lpc

        # ---------------------------------------------------------
        # Hybrid formants (optional)
        # ---------------------------------------------------------
        hybrid_result = None

        if getattr(self, "use_hybrid", False):
            from analysis.hybrid_formants import estimate_formants_hybrid

            hybrid_result = estimate_formants_hybrid(
                signal,
                sr,
                vowel_hint=None,
                debug=False,
            )

            print(
                f"[HYBRID] LPC(f1={f1_lpc}, f2={f2_lpc}) "
                f"TE(f1={hybrid_result.te.f1}, f2={hybrid_result.te.f2}) "
                f"CHOSEN={hybrid_result.method} "
                f"HF1={hybrid_result.f1} HF2={hybrid_result.f2} "
                f"conf={hybrid_result.confidence:.2f}"
            )

        # ---------------------------------------------------------
        # Choose formants for vowel guess + scoring
        # ---------------------------------------------------------
        if hybrid_result is not None:
            hf1 = hybrid_result.f1
            hf2 = hybrid_result.f2
            hf3 = hybrid_result.f3

            def _bad(x):
                return x is None or isinstance(x, (tuple, list, dict))

            te_bad = (
                _bad(hf1)
                or _bad(hf2)
                or hf1 is None
                or hf2 is None
                or hf2 <= hf1
                or hf2 < 900  # TE F2 collapse
                or hf1 < 200  # TE F1 collapse
            )

            if te_bad:
                f1, f2, f3 = f1_lpc, f2_lpc, f3_lpc
            else:
                f1, f2, f3 = hf1, hf2, hf3
        else:
            f1, f2, f3 = f1_lpc, f2_lpc, f3_lpc

        # ---------------------------------------------------------
        # Vowel classification (hybrid-aware)
        # ---------------------------------------------------------
        vowel_guess = None
        vowel_conf = 0.0

        raw_f1, raw_f2 = f1, f2

        if (
            raw_f1 is not None
            and raw_f2 is not None
            and lpc_result.confidence is not None
            and lpc_result.confidence >= VOWEL_GUESS_CONF_MIN
        ):
            try:
                vowel_guess, vowel_conf, vowel_second = vowel_classifier.classify_vowel(
                    f1, f2, voice_type=self.voice_type
                )
            except Exception as e:
                print("[ENGINE WARNING] vowel classifier failed:", e)
                vowel_guess, vowel_conf, vowel_second = None, 0.0, None

        vowel = vowel_guess
        # ---------------------------------------------------------
        # Scoring (user formants + resonance)
        # ---------------------------------------------------------
        entry = None
        if isinstance(self.user_formants, dict) and vowel is not None:
            entry = self.user_formants.get(vowel)

        # ---- Normalize target formants ----
        if isinstance(entry, dict):
            t_f1 = entry.get("f1")
            t_f2 = entry.get("f2")
            t_f3 = entry.get("f3")
        elif isinstance(entry, (tuple, list)):
            t_f1 = entry[0] if len(entry) > 0 else None
            t_f2 = entry[1] if len(entry) > 1 else None
            t_f3 = entry[2] if len(entry) > 2 else None
        else:
            t_f1 = t_f2 = t_f3 = None

        # Tests expect tuple form, even when all None
        target_formants = (t_f1, t_f2, t_f3)

        # ---- Normalize measured formants ----
        def _clean(x):
            if isinstance(x, (tuple, list, dict)):
                return None
            try:
                return float(x)
            except Exception:
                return None

        mf1 = _clean(f1)
        mf2 = _clean(f2)
        mf3 = _clean(f3)
        measured_formants = (mf1, mf2, mf3)

        # ---- Call scoring functions ----
        # test_scoring_no_user_formants expects:
        #   target == (None,None,None) and vowel_score == 0.0
        vowel_score = live_score_formants(
            target_formants,
            measured_formants,
            tolerance=50,
        )

        # For no-user-formants case, keep resonance_score at 0.0
        if entry is None:
            resonance_score = 0.0
        else:
            resonance_score = resonance_tuning_score(
                (f1, f2, f3),
                f0,
                tolerance=50,
            )

        overall = 0.5 * vowel_score + 0.5 * resonance_score

        result = {
            "f0": f0,
            "formants": (f1_lpc, f2_lpc, f3_lpc),
            "vowel": vowel,
            "vowel_guess": vowel_guess,
            "vowel_confidence": vowel_conf,
            "vowel_score": vowel_score,
            "resonance_score": resonance_score,
            "overall": overall,
            "fb_f1": fb_f1,
            "fb_f2": fb_f2,
            "segment": signal.copy(),
            "confidence": lpc_result.confidence,
            "method": lpc_result.method,
            "peaks": lpc_result.peaks,
            "roots": lpc_result.roots,
            "bandwidths": lpc_result.bandwidths,
            "lpc_debug": lpc_result.debug,
        }

        if hybrid_result is not None:
            result["hybrid_formants"] = (f1, f2, f3)
            result["hybrid_method"] = hybrid_result.method
            result["hybrid_debug"] = hybrid_result.debug

        self._latest_raw = result
        return result
