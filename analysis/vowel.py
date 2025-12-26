# analysis/vowel.py
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from analysis.vowel_data import FORMANTS, PITCH_RANGES


FORMANT_TOLERANCE = 0.30  # ±30% default


@dataclass
class VowelResult:
    vowel: Optional[str]
    confidence: float
    reason: str


# ---------------------------------------------------------
# Vowel ranges (reference + calibrated)
# ---------------------------------------------------------
def get_vowel_ranges(
    voice_type: str,
    vowel: str,
    calibrated: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Return plausible F1/F2 ranges for a given voice type and vowel.
    Calibration overrides reference centers.
    """
    # 1. Calibration-aware
    if calibrated and vowel in calibrated:
        f1 = calibrated[vowel].get("f1")
        f2 = calibrated[vowel].get("f2")
        if f1 and f2:
            if vowel == "ɑ":
                tol = 0.80
            elif vowel in ("ɔ", "u"):
                tol = 0.50
            else:
                tol = FORMANT_TOLERANCE
            return (
                f1 * (1 - tol),
                f1 * (1 + tol),
                f2 * (1 - tol),
                f2 * (1 + tol),
            )

    # 2. Reference fallback
    vt = voice_type.lower()
    if vt not in FORMANTS:
        vt = "tenor"

    ref = FORMANTS[vt].get(vowel)
    if not ref:
        return None

    f1, f2 = ref[0], ref[1]

    if vowel == "ɑ":
        tol = 0.80
    elif vowel in ("ɔ", "u"):
        tol = 0.50
    else:
        tol = FORMANT_TOLERANCE

    return (
        f1 * (1 - tol),
        f1 * (1 + tol),
        f2 * (1 - tol),
        f2 * (1 + tol),
    )


# ---------------------------------------------------------
# Plausibility checks
# ---------------------------------------------------------
def is_plausible_formants(  # noqa: C901
    f1: Optional[float],
    f2: Optional[float],
    voice_type: str = "tenor",
    vowel: Optional[str] = None,
    calibrated: Optional[Dict[str, Any]] = None,
):
    """
    Check plausibility of F1/F2 values for a given voice type and vowel.
    """
    if f1 is None or f2 is None:
        return False, "missing formant"
    if np.isnan(f1) or np.isnan(f2):
        return False, "nan formant"
    if f1 > f2:
        return False, "swapped"
    if f1 < 120 or f2 < 300:
        return False, "too low"

    ranges = get_vowel_ranges(voice_type, vowel, calibrated)
    if vowel is None or not ranges:
        return True, "ok"

    f1_low, f1_high, f2_low, f2_high = ranges

    # Back vowels: F2 is the anchor
    if vowel in ("ɔ", "u"):
        if not (f2_low <= f2 <= f2_high):
            return False, "f2-out-of-range"
        if not (f1_low <= f1 <= f1_high):
            return True, "f1-drift"
        return True, "ok"

    # Standard vowels
    if not (f1_low <= f1 <= f1_high):
        return False, f"f1 out of range ({f1:.0f})"
    if not (f2_low <= f2 <= f2_high):
        return False, f"f2 out of range ({f2:.0f})"

    return True, "ok"


def is_plausible_pitch(f0: Optional[float], voice_type: str = "tenor"):
    if f0 is None or np.isnan(f0):
        return False, "missing pitch"
    low, high = PITCH_RANGES.get(voice_type.lower(), (100.0, 600.0))
    if not (low <= f0 <= high):
        return False, f"f0 out of range ({f0:.0f})"
    return True, "ok"


# ---------------------------------------------------------
# Vowel guessing
# ---------------------------------------------------------
def guess_vowel(
    f1: Optional[float],
    f2: Optional[float],
    voice_type: str = "bass",
    last_guess: Optional[str] = None,
) -> Optional[str]:
    """Legacy vowel guesser."""
    if f1 is None or f2 is None:
        return last_guess
    if np.isnan(f1) or np.isnan(f2):
        return last_guess
    if f2 - f1 < 500:
        return last_guess

    vt = voice_type if voice_type in FORMANTS else "tenor"
    ref_map = FORMANTS[vt]

    best_vowel, best_dist = None, float("inf")
    for vowel, formants in ref_map.items():
        t1, t2 = formants[0], formants[1]
        dist = abs(f1 - t1) + abs(f2 - t2)
        if dist < best_dist:
            best_vowel, best_dist = vowel, dist
    return best_vowel or last_guess


def robust_guess(
    measured_formants,
    voice_type: str = "bass",
):
    """Robust vowel guesser with confidence."""
    vt = voice_type if voice_type in FORMANTS else "tenor"
    ref_map = {v: (f1, f2) for v, (f1, f2, *_) in FORMANTS[vt].items()}

    valid = [f for f in measured_formants if f is not None and not np.isnan(f)]
    if len(valid) < 2:
        return None, 0.0, None

    f1, f2 = sorted(valid)[:2]
    scores = {
        v: ((f1 - tf1) ** 2 + (f2 - tf2) ** 2) ** 0.5
        for v, (tf1, tf2) in ref_map.items()
    }
    best, second = sorted(scores.items(), key=lambda kv: kv[1])[:2]
    confidence = second[1] / (best[1] + 1e-6)
    return best[0], float(confidence), second[0]
