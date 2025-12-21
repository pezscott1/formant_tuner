import logging
from typing import Optional, Mapping, Any
import numpy as np
from analysis.vowel_data import FORMANTS, VOWEL_MAP, PITCH_RANGES

logger = logging.getLogger(__name__)

FORMANT_TOLERANCE = 0.25  # ±25% tolerance around reference values


def get_vowel_ranges(voice_type, vowel):
    """Return plausible F1/F2 ranges for a given voice type and vowel."""
    vt = voice_type.lower()
    if vt not in FORMANTS:
        vt = "tenor"
    ref = FORMANTS[vt].get(vowel)
    if not ref:
        return None
    f1, f2 = ref[0], ref[1]
    f1_low, f1_high = f1 * (1 - FORMANT_TOLERANCE), f1 * (
        1 + FORMANT_TOLERANCE
    )
    f2_low, f2_high = f2 * (1 - FORMANT_TOLERANCE), f2 * (
        1 + FORMANT_TOLERANCE
    )
    return f1_low, f1_high, f2_low, f2_high


def is_plausible_formants(f1, f2, voice_type="tenor", vowel=None):
    """Check plausibility of F1/F2 values for a given voice type and vowel."""
    if f1 is None or f2 is None:
        return False, "missing formant"
    if f1 > f2:
        return False, "f1 > f2 (swapped)"

    # Global physiological minimums
    if f1 < 150 or f2 < 400:
        return False, "formants too low"

    # Vowel-specific ranges
    ranges = get_vowel_ranges(voice_type, vowel)
    if not ranges:
        return True, "ok"

    f1_low, f1_high, f2_low, f2_high = ranges
    if not (f1_low <= f1 <= f1_high):
        return False, f"f1 out of range ({f1:.0f} Hz)"
    if not (f2_low <= f2 <= f2_high):
        return False, f"f2 out of range ({f2:.0f} Hz)"

    return True, "ok"


def is_plausible_pitch(f0, voice_type="tenor"):
    """Check plausibility of pitch F0 for a given voice type."""
    if f0 is None or np.isnan(f0):
        return False, "missing pitch"
    vt = voice_type.lower()
    low, high = PITCH_RANGES.get(vt, (100, 600))
    if not (low <= f0 <= high):
        return False, f"f0 out of range ({f0:.0f} Hz)"
    return True, "ok"


def guess_vowel(f1, f2, voice_type="bass", last_guess=None):
    """
    Guess the closest vowel based on measured F1/F2 and reference formants.
    Falls back to last_guess if inputs are missing or implausible.
    """
    if f1 is None or f2 is None:
        return last_guess
    if f2 - f1 < 500:
        return last_guess
    best_vowel, best_dist = None, float("inf")
    ref_map = FORMANTS.get(voice_type, VOWEL_MAP)
    for vowel, formants in ref_map.items():
        t1, t2 = formants[0], formants[1]
        dist = abs(f1 - t1) + abs(f2 - t2)
        if dist < best_dist:
            best_vowel, best_dist = vowel, dist
    return best_vowel or last_guess


def robust_guess(measured_formants, voice_type="bass"):
    """Guess vowel robustly from measured formants."""
    if voice_type in FORMANTS:
        ref_map = {
            v: (f1, f2) for v, (f1, f2, *_) in FORMANTS[voice_type].items()
        }
    else:
        ref_map = VOWEL_MAP
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


def get_expected_formants(
    voice_type_local: Optional[str],
    vowel: str,
    f0: Optional[float] = None,
    user_offsets_local: Optional[Mapping[str, Any]] = None,
):
    """Return expected F1/F2 for a given voice type and vowel.

    f0 and user_offsets_local are accepted for API compatibility but not used.
    """
    # Mark intentionally unused to satisfy linters
    _ = f0
    _ = user_offsets_local

    vt = voice_type_local.lower() if voice_type_local else "tenor"
    preset = FORMANTS.get(vt, FORMANTS.get("tenor"))
    base = preset.get(vowel)
    if not base:
        return None, None
    f1, f2 = float(base[0]), float(base[1])
    return int(round(f1)), int(round(f2))
