# analysis/plausibility.py
import numpy as np
from typing import Optional, Dict, Any
from analysis.vowel_data import PITCH_RANGES, VOWEL_CENTERS


# ---------------------------------------------------------
# Utility: check if a calibration entry is valid
# ---------------------------------------------------------

def _valid_calibration_entry(entry):
    if not entry:
        return False
    f1 = entry.get("f1")
    f2 = entry.get("f2")
    f0 = entry.get("f0")
    # Must have real formants and real pitch
    if f1 is None or f2 is None:
        return False
    if f0 is None or np.isnan(f0):
        return False
    return True


# ---------------------------------------------------------
# Adaptive vowel window
# ---------------------------------------------------------

_DEFAULT_WINDOWS = {
    "tenor": {
        "a": (450.0, 950.0, 500.0, 2300.0),
        "ɑ": (450.0, 950.0, 500.0, 2300.0),
        # add other vowels as needed
    }
}


def vowel_window(voice_type: str, vowel: str, calibrated: dict | None = None):
    # 1. Try calibrated
    if calibrated is not None:
        if vowel in calibrated:
            entry = calibrated[vowel]
            f1 = entry.get("f1")
            f2 = entry.get("f2")
            f0 = entry.get("f0")

            if f1 is not None and f2 is not None and f0 is not None:
                f1 = float(f1)
                f2 = float(f2)
                lo1 = f1 * 0.70
                hi1 = f1 * 1.30
                lo2 = f2 * 0.80
                hi2 = f2 * 1.20
                return lo1, hi1, lo2, hi2

        # 🔥 KEY CHANGE:
        # When calibrated dict is provided but vowel is not yet calibrated,
        # do NOT fall back to VOWEL_CENTERS. Let callers see "no window".
        return None

    # 2. Fallback to table-based centers (runtime classification only)
    vt = voice_type if voice_type in VOWEL_CENTERS else "tenor"
    centers = VOWEL_CENTERS.get(vt, {})
    entry = centers.get(vowel)

    if entry is None and vt in _DEFAULT_WINDOWS:
        default_win = _DEFAULT_WINDOWS[vt].get(vowel)
        if default_win is not None:
            return default_win

    if entry is None:
        return None

    base_f1, base_f2, *_ = entry
    lo1 = base_f1 * 0.70
    hi1 = base_f1 * 1.30
    lo2 = base_f2 * 0.80
    hi2 = base_f2 * 1.20
    return lo1, hi1, lo2, hi2

# ---------------------------------------------------------
# Plausibility checks
# ---------------------------------------------------------


def is_plausible_formants(
    f1: Optional[float],
    f2: Optional[float],
    voice_type: str = "baritone",
    vowel: Optional[str] = None,
    calibrated: Optional[Dict[str, Any]] = None,
):
    """
    Adaptive plausibility check.

    - Wide windows during calibration
    - Tight adaptive windows after acceptance
    - Never trusts bad calibration entries
    - Special handling for fronted baritone vowels
    """

    # -----------------------------
    # Basic sanity
    # -----------------------------
    if f1 is None or f2 is None:
        return False, "missing"
    if np.isnan(f1) or np.isnan(f2):
        return False, "nan"
    if f1 > f2:
        return False, "swapped"
    if f1 < 120 or f2 < 250:
        return False, "too-low"

    # If no vowel target → accept
    if vowel is None:
        return True, "ok"

    # -----------------------------
    # Get adaptive or fallback window
    # -----------------------------
    win = vowel_window(voice_type, vowel, calibrated)
    if win is None:
        return True, "ok"

    f1_low, f1_high, f2_low, f2_high = win

    # -----------------------------
    # Special handling for fronted vowels
    # -----------------------------
    # Scott's /ɔ/ and /u/ are fronted (high F2)
    if vowel in ("ɔ", "u"):
        # F2 is the anchor
        if not (f2_low <= f2 <= f2_high):
            return False, f"f2-out ({f2:.0f})"
        # Allow wide F1 drift
        return True, "ok"

    # /i/ also uses F2 anchor
    if vowel == "i":
        if not (f2_low <= f2 <= f2_high):
            return False, f"f2-out ({f2:.0f})"
        return True, "ok"

    # -----------------------------
    # Standard vowels
    # -----------------------------
    if not (f1_low <= f1 <= f1_high):
        return False, f"f1-out ({f1:.0f})"
    if not (f2_low <= f2 <= f2_high):
        return False, f"f2-out ({f2:.0f})"

    return True, "ok"


# ---------------------------------------------------------
# Pitch plausibility
# ---------------------------------------------------------

def is_plausible_pitch(f0: Optional[float], voice_type: str = "baritone"):
    if f0 is None or np.isnan(f0):
        return False, "missing pitch"
    low, high = PITCH_RANGES.get(voice_type.lower(), (80.0, 600.0))
    if not (low <= f0 <= high):
        return False, f"f0-out-of-range ({f0:.0f})"
    return True, "ok"
