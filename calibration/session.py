from datetime import datetime, timezone
from analysis.vowel import is_plausible_formants


class CalibrationSession:
    def __init__(self, profile_name, voice_type, vowels,
                 profile_manager=None, existing_profile=None):

        self.profile_name = profile_name
        self.voice_type = voice_type
        self.vowels = vowels
        self.profile_manager = profile_manager

        # Core data structures
        self.data = {}          # vowel → dict of f1,f2,f0,confidence,stability,weight
        self.retries_map = {}   # vowel → retry count

        # Load existing profile if present
        if isinstance(existing_profile, dict):
            for vowel, entry in existing_profile.items():
                if vowel == "voice_type":
                    continue
                if isinstance(entry, dict):
                    # Make a shallow copy so we can enrich it
                    self.data[vowel] = entry.copy()

    def increment_retry(self, vowel: str) -> None:
        """Increment retry count for a vowel."""
        if vowel in self.retries_map:
            self.retries_map[vowel] += 1
        else:
            self.retries_map[vowel] = 1

    # ---------------------------------------------------------
    # Handle a captured vowel result
    # ---------------------------------------------------------
    def handle_result(
            self,
            vowel: str,
            f1: float,
            f2: float,
            f0: float | None,
            confidence: float,
            stability: float,
    ):
        import numpy as np
        """
        Update the vowel entry using weighted averaging.
        Backward‑compatible with legacy profiles that lack weight/confidence/stability.
        """

        # If this vowel has no previous data, create a fresh entry
        if vowel not in self.data:
            self.data[vowel] = {
                "f1": f1,
                "f2": f2,
                "f0": f0,
                "confidence": confidence,
                "stability": stability,
                "weight": 1.0,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            return True, False, f"Accepted first measurement for /{vowel}/"

        # Existing entry (may be legacy)
        old = self.data[vowel]

        # -----------------------------
        # Backward‑compatibility fixes
        # -----------------------------
        old_f1 = old.get("f1", f1)
        old_f2 = old.get("f2", f2)
        old_f0 = old.get("f0", f0)
        # Sanitize legacy NaN values
        if old_f0 is None or not np.isfinite(old_f0):
            old_f0 = None
        old_conf = old.get("confidence", 1.0)
        old_stab = old.get("stability", 0.0)
        old_weight = old.get("weight", 1.0)

        # -----------------------------
        # Weighted update
        # -----------------------------
        new_weight = old_weight + 1.0

        new_f1 = (old_f1 * old_weight + f1) / new_weight
        new_f2 = (old_f2 * old_weight + f2) / new_weight

        if f0 is not None and old_f0 is not None:
            new_f0 = (old_f0 * old_weight + f0) / new_weight
        elif f0 is not None:
            new_f0 = f0
        else:
            new_f0 = old_f0
        print(f"[HANDLE_RESULT] f0={f0} type={type(f0)} "
              f"isfinite={np.isfinite(f0) if f0 is not None else 'None'}")
        # Confidence and stability smoothing
        new_conf = (old_conf * old_weight + confidence) / new_weight
        new_stab = (old_stab * old_weight + stability) / new_weight

        # Store updated entry
        self.data[vowel] = {
            "f1": new_f1,
            "f2": new_f2,
            "f0": new_f0,
            "confidence": new_conf,
            "stability": new_stab,
            "weight": new_weight,
            "saved_at": datetime.now(timezone.utc).isoformat()
        }

        return True, False, f"Updated /{vowel}/ (weight={new_weight:.1f})"

    # ---------------------------------------------------------
    # Save profile
    # ---------------------------------------------------------
    def save_profile(self):
        # Use self.data (not self.results)
        normalized = normalize_profile_for_save(self.data, self.retries_map)
        normalized["voice_type"] = self.voice_type

        # Prevent double suffixing
        if self.profile_name.endswith(f"_{self.voice_type}"):
            base_name = self.profile_name
        else:
            base_name = f"{self.profile_name}_{self.voice_type}"

        if self.profile_manager is None:
            raise RuntimeError("profile_manager is not set on CalibrationSession")

        self.profile_manager.save_profile(base_name, normalized)
        return base_name


# ---------------------------------------------------------
# Normalize for saving
# ---------------------------------------------------------
def normalize_profile_for_save(user_formants, retries_map):
    out = {}

    for vowel, vals in user_formants.items():
        if vals is None:
            continue

        f1 = vals["f1"]
        f2 = vals["f2"]
        f0 = vals["f0"]
        conf = vals["confidence"]
        stab = vals["stability"]

        # Swap if reversed
        if f1 is not None and f2 is not None and f1 > f2:
            f1, f2 = f2, f1

        retries = retries_map.get(vowel, 0)
        ok, reason = is_plausible_formants(f1, f2, vowel=vowel)
        reason_text = "ok" if ok else reason

        out[vowel] = {
            "f1": f1,
            "f2": f2,
            "f0": f0,
            "confidence": conf,
            "stability": stab,
            "weight": vals.get("weight", 1),
            "retries": retries,
            "reason": reason_text,
            "saved_at": vals.get("saved_at", datetime.now(timezone.utc).isoformat()),
            "source": "calibration",
        }

    return out
