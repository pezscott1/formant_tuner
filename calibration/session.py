# calibration/session.py
import os
import json
from datetime import timezone, datetime
from analysis.vowel import is_plausible_formants
from typing import Any, Dict


PROFILES_DIR = "profiles"
os.makedirs(PROFILES_DIR, exist_ok=True)


def profile_path(base_name: str) -> str:
    """Return the full path to the JSON profile file for a base name."""
    return os.path.join(PROFILES_DIR, f"{base_name}_profile.json")


class CalibrationSession:
    """
    Pure calibration logic:

      - tracks which vowel is active
      - stores accepted results
      - retry logic
      - decides when calibration is complete
      - saves the profile to disk
    """

    def __init__(self, profile_name: str, voice_type: str, vowels):
        self.profile_name = profile_name
        self.voice_type = voice_type
        self.vowels = list(vowels)
        self.voice_type = voice_type
        self.current_index = 0
        # vowel -> (f1, f2, f0)
        self.results = {}
        self.retries_map = {v: 0 for v in self.vowels}
        self.max_retries = 3

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    @property
    def current_vowel(self):
        if 0 <= self.current_index < len(self.vowels):
            return self.vowels[self.current_index]
        return None

    def is_complete(self) -> bool:
        return self.current_index >= len(self.vowels)

    # ---------------------------------------------------------
    # Handle a compute result
    # ---------------------------------------------------------
    def handle_result(self, f1, f2, f0):
        """
        Returns:
            accepted: bool
            skipped: bool
            message: str
        """
        vowel = self.current_vowel
        if vowel is None:
            return False, False, "No vowel active"

        def _is_nan(x):
            return isinstance(x, float) and x != x

        ok_formants = (
                f1 is not None
                and f2 is not None
                and not _is_nan(f1)
                and not _is_nan(f2)
        )
        ok_pitch = f0 is not None and not _is_nan(f0)

        # ✅ Successful capture
        if ok_formants:
            self.results[vowel] = (
                float(f1),
                float(f2),
                float(f0) if ok_pitch else None,
            )
            self.current_index += 1
            return True, False, f"/{vowel}/ accepted"

        # ❌ Retry
        retries = self.retries_map[vowel]
        if retries < self.max_retries:
            self.retries_map[vowel] += 1
            return False, False, f"/{vowel}/ retry {self.retries_map[vowel]}"

        # ❌ Skip
        self.current_index += 1
        return False, True, f"/{vowel}/ skipped after {self.max_retries} attempts"

    # ---------------------------------------------------------
    # Save profile
    # ---------------------------------------------------------
    def save_profile(self) -> str:
        """
        Convert collected formants and retries into a rich profile dict
        and save it as <base_name>_profile.json.

        Returns the base_name (e.g., "scott_tenor").
        """
        base_name = f"{self.profile_name}_{self.voice_type}"
        path = profile_path(base_name)

        # Build user_formants mapping: vowel -> (f1, f2, f0)
        user_formants = {}
        for vowel, triple in self.results.items():
            f1, f2, f0 = triple
            user_formants[vowel] = (f1, f2, f0)

        profile_dict = normalize_profile_for_save(
            user_formants,
            retries_map=self.retries_map,
        )

        profile_dict: Dict[str, Any] = profile_dict
        profile_dict["voice_type"] = self.voice_type
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(profile_dict, fh, indent=2)

        return base_name


def normalize_profile_for_save(user_formants, retries_map=None):
    """
    Normalize a user_formants mapping (vowel -> (f1, f2, f0)) into
    a dict suitable for JSON saving.
    """
    out = {}
    retries_map = retries_map or {}
    if not isinstance(user_formants, dict):
        return out

    for vowel, vals in user_formants.items():
        f1 = f2 = f0 = None
        try:
            if isinstance(vals, (list, tuple)):
                if len(vals) > 0:
                    f1 = None if vals[0] is None else float(vals[0])
                if len(vals) > 1:
                    f2 = None if vals[1] is None else float(vals[1])
                if len(vals) > 2:
                    f0 = None if vals[2] is None else float(vals[2])
            elif isinstance(vals, dict):
                f1 = None if vals.get("f1") is None else float(vals.get("f1"))
                f2 = None if vals.get("f2") is None else float(vals.get("f2"))
                f0 = None if vals.get("f0") is None else float(vals.get("f0"))
        except Exception as e:  # noqa: E722
            print("normalize_profile_for_save failed: %s", e)
            f1, f2, f0 = None, None, None

        # Sanity: ensure f1 <= f2
        if f1 is not None and f2 is not None and f1 > f2:
            f1, f2 = f2, f1

        retries = int(retries_map.get(vowel, 0) or 0)
        ok, reason = is_plausible_formants(f1, f2)
        reason_text = "ok" if ok else reason

        out[vowel] = {
            "f1": None if f1 is None else float(f1),
            "f2": None if f2 is None else float(f2),
            "f0": None if f0 is None else float(f0),   # ✅ pitch, not f3
            "retries": retries,
            "reason": reason_text,
            "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": "calibration",
        }

    return out
