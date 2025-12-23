# calibration/session.py

import os
import json
from datetime import timezone, datetime
from analysis.vowel import is_plausible_formants
from typing import Any


PROFILES_DIR = "profiles"
os.makedirs(PROFILES_DIR, exist_ok=True)


def profile_path(base_name: str) -> str:
    return os.path.join(PROFILES_DIR, f"{base_name}_profile.json")


def merge_formants(old_vals, new_vals, _vowel):
    """
    Merge logic:
      - If new is None or implausible → keep old
      - If old is None → use new
      - If both plausible → choose the one closer to expected vowel ranges
    """
    if old_vals is None:
        return new_vals

    old_f1, old_f2, old_f0 = old_vals
    new_f1, new_f2, new_f0 = new_vals

    # If new is missing or implausible → keep old
    ok_new, _ = is_plausible_formants(new_f1, new_f2, vowel=_vowel)
    if not ok_new:
        return old_vals

    ok_old, _ = is_plausible_formants(old_f1, old_f2, vowel=_vowel)
    if not ok_old:
        return new_vals

    # Both plausible → choose the one closer to expected vowel space
    # (simple heuristic: smaller |F1-F2| distance)
    old_dist = abs(old_f2 - old_f1)
    new_dist = abs(new_f2 - new_f1)

    return new_vals if new_dist < old_dist else old_vals


class CalibrationSession:
    def __init__(self, profile_name: str, voice_type: str, vowels):
        self.profile_name = profile_name
        self.voice_type = voice_type
        self.vowels = list(vowels)
        self.current_index = 0
        self.results = {}  # vowel -> (f1, f2, f0)
        self.retries_map = {v: 0 for v in self.vowels}
        self.max_retries = 3

    @property
    def current_vowel(self):
        if 0 <= self.current_index < len(self.vowels):
            return self.vowels[self.current_index]
        return None

    def is_complete(self) -> bool:
        return self.current_index >= len(self.vowels)

    def handle_result(self, f1, f2, f0):
        vowel = self.current_vowel
        if vowel is None:
            return False, False, "No vowel active"

        def _is_nan(x):
            return isinstance(x, float) and x != x

        print("Captured:", f1, f2, f0)
        ok_formants = (
            f1 is not None and f2 is not None and not _is_nan(f1) and not _is_nan(f2)
        )
        ok_pitch = f0 is not None and not _is_nan(f0)

        if ok_formants:
            self.results[vowel] = (
                float(f1),
                float(f2),
                float(f0) if ok_pitch else None,
            )
            self.current_index += 1
            return True, False, f"/{vowel}/ accepted"

        retries = self.retries_map[vowel]
        if retries < self.max_retries:
            self.retries_map[vowel] += 1
            return False, False, f"/{vowel}/ retry {self.retries_map[vowel]}"

        self.current_index += 1
        return False, True, f"/{vowel}/ skipped after {self.max_retries} attempts"

    # ---------------------------------------------------------
    # Save profile (with merge + overwrite)
    # ---------------------------------------------------------
    def save_profile(self) -> str:
        base_name = f"{self.profile_name}_{self.voice_type}"
        path = profile_path(base_name)

        # Build new formants
        new_formants = {
            vowel: (f1, f2, f0)
            for vowel, (f1, f2, f0) in self.results.items()
        }

        # Load existing profile if present
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                old_profile = json.load(fh)
        else:
            old_profile = {}

        # Extract old formants
        old_formants = {}
        for vowel, data in old_profile.items():
            if isinstance(data, dict):
                old_formants[vowel] = (
                    data.get("f1"),
                    data.get("f2"),
                    data.get("f0"),
                )

        # Merge
        merged = {}
        for vowel in self.vowels:
            old_vals = old_formants.get(vowel)
            new_vals = new_formants.get(vowel)
            if new_vals is None:
                merged[vowel] = old_vals
            else:
                merged[vowel] = merge_formants(old_vals, new_vals, vowel)

        # Normalize + save
        profile_dict: dict[str, Any] = normalize_profile_for_save(
            merged,
            retries_map=self.retries_map,
        )
        profile_dict["voice_type"] = self.voice_type

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(profile_dict, fh, indent=2)

        return base_name

    def increment_retry(self, vowel: str) -> None:
        if vowel in self.retries_map:
            self.retries_map[vowel] += 1
        else:
            self.retries_map[vowel] = 1


def normalize_profile_for_save(user_formants, retries_map=None):  # noqa: C901
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
        except Exception:
            f1, f2, f0 = None, None, None

        if f1 is not None and f2 is not None and f1 > f2:
            f1, f2 = f2, f1

        retries = int(retries_map.get(vowel, 0) or 0)
        ok, reason = is_plausible_formants(f1, f2, vowel=vowel)
        reason_text = "ok" if ok else reason

        out[vowel] = {
            "f1": None if f1 is None else float(f1),
            "f2": None if f2 is None else float(f2),
            "f0": None if f0 is None else float(f0),
            "retries": retries,
            "reason": reason_text,
            "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": "calibration",
        }

    return out
