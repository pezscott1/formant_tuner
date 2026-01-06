# calibration/session.py
from datetime import datetime, timezone
from analysis.plausibility import is_plausible_formants
import numpy as np
from analysis.vowel_data import TRIANGLES, TRIANGLE_WEIGHTS


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

    def _weights_for(self, vowel):
        return TRIANGLE_WEIGHTS.get(vowel, (1 / 3, 1 / 3, 1 / 3))

    def get_calibrated_anchors(self):
        """Return a dict of vowel → (f1, f2) for calibrated vowels only."""
        anchors = {}
        for vowel, entry in self.data.items():
            f1 = entry.get("f1")
            f2 = entry.get("f2")
            if f1 is not None and f2 is not None:
                anchors[vowel] = (float(f1), float(f2))
        return anchors

    @staticmethod
    def barycentric_interpolate(target, triangle):
        """
        target: (wA, wB, wC) barycentric weights that sum to 1
        triangle: dict with keys 'A','B','C' mapping to (f1,f2) coordinates
        """
        A = np.array(triangle["A"])
        B = np.array(triangle["B"])
        C = np.array(triangle["C"])

        wA, wB, wC = target  # barycentric weights
        return wA * A + wB * B + wC * C

    def compute_interpolated_vowels(self):
        anchors = self.get_calibrated_anchors()
        out = {}

        for vowel, (A, B, C) in TRIANGLES.items():
            if {A, B, C} <= anchors.keys():
                w = self._weights_for(vowel)
                # Get F0 values for the triangle vertices
                f0A = self.data[A].get("f0")
                f0B = self.data[B].get("f0")
                f0C = self.data[C].get("f0")

                # Interpolate F0 using the same barycentric weights
                f0 = None
                if all(isinstance(x, (int, float)) for x in (f0A, f0B, f0C)):
                    f0 = w[0] * f0A + w[1] * f0B + w[2] * f0C

                result = self.barycentric_interpolate(
                    w,
                    {"A": anchors[A], "B": anchors[B], "C": anchors[C]},
                )
                f1, f2 = map(float, result)

                out[vowel] = {
                    "f1": float(f1),
                    "f2": float(f2),
                    "f0": float(f0) if f0 is not None else None,
                    "confidence": 1.0,
                    "stability": 0.0,
                    "weight": 0.0,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                }

        return out

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

        # -----------------------------
        # 0. Basic sanity
        # -----------------------------
        if f1 is None or f2 is None or not np.isfinite(f1) or not np.isfinite(f2):
            return self._reject_capture(vowel, "invalid formant values")

        if f0 is not None and not np.isfinite(f0):
            f0 = None

        # -----------------------------
        # 1. Vowel-model plausibility
        # -----------------------------
        ok, reason = is_plausible_formants(
            f1, f2,
            voice_type=self.voice_type,
            vowel=vowel,
            calibrated=self.data,
        )
        if not ok:
            return self._reject_capture(vowel, reason)

        # -----------------------------
        # 2. Confidence gating (relaxed)
        # -----------------------------
        # Hybrid confidence is often 0.3–0.5 for back vowels.
        if confidence is not None and confidence < 0.25:
            return self._reject_capture(vowel, f"low confidence {confidence:.2f}")

        # -----------------------------
        # 3. First measurement
        # -----------------------------
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

        # -----------------------------
        # 4. Weighted averaging
        # -----------------------------
        old = self.data[vowel]
        old_f1 = old.get("f1", f1)
        old_f2 = old.get("f2", f2)
        old_f0 = old.get("f0", f0)
        if old_f0 is None or not np.isfinite(old_f0):
            old_f0 = None

        old_conf = old.get("confidence", 1.0)
        old_stab = old.get("stability", 0.0)
        old_weight = old.get("weight", 1.0)

        new_weight = old_weight + 1.0

        new_f1 = (old_f1 * old_weight + f1) / new_weight
        new_f2 = (old_f2 * old_weight + f2) / new_weight
        old_f0: float
        confidence: float
        # --- F0 update ---
        if f0 is not None and old_f0 is not None:
            f0_val = float(f0)
            old_f0_val = float(old_f0)
            new_f0 = (old_f0_val * old_weight + f0_val) / new_weight

        elif f0 is not None:
            new_f0 = float(f0)

        else:
            new_f0 = old_f0

        new_conf = (float(old_conf) * old_weight + float(confidence)) / new_weight
        new_stab = (float(old_stab) * old_weight + float(stability)) / new_weight

        self.data[vowel] = {
            "f1": new_f1,
            "f2": new_f2,
            "f0": new_f0,
            "confidence": new_conf,
            "stability": new_stab,
            "weight": new_weight,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        return True, False, f"Updated /{vowel}/ (weight={new_weight:.1f})"

    def _reject_capture(self, vowel: str, reason: str):
        self.increment_retry(vowel)
        return False, True, f"Rejected /{vowel}/: {reason}"

    # ---------------------------------------------------------
    # Save profile
    # ---------------------------------------------------------
    def save_profile(self):
        """
        Save the current calibration data as a profile.

        Tests expect:
          - The per-vowel entries under their vowel keys (e.g. "a")
          - All fields (f1,f2,f0,confidence,stability,weight,saved_at) preserved
          - A top-level "voice_type" field
        """
        if self.profile_manager is None:
            raise RuntimeError("profile_manager is not set on CalibrationSession")

        # Shallow copy of self.data so we don't mutate the original
        profile_data = dict(self.data)
        # Add interpolated vowels
        interpolated = self.compute_interpolated_vowels()
        for v, vals in interpolated.items():
            if v not in profile_data:
                profile_data[v] = vals
        profile_data["voice_type"] = self.voice_type

        # Prevent double suffixing
        if self.profile_name.endswith(f"_{self.voice_type}"):
            base_name = self.profile_name
        else:
            base_name = f"{self.profile_name}_{self.voice_type}"

        self.profile_manager.save_profile(base_name, profile_data)
        return base_name

    def retry_count(self, vowel: str) -> int:
        return self.retries_map.get(vowel, 0)

    def reset_retry(self, vowel: str) -> None:
        self.retries_map[vowel] = 0


def normalize_profile_for_save(user_formants, retries_map):
    out = {}

    for vowel, vals in user_formants.items():
        if vals is None:
            continue

        f1 = vals.get("f1")
        f2 = vals.get("f2")
        f0 = vals.get("f0")
        conf = vals.get("confidence", 0.0)
        stab = vals.get("stability", float("inf"))

        # Swap if reversed
        if f1 is not None and f2 is not None and f1 > f2:
            f1, f2 = f2, f1

        # Plausibility metadata (tests monkeypatch this)
        plausible, reason = is_plausible_formants(f1, f2, vowel=vowel)
        reason_text = reason if not plausible else "ok"

        retries = retries_map.get(vowel, 0)

        out[vowel] = {
            "f1": f1,
            "f2": f2,
            "f0": f0,
            "confidence": conf,
            "stability": stab,
            # tests expect NO weight field
            "retries": retries,
            "reason": reason_text,
            "saved_at": vals.get("saved_at", datetime.now(timezone.utc).isoformat()),
            "source": "calibration",
        }

    return out
