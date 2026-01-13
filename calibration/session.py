# calibration/session.py
from datetime import datetime, timezone
from analysis.plausibility import is_plausible_formants
import numpy as np
import unicodedata
from analysis.vowel_data import TRIANGLES, TRIANGLE_WEIGHTS


def _norm(v: str) -> str:
    return unicodedata.normalize("NFC", v)


class CalibrationSession:
    def __init__(self, profile_name, voice_type, vowels,
                 profile_manager=None, existing_profile=None):

        self.profile_name = profile_name
        self.voice_type = voice_type
        self.vowels = vowels
        self.profile_manager = profile_manager

        # Core data structures
        # Note: keys in self.data may be in any Unicode form; lookups normalize.
        self.data = {}          # vowel → dict of f1,f2,f0,confidence,stability,weight
        self.retries_map = {}   # vowel → retry count
        self.calibrated_vowels = set()
        self.interpolated_vowels = set()

        # Load existing profile if present (backwards-compatible storage only)
        if isinstance(existing_profile, dict):
            for vowel, entry in existing_profile.items():
                if vowel == "voice_type":
                    continue
                if isinstance(entry, dict):
                    self.data[vowel] = entry.copy()

    def _weights_for(self, vowel):
        return TRIANGLE_WEIGHTS.get(vowel, (1 / 3, 1 / 3, 1 / 3))

    def _get_data_entry(self, vowel):
        """
        Lookup a data entry by vowel key, trying exact key then normalized key.
        Returns the dict or None.
        """
        if vowel in self.data:
            return self.data[vowel]
        n = _norm(vowel)
        if n in self.data:
            return self.data[n]
        # also try decomposed form if original was normalized
        # (this covers both directions)
        for k in (vowel, n):
            if k in self.data:
                return self.data[k]
        return None

    def get_calibrated_anchors(self):
        """
        Return a dict of normalized vowel → (f1, f2) for calibrated anchors.

        Strict rule: only vowels explicitly present in self.calibrated_vowels
        are considered anchors for interpolation. Keys are normalized to NFC.
        """
        anchors = {}
        for v in self.calibrated_vowels:
            vn = _norm(v)
            entry = self._get_data_entry(v)
            if not isinstance(entry, dict):
                continue
            f1 = entry.get("f1")
            f2 = entry.get("f2")
            if (f1 is not None and f2 is not None
                    and np.isfinite(f1) and np.isfinite(f2)):
                anchors[vn] = (float(f1), float(f2))
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
        """
        Compute interpolated vowels based on TRIANGLES and TRIANGLE_WEIGHTS.

        - Uses only self.calibrated_vowels as anchors (triangle-only).
        - Normalizes keys (NFC) for matching/lookups.
        - Returns a dict mapping the original TRIANGLES keys -> interpolated entry.
        - Does NOT mutate self.data or self.calibrated_vowels.
        """
        anchors = self.get_calibrated_anchors()  # normalized keys
        out = {}

        # Precompute normalized calibrated set for quick membership checks
        calibrated_norm = {_norm(v) for v in self.calibrated_vowels}

        for vowel, (A, B, C) in TRIANGLES.items():
            # keep the original triangle key for output
            v_key = vowel

            # normalized forms for matching
            v_norm = _norm(vowel)
            A_n, B_n, C_n = _norm(A), _norm(B), _norm(C)

            # Skip if vowel explicitly calibrated (compare normalized forms)
            if v_norm in calibrated_norm:
                continue

            # Only interpolate if the triangle's vertices are present in anchors
            if {A_n, B_n, C_n} <= set(anchors.keys()):
                w = self._weights_for(vowel)

                # Get F0 values for the triangle vertices (may be None)
                f0A = self._get_data_entry(A) and self._get_data_entry(A).get("f0")
                f0B = self._get_data_entry(B) and self._get_data_entry(B).get("f0")
                f0C = self._get_data_entry(C) and self._get_data_entry(C).get("f0")

                # Interpolate F0 using the same barycentric
                # weights if all present and finite
                f0 = None
                if all(isinstance(x, (int, float))
                       and np.isfinite(x) for x in (f0A, f0B, f0C)):
                    f0 = w[0] * f0A + w[1] * f0B + w[2] * f0C

                result = self.barycentric_interpolate(
                    w,
                    {"A": anchors[A_n], "B": anchors[B_n], "C": anchors[C_n]},
                )
                f1, f2 = map(float, result)

                # store under the original triangle key
                # so tests comparing to TRIANGLES.keys() match
                out[v_key] = {
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
            self.calibrated_vowels.add(vowel)
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
        self.calibrated_vowels.add(vowel)
        return True, False, f"Updated /{vowel}/ (weight={new_weight:.1f})"

    def _reject_capture(self, vowel: str, reason: str):
        self.increment_retry(vowel)
        return False, True, f"Rejected /{vowel}/: {reason}"

    # ---------------------------------------------------------
    # Save profile
    # ---------------------------------------------------------
    def save_profile(self):
        """
        Save the profile using the profile_manager.

        - Uses compute_interpolated_vowels() to determine interpolated entries.
        - Does not mutate self.data.
        - Saves calibrated_vowels exactly from self.calibrated_vowels.
        - Saves interpolated_vowels exactly from the computed interpolation.
        """
        if not self.profile_manager:
            raise RuntimeError("No profile manager")

        base = f"{self.profile_name}_{self.voice_type}"

        interp = self.compute_interpolated_vowels()

        # calibrated vowels are exactly the explicit set
        # (use original keys from self.calibrated_vowels)
        calibrated = sorted(self.calibrated_vowels)

        # interpolated vowels are exactly the computed ones (keys are normalized)
        interpolated = {v: interp[v] for v in sorted(interp.keys())}

        # Backwards compatibility: if no calibrated_vowels,
        # treat all data entries as calibrated
        if not calibrated:
            calibrated = sorted(v for v, e in self.data.items()
                                if isinstance(e, dict))
            interpolated = {}

        profile_data = {
            "calibrated_vowels": {v: self._get_data_entry(v) for v in calibrated},
            "interpolated_vowels": interpolated,
            "voice_type": self.voice_type,
        }

        self.profile_manager.save_profile(base, profile_data)
        return base

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
            "retries": retries,
            "reason": reason_text,
            "saved_at": vals.get("saved_at", datetime.now(timezone.utc).isoformat()),
            "source": "calibration",
        }

    return out
