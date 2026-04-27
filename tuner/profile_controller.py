# tuner/profile_controller.py
import logging
import os
import json
from pathlib import Path
from datetime import datetime, UTC
import platform

logger = logging.getLogger(__name__)

KNOWN_VOICE_TYPES = frozenset({
    "bass", "baritone", "tenor", "alto",
    "mezzo", "mezzo-soprano", "contralto", "soprano",
})

# Generous plausible bounds for classical/operatic voice formants
_F1_MIN, _F1_MAX = 50.0, 2000.0
_F2_MIN, _F2_MAX = 200.0, 4000.0


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ProfileManager:
    ACTIVE_FILE = "active_profile.json"

    def __init__(self, profiles_dir=None, analyzer=None):
        if profiles_dir is None:
            # Production default
            self.profiles_dir = str(self.get_profile_dir())
        else:
            # Tests and custom paths
            self.profiles_dir = str(profiles_dir)

        self.analyzer = analyzer

        if self.profiles_dir != "":
            os.makedirs(self.profiles_dir, exist_ok=True)

        self.active_profile_name = None
        self._load_active_profile()

    def _join(self, filename: str) -> str:
        """Join filename to profiles_dir, handling empty-dir case."""
        return filename if self.profiles_dir == "" \
            else os.path.join(self.profiles_dir, filename)

    # ---------------------------------------------------------
    # Listing + name conversion
    # ---------------------------------------------------------
    @staticmethod
    def get_profile_dir():
        system = platform.system()

        if system == "Windows":
            base = Path(os.getenv("APPDATA")) / "Tuner" / "profiles"
        elif system == "Darwin":  # macOS
            base = Path.home() / "Library" / "Application Support" / "Tuner" / "profiles"
        else:  # Linux and others
            base = Path.home() / ".config" / "tuner" / "profiles"

        base.mkdir(parents=True, exist_ok=True)
        return base

    def list_profiles(self):
        """Return a sorted list of base profile names."""
        return sorted(
            fn[:-len("_profile.json")]
            for fn in os.listdir(self.profiles_dir)
            if fn.endswith("_profile.json") and fn != self.ACTIVE_FILE
        )

    @staticmethod
    def display_name(base):
        return base.replace("_", " ")

    # ---------------------------------------------------------
    # Existence checks
    # ---------------------------------------------------------
    def profile_exists(self, base):
        path = os.path.join(self.profiles_dir, f"{base}_profile.json")
        return os.path.exists(path)

    # ---------------------------------------------------------
    # Saving profiles
    # ---------------------------------------------------------
    def save_profile(self, base, data, model_bytes=None):
        if "voice_type" not in data:
            data["voice_type"] = self.analyzer.voice_type

        json_path = os.path.join(self.profiles_dir, f"{base}_profile.json")
        model_path = json_path.replace("_profile.json", "_model.pkl")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        if model_bytes is not None:
            with open(model_path, "wb") as f:
                f.write(model_bytes)

        logger.info("Profile saved: %s", base)
        self.set_active_profile(base)

    # ---------------------------------------------------------
    # Active profile tracking
    # ---------------------------------------------------------
    def set_active_profile(self, name):
        self.active_profile_name = name
        # Only write active_profile.json when profiles_dir is not empty
        if self.profiles_dir == "":
            return
        path = self._join(self.ACTIVE_FILE)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"active": name}, f)

    def _load_active_profile(self):
        path = self._join(self.ACTIVE_FILE)
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.active_profile_name = data.get("active")
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not load active profile record: %s", e)
            self.active_profile_name = None

    # ---------------------------------------------------------
    # Apply profile to analyzer
    # ---------------------------------------------------------
    def apply_profile(self, base_name: str):
        """
        Load profile JSON, update analyzer.voice_type and analyzer.user_formants.
        Returns base_name on success (even if JSON is empty).
        """
        raw = self.load_profile_json(base_name)

        # Always return base_name, even if raw is {}
        voice_type = raw.get("voice_type")
        # Fallback: infer from profile base name
        if not voice_type:
            inferred_map = {
                "alpha": "tenor",
                "bass": "bass",
                "baritone": "baritone",
                "tenor": "tenor",
                "alto": "alto",
                "soprano": "soprano",
            }
            voice_type = inferred_map.get(base_name, None)

        if voice_type and voice_type not in KNOWN_VOICE_TYPES:
            logger.warning(
                "Profile '%s' has unknown voice_type '%s'; keeping current",
                base_name, voice_type,
            )
            voice_type = None

        if voice_type:
            self.analyzer.voice_type = voice_type

        cal = raw.get("calibrated_vowels", {})
        if isinstance(cal, list):
            # Convert list of vowel names into a dict of entries from raw
            cal = {v: raw.get(v, {}) for v in cal}

        interp = raw.get("interpolated_vowels", {})
        if isinstance(interp, list):
            interp = {}
        # Merge them into the analyzer's vowel set
        merged = {**cal, **interp}

        # Normalize entries (f1, f2, f0, confidence, stability)
        user_formants = self.extract_formants(merged)

        # Store full profile for UI and vowel map
        self.analyzer.active_profile = {
            **user_formants,
            "calibrated_vowels": cal,
            "interpolated_vowels": interp,
            "voice_type": voice_type,
        }
        # Store merged vowel dictionary for analysis
        self.analyzer.user_formants = user_formants
        # Apply to analyzer
        if hasattr(self.analyzer, "set_user_formants"):
            self.analyzer.set_user_formants(user_formants)
        else:
            self.analyzer.user_formants = user_formants
        self.set_active_profile(base_name)
        return base_name

    # ---------------------------------------------------------
    # Public loader wrapper
    # ---------------------------------------------------------
    def load_profile(self, base: str) -> dict:
        """Public wrapper for loading a profile by base name."""
        return self.load_profile_json(base)

    # ---------------------------------------------------------
    # Internal JSON loader
    # ---------------------------------------------------------

    def load_profile_json(self, base_name):
        # CASE 1: Path object
        if isinstance(base_name, Path):
            try:
                with open(base_name, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Could not load profile from path %s: %s", base_name, e)
                return {}
            if not isinstance(data, dict):
                logger.warning("Profile at %s is not a JSON object; ignoring", base_name)
                return {}
            return data

        # CASE 2: string profile name
        profile_path = self._join(f"{base_name}_profile.json")

        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not load profile '%s': %s", base_name, e)
            data = {}

        if not isinstance(data, dict):
            logger.warning("Profile '%s' is not a JSON object; ignoring", base_name)
            return {}

        return data

    # ---------------------------------------------------------
    # Extract formants from rich profile JSON
    # ---------------------------------------------------------

    @staticmethod
    def extract_formants(raw_dict):
        out = {}
        if not isinstance(raw_dict, dict):
            return out
        for vowel, entry in raw_dict.items():
            if vowel in ("voice_type", "calibrated_vowels", "interpolated_vowels"):
                continue
            if not isinstance(entry, (dict, list, tuple)):
                continue
            norm = _normalize_profile_entry(entry)
            if not isinstance(norm, dict):
                continue
            f1 = norm.get("f1")
            f2 = norm.get("f2")
            # Require numeric f1/f2
            try:
                if f1 is not None:
                    f1 = float(f1)
                if f2 is not None:
                    f2 = float(f2)
            except (ValueError, TypeError):
                continue
            if f1 is not None and not (_F1_MIN <= f1 <= _F1_MAX):
                logger.warning("Skipping vowel '%s': F1=%.1f outside plausible range", vowel, f1)
                continue
            if f2 is not None and not (_F2_MIN <= f2 <= _F2_MAX):
                logger.warning("Skipping vowel '%s': F2=%.1f outside plausible range", vowel, f2)
                continue
            f0 = norm.get("f0")
            if f0 is None:
                f0 = norm.get("f3")
            confidence = float(norm.get("confidence", 0.0))
            stability = float(norm.get("stability", float("inf")))
            out[vowel] = {
                "f1": f1,
                "f2": f2,
                "f0": f0,
                "confidence": confidence,
                "stability": stability,
            }
        return out

    # ---------------------------------------------------------
    # Deletion
    # ---------------------------------------------------------
    def delete_profile(self, base):
        json_path = self._join(f"{base}_profile.json")
        model_path = json_path.replace("_profile.json", "_model.pkl")

        if os.path.exists(json_path):
            os.remove(json_path)
        if os.path.exists(model_path):
            os.remove(model_path)

        if self.active_profile_name == base:
            self.active_profile_name = None
            active_path = os.path.join(self.profiles_dir, self.ACTIVE_FILE)
            if os.path.exists(active_path):
                os.remove(active_path)


def _normalize_profile_entry(entry):
    if isinstance(entry, dict):
        return entry

    if isinstance(entry, (tuple, list)):
        vals = list(entry) + [None, None, None]
        f1, f2, f3, f0, conf, stab = (vals + [None, None])[:6]
        return {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f0": f0,
            "confidence": conf if conf is not None else 0.0,
            "stability": stab if stab is not None else float("inf"),
        }
    return {"f1": None, "f2": None, "f3": None, "f0": None}
