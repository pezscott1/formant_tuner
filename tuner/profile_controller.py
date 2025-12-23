import os
import json


class ProfileManager:
    """
    Handles all profile persistence for the tuner and calibration system.

    Responsibilities:
      - List available profiles
      - Convert base <-> display names
      - Save new profiles (JSON + optional model)
      - Load/apply profiles into the analyzer
      - Track the active profile (active_profile.json)
      - Delete profiles
    """

    ACTIVE_FILE = "active_profile.json"

    def __init__(self, profiles_dir, analyzer):
        self.profiles_dir = profiles_dir
        self.analyzer = analyzer
        os.makedirs(self.profiles_dir, exist_ok=True)

        self.active_profile_name = None
        self._load_active_profile()

    # ---------------------------------------------------------
    # Listing + name conversion
    # ---------------------------------------------------------
    def list_profiles(self):
        """Return a sorted list of base profile names."""
        return sorted(
            fn[:-len("_profile.json")]
            for fn in os.listdir(self.profiles_dir)
            if fn.endswith("_profile.json") and fn != self.ACTIVE_FILE
        )

    @staticmethod
    def display_name(base):
        """Convert 'tenor_user1' -> 'tenor user1'."""
        return base.replace("_", " ")

    @staticmethod
    def base_from_display(display):
        """Convert 'tenor user1' -> 'tenor_user1'."""
        if display.startswith("➕"):
            return None
        return display.replace(" ", "_")

    # Alias for UI compatibility
    base_name_from_display = base_from_display

    # ---------------------------------------------------------
    # Existence checks
    # ---------------------------------------------------------
    def profile_exists(self, base):
        """Return True if <base>_profile.json exists."""
        path = os.path.join(self.profiles_dir, f"{base}_profile.json")
        return os.path.exists(path)

    # ---------------------------------------------------------
    # Saving profiles (used by calibration)
    # ---------------------------------------------------------
    def save_profile(self, base, data, model_bytes=None):
        """
        Save a profile:
          - data: dict to write as JSON (rich format)
          - model_bytes: optional binary model data
        """

        # ✅ Ensure voice_type is stored in the JSON
        #    This makes it recoverable when the profile is reloaded.
        if "voice_type" not in data:
            data["voice_type"] = self.analyzer.voice_type

        json_path = os.path.join(self.profiles_dir, f"{base}_profile.json")
        model_path = json_path.replace("_profile.json", "_model.pkl")

        # Write JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Write model if provided
        if model_bytes is not None:
            with open(model_path, "wb") as f:
                f.write(model_bytes)

        # Update active profile
        self.set_active_profile(base)

    # ---------------------------------------------------------
    # Active profile tracking
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Active profile tracking
    # ---------------------------------------------------------
    def set_active_profile(self, name, *, notify=False):
        """
        Persist the active profile name to active_profile.json.
        UI popups are handled by the controller, not here.
        """
        self.active_profile_name = name
        path = os.path.join(self.profiles_dir, self.ACTIVE_FILE)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"active": name}, f)

    def apply_profile(self, base):
        """
        Load a profile into the analyzer WITHOUT triggering UI popups.
        """
        self.active_profile_name = base
        profile_path = os.path.join(self.profiles_dir, f"{base}_profile.json")

        raw = self._load_profile_json(profile_path)

        # Load voice type
        voice_type = raw.get("voice_type", "bass")
        self.analyzer.voice_type = voice_type

        # Load formants
        normalized = self._extract_formants(raw)
        self.analyzer.set_user_formants(normalized)

        # Persist active profile silently
        self.set_active_profile(base, notify=False)

        return base

    def _load_active_profile(self):
        """Load active profile from disk if present."""
        path = os.path.join(self.profiles_dir, self.ACTIVE_FILE)
        if not os.path.exists(path):
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.active_profile_name = data.get("active")
        except Exception:
            self.active_profile_name = None

    # ---------------------------------------------------------
    # Internal JSON loader
    # ---------------------------------------------------------
    # noinspection PyMethodMayBeStatic
    def _load_profile_json(self, path):
        """Load a profile JSON file and return its dict."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    # ---------------------------------------------------------
    # Extract formants from rich profile JSON
    # ---------------------------------------------------------
    # noinspection PyMethodMayBeStatic
    def _extract_formants(self, raw_dict):
        """
        Convert rich profile entries like:
            { "a": { "f1":..., "f2":..., "f0":..., ... }, ... }
        into:
            { "a": (f1, f2, f0), ... }
        """
        out = {}
        if not isinstance(raw_dict, dict):
            return out

        for vowel, entry in raw_dict.items():
            if not isinstance(entry, dict):
                continue

            f1 = entry.get("f1")
            f2 = entry.get("f2")
            f0 = entry.get("f0")  # ✅ pitch, not f3

            # Store as (f1, f2, f0)
            out[vowel] = (f1, f2, f0)

        return out

    # ---------------------------------------------------------
    # Deletion
    # ---------------------------------------------------------
    def delete_profile(self, base):
        """Delete the JSON + model file for a profile."""
        json_path = os.path.join(self.profiles_dir, f"{base}_profile.json")
        model_path = json_path.replace("_profile.json", "_model.pkl")

        if os.path.exists(json_path):
            os.remove(json_path)
        if os.path.exists(model_path):
            os.remove(model_path)

        # If it was active, clear active_profile.json
        if self.active_profile_name == base:
            self.active_profile_name = None
            active_path = os.path.join(self.profiles_dir, self.ACTIVE_FILE)
            if os.path.exists(active_path):
                os.remove(active_path)
