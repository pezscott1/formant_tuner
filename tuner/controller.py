# tuner/controller.py
import numpy as np

from analysis.engine import FormantAnalysisEngine
from analysis.smoothing import PitchSmoother, MedianSmoother, LabelSmoother
from tuner.live_analyzer import LiveAnalyzer
from tuner.profile_controller import ProfileManager


class Tuner:
    def __init__(
        self,
        voice_type="bass",
        profiles_dir="profiles",
        sample_rate=48000,
        engine=None,
    ):
        # Engine: allow injection for tests, otherwise construct
        self.engine = engine or FormantAnalysisEngine(voice_type=voice_type)

        # Smoothers
        self.pitch_smoother = PitchSmoother(sr=sample_rate, min_confidence=0.25)
        self.formant_smoother = MedianSmoother(min_confidence=0.25)
        self.label_smoother = LabelSmoother(min_confidence=0.25)

        # Live analyzer
        self.live_analyzer = LiveAnalyzer(
            engine=self.engine,
            pitch_smoother=self.pitch_smoother,
            formant_smoother=self.formant_smoother,
            label_smoother=self.label_smoother,
        )

        # Profile manager
        self.profile_manager = ProfileManager(
            profiles_dir=profiles_dir,
            analyzer=self.engine,
        )

        self.voice_type = voice_type
        self.active_profile = None

    # ---------------------------------------------------------
    # Profile operations
    # ---------------------------------------------------------
    def load_profile(self, base_name):
        """
        Load a profile and apply it to the engine.
        """

        # Step 1: Apply profile
        applied = self.profile_manager.apply_profile(base_name)
        self.active_profile = applied

        # If apply_profile returned a string → treat as error
        if isinstance(applied, str) and applied != base_name:
            # Reset engine.voice_type to tuner.voice_type
            self.engine.voice_type = self.voice_type
            return applied

        # Step 2: Load raw JSON
        raw = self.profile_manager.load_profile_json(base_name)

        # Step 3: Extract (f1, f2, f3, conf, stab)
        extracted = self.profile_manager.extract_formants(raw)

        # Step 4: Build user_formants with default conf/stab
        cleaned = {
            vowel: (f1, f2, f3, 0.0, float("inf"))
            for vowel, (f1, f2, f3, *_) in extracted.items()
        }

        # Step 5: Assign to engine
        self.engine.user_formants = cleaned

        return applied

    def list_profiles(self):
        return self.profile_manager.list_profiles()

    def delete_profile(self, base_name):
        self.profile_manager.delete_profile(base_name)

    # ---------------------------------------------------------
    # Analysis interface
    # ---------------------------------------------------------
    def poll_latest_processed(self):
        """
        Read latest raw frame from engine, pass through LiveAnalyzer,
        then (optionally) classify the vowel using the active profile.
        """
        raw = self.engine.get_latest_raw()
        if raw is None:
            return None

        processed = self.live_analyzer.process_raw(raw)
        if processed is None:
            return None

        # If no profile loaded, just return smoothed processed values
        if not self.active_profile or "formants" not in processed:
            return processed

        # Reject unstable frames (optional but recommended)
        if not processed.get("stable", True):
            processed["profile_vowel"] = None
            processed["profile_confidence"] = 0.0
            return processed

        f1, f2, _ = processed["formants"]
        vowel, confidence = self._classify_vowel_from_profile(f1, f2)

        processed["profile_vowel"] = vowel
        processed["profile_confidence"] = confidence

        return processed

    # ---------------------------------------------------------
    # Calibration-aware vowel classifier
    # ---------------------------------------------------------
    def _classify_vowel_from_profile(self, f1, f2):
        if f1 is None or f2 is None:
            return None, 0.0

        profile = self.active_profile
        if not profile:
            return None, 0.0

        # Collect centroids
        centroids = {
            vowel: (data.get("f1"), data.get("f2"))
            for vowel, data in profile.items()
            if isinstance(data, dict)
            and data.get("f1") is not None
            and data.get("f2") is not None
        }

        if not centroids:
            return None, 0.0

        # Normalize input using profile mean/std
        f1_vals = [v[0] for v in centroids.values()]
        f2_vals = [v[1] for v in centroids.values()]

        f1_mean, f1_std = float(np.mean(f1_vals)), float(np.std(f1_vals) + 1e-6)
        f2_mean, f2_std = float(np.mean(f2_vals)), float(np.std(f2_vals) + 1e-6)

        nf1 = (f1 - f1_mean) / f1_std
        nf2 = (f2 - f2_mean) / f2_std

        # Compute normalized distance to each vowel centroid
        best_vowel = None
        best_dist = float("inf")

        for vowel, (pf1, pf2) in centroids.items():
            d1 = (pf1 - f1_mean) / f1_std
            d2 = (pf2 - f2_mean) / f2_std
            dist = (nf1 - d1) ** 2 + (nf2 - d2) ** 2
            if dist < best_dist:
                best_dist = dist
                best_vowel = vowel

        confidence = float(np.exp(-best_dist))
        return best_vowel, confidence
