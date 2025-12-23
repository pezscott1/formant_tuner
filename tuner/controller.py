# tuner/controller.py
import numpy as np

from analysis.engine import FormantAnalysisEngine
from analysis.smoothing import PitchSmoother, MedianSmoother, LabelSmoother
from tuner.live_analyzer import LiveAnalyzer
from tuner.profile_controller import ProfileManager


class Tuner:
    def __init__(
            self, engine=None, voice_type="bass", profiles_dir="profiles", sample_rate=44100):
        # Shared DSP engine
        self.engine = engine or FormantAnalysisEngine(voice_type=voice_type)

        # Smoothers
        self.pitch_smoother = PitchSmoother(sr=sample_rate)
        self.formant_smoother = MedianSmoother()
        self.label_smoother = LabelSmoother()

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
        """Load a profile and apply it to the engine."""
        self.active_profile = self.profile_manager.apply_profile(base_name)

        if isinstance(self.active_profile, str):
            # ProfileManager.apply_profile returns base name
            self.engine.voice_type = self.voice_type

        return self.active_profile

    def list_profiles(self):
        return self.profile_manager.list_profiles()

    def delete_profile(self, base_name):
        self.profile_manager.delete_profile(base_name)

    # ---------------------------------------------------------
    # Audio control placeholders (no-op)
    # ---------------------------------------------------------
    def start(self):
        """Deprecated: mic control is handled by TunerWindow."""
        return

    def stop(self):
        """Deprecated: mic control is handled by TunerWindow."""
        return

    # ---------------------------------------------------------
    # Analysis interface (optional, calibration-aware)
    # ---------------------------------------------------------
    def poll_latest_processed(self):
        """
        Read latest raw frame from engine, pass through LiveAnalyzer,
        then (optionally) classify the vowel using the active profile.

        Returns processed dict or None.
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

        f1, f2, _ = processed["formants"]
        vowel, confidence = self._classify_vowel_from_profile(f1, f2)

        processed["profile_vowel"] = vowel
        processed["profile_confidence"] = confidence

        # Note: no extra label smoothing here; that lives in LiveAnalyzer.
        return processed

    # ---------------------------------------------------------
    # Calibration-aware vowel classifier
    # ---------------------------------------------------------
    def _classify_vowel_from_profile(self, f1, f2):
        """
        Use the active profile as vowel centroids.
        Compute distance in normalized F1/F2 space.
        Returns (vowel, confidence).
        """
        if f1 is None or f2 is None:
            return None, 0.0

        profile = self.active_profile
        if not profile:
            return None, 0.0

        # Collect centroids
        centroids = {}
        for vowel, data in profile.items():
            if not isinstance(data, dict):
                continue
            pf1 = data.get("f1")
            pf2 = data.get("f2")
            if pf1 is None or pf2 is None:
                continue
            centroids[vowel] = (pf1, pf2)

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

        # Convert distance to confidence (simple heuristic)
        confidence = float(np.exp(-best_dist))

        return best_vowel, confidence
