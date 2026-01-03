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
        self.stream = None
        self.current_tolerance = None
        # Engine: allow injection for tests, otherwise construct
        self.engine = engine or FormantAnalysisEngine(
            voice_type=voice_type, use_hybrid=True,)
        self.engine.profile_classifier = self._classify_vowel_from_profile
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
        self.last_vowel = None
        self.voice_type = voice_type
        self.active_profile = None

    # ---------------------------------------------------------
    # Profile operations
    # ---------------------------------------------------------
    def load_profile(self, base_name):
        applied = self.profile_manager.apply_profile(base_name)

        # If apply_profile returned an error string, bail out
        if isinstance(applied, str) and applied != base_name:
            self.engine.voice_type = self.voice_type
            return applied

        # Load raw JSON (for our own use, e.g., classifier)
        raw = self.profile_manager.load_profile_json(base_name)

        # Extract formants (dict of dicts)
        extracted = self.profile_manager.extract_formants(raw)

        # Store active profile as the dict of centroids
        self.active_profile = extracted
        print("ACTIVE PROFILE:", type(self.active_profile), self.active_profile)

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
        processed = self.live_analyzer.get_latest_processed()
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

        if vowel is not None and confidence > 0.2:
            # Update controller state
            self.last_vowel = vowel
            # Crucial: drive the hybrid selector hint
            self.engine.vowel_hint = vowel

        return processed

    # ---------------------------------------------------------
    # Calibration-aware vowel classifier
    # ---------------------------------------------------------
    def _classify_vowel_from_profile(self, f1, f2):
        if f1 is None or f2 is None:
            return None, 0.0

        if not self.active_profile:
            # fallback to reference vowel centers
            from analysis.vowel_data import VOWEL_CENTERS
            centers = VOWEL_CENTERS.get(self.voice_type, {})
            # convert to same dict format as calibrated profiles
            profile = {v: {"f1": c[0], "f2": c[1]} for v, c in centers.items()}
        else:
            profile = self.active_profile

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

    def update_tolerance(self, text: str) -> int:
        try:
            value = int(text)
            if value <= 0:
                raise ValueError
            self.current_tolerance = value
        except Exception:
            pass
        return self.current_tolerance

    def start_mic(self, stream_factory):
        if getattr(self, "stream", None) is not None:
            return False

        try:
            # Fresh smoothing state for each session
            self.live_analyzer.reset()

            self.stream = stream_factory()
            self.stream.start()
            self.live_analyzer.start_worker()
            return True
        except Exception:
            self.stream = None
            return False

    def stop_mic(self):
        stream = getattr(self, "stream", None)
        if stream is None:
            return False

        try:
            stream.stop()
            stream.close()
            self.last_vowel = None
            self.engine.vowel_hint = None
        except Exception:
            pass

        self.stream = None
        self.live_analyzer.stop_worker()
        self.live_analyzer.reset()
        return True
