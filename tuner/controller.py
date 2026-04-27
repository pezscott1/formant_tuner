# tuner/controller.py
import logging
import numpy as np

logger = logging.getLogger(__name__)

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
        self.active_profile = None

        # Centroid normalization cache: avoids recomputing mean/std every frame
        self._centroid_cache_key = None   # id(profile)
        self._centroid_cache = None       # (norm_centroids, f1_mean, f1_std, f2_mean, f2_std)
        self._fallback_profiles: dict[str, dict] = {}  # voice_type -> stable profile dict

    # ---------------------------------------------------------
    # Engine / analyzer delegation
    # ---------------------------------------------------------
    @property
    def voice_type(self) -> str:
        return self.engine.voice_type

    @voice_type.setter
    def voice_type(self, value: str):
        self.engine.voice_type = value

    @property
    def vowel_hint(self):
        return self.engine.vowel_hint

    @vowel_hint.setter
    def vowel_hint(self, value):
        self.engine.vowel_hint = value

    @property
    def user_formants(self) -> dict:
        return self.engine.user_formants

    def set_user_formants(self, user_formants: dict):
        self.engine.user_formants = user_formants
        self.live_analyzer.user_formants = user_formants

    def pause_analyzer(self):
        self.live_analyzer.pause()

    def resume_analyzer(self):
        self.live_analyzer.resume()

    def reset_analyzer(self):
        self.live_analyzer.reset()

    def submit_audio(self, segment):
        self.live_analyzer.submit_audio_segment(segment)

    # ---------------------------------------------------------
    # Profile operations
    # ---------------------------------------------------------
    def load_profile(self, base_name):
        applied = self.profile_manager.apply_profile(base_name)

        if isinstance(applied, str) and applied != base_name:
            self.engine.voice_type = self.voice_type
            return applied

        raw = self.profile_manager.load_profile_json(base_name)
        extracted = self.profile_manager.extract_formants(raw)
        self.active_profile = extracted

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
            self.engine.vowel_hint = vowel

        return processed

    # ---------------------------------------------------------
    # Calibration-aware vowel classifier
    # ---------------------------------------------------------
    def _classify_vowel_from_profile(self, f1, f2):
        if f1 is None or f2 is None:
            return None, 0.0

        profile = self._get_active_profile()
        cache = self._get_centroid_cache(profile)
        if cache is None:
            return None, 0.0

        norm_centroids, f1_mean, f1_std, f2_mean, f2_std = cache
        nf1 = (f1 - f1_mean) / f1_std
        nf2 = (f2 - f2_mean) / f2_std

        best_vowel = None
        best_dist = float("inf")
        for vowel, (d1, d2) in norm_centroids.items():
            dist = (nf1 - d1) ** 2 + (nf2 - d2) ** 2
            if dist < best_dist:
                best_dist = dist
                best_vowel = vowel

        return best_vowel, float(np.exp(-best_dist))

    def _get_active_profile(self) -> dict:
        """Return the active profile, or a stable fallback dict for the current voice type."""
        if self.active_profile:
            return self.active_profile
        if self.voice_type not in self._fallback_profiles:
            from analysis.vowel_data import VOWEL_CENTERS
            centers = VOWEL_CENTERS.get(self.voice_type, {})
            self._fallback_profiles[self.voice_type] = {
                v: {"f1": c[0], "f2": c[1]} for v, c in centers.items()
            }
        return self._fallback_profiles[self.voice_type]

    def _get_centroid_cache(self, profile: dict):
        """Return cached (norm_centroids, f1_mean, f1_std, f2_mean, f2_std), building if stale."""
        if self._centroid_cache_key == id(profile) and self._centroid_cache is not None:
            return self._centroid_cache

        centroids = {
            vowel: (data.get("f1"), data.get("f2"))
            for vowel, data in profile.items()
            if isinstance(data, dict)
            and data.get("f1") is not None
            and data.get("f2") is not None
        }
        if not centroids:
            return None

        f1_vals = [v[0] for v in centroids.values()]
        f2_vals = [v[1] for v in centroids.values()]
        f1_mean = float(np.mean(f1_vals))
        f1_std = float(np.std(f1_vals) + 1e-6)
        f2_mean = float(np.mean(f2_vals))
        f2_std = float(np.std(f2_vals) + 1e-6)

        norm_centroids = {
            vowel: ((pf1 - f1_mean) / f1_std, (pf2 - f2_mean) / f2_std)
            for vowel, (pf1, pf2) in centroids.items()
        }

        self._centroid_cache = (norm_centroids, f1_mean, f1_std, f2_mean, f2_std)
        self._centroid_cache_key = id(profile)
        return self._centroid_cache

    def update_tolerance(self, text: str) -> int:
        try:
            value = int(text)
            if value <= 0:
                raise ValueError
            self.current_tolerance = value
        except (ValueError, TypeError):
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
            logger.exception("Failed to start mic stream")
            self.stream = None
            return False

    def stop_mic(self):
        stream = getattr(self, "stream", None)
        if stream is None:
            return False

        try:
            stream.stop()
            stream.close()
            self.engine.vowel_hint = None
        except Exception:
            logger.warning("Error stopping mic stream", exc_info=True)

        self.stream = None
        self.live_analyzer.stop_worker()
        self.live_analyzer.reset()
        return True
