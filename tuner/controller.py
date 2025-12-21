# tuner/controller.py
from analysis.engine import FormantAnalysisEngine
from analysis.smoothing import PitchSmoother, MedianSmoother, LabelSmoother

from tuner.live_analyzer import LiveAnalyzer
from tuner.profile_controller import ProfileManager


class Tuner:
    """
    High-level controller for the live tuner.

    Responsibilities:
      - Own the FormantAnalysisEngine
      - Own the LiveAnalyzer (smoothing + plausibility)
      - Own the ProfileManager
      - Expose a simple interface for profiles + analysis

    NOTE:
      This class NO LONGER owns any mic / audio pipeline.
      Audio is handled entirely by TunerWindow via sounddevice.InputStream,
      which feeds self.engine.process_frame(...).
    """

    def __init__(self, voice_type="bass", profiles_dir="profiles"):
        # -----------------------------------------------------
        # Core DSP engine
        # -----------------------------------------------------
        self.engine = FormantAnalysisEngine(voice_type=voice_type)

        # -----------------------------------------------------
        # Smoothers
        # -----------------------------------------------------
        self.pitch_smoother = PitchSmoother()
        self.formant_smoother = MedianSmoother()
        self.label_smoother = LabelSmoother()

        # -----------------------------------------------------
        # Live analyzer (pure logic)
        # -----------------------------------------------------
        self.live_analyzer = LiveAnalyzer(
            engine=self.engine,
            pitch_smoother=self.pitch_smoother,
            formant_smoother=self.formant_smoother,
            label_smoother=self.label_smoother,
        )

        # -----------------------------------------------------
        # Profile manager
        # -----------------------------------------------------
        self.profile_manager = ProfileManager(
            profiles_dir=profiles_dir,
            analyzer=self.engine,
        )

        # State
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
        """
        Deprecated: mic control is handled by TunerWindow.
        Kept for compatibility; does nothing.
        """
        return

    def stop(self):
        """
        Deprecated: mic control is handled by TunerWindow.
        Kept for compatibility; does nothing.
        """
        return

    # ---------------------------------------------------------
    # Analysis interface
    # ---------------------------------------------------------
    def poll_latest_processed(self):
        """
        Read latest raw frame from engine, pass through LiveAnalyzer,
        and return processed dict or None.
        """
        raw = self.engine.get_latest_raw()
        if raw is None:
            return None
        return self.live_analyzer.process_raw(raw)
