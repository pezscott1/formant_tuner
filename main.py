import os
from analysis.engine import FormantAnalysisEngine
from tuner.profile_controller import ProfileManager
from tuner.live_analyzer import LiveAnalyzer
from analysis.smoothing import PitchSmoother, MedianSmoother, LabelSmoother
from tuner.window import TunerWindow
from tuner.controller import Tuner
from PyQt5.QtWidgets import QApplication


def main():
    app = QApplication([])

    # ------------------------------------------------------------
    # 1. Create the analysis engine
    # ------------------------------------------------------------
    engine = FormantAnalysisEngine(voice_type="bass")

    # ------------------------------------------------------------
    # 2. Create the profile manager (REQUIRES profiles_dir + analyzer)
    # ------------------------------------------------------------
    profiles_dir = os.path.join(os.getcwd(), "profiles")
    profile_manager = ProfileManager(profiles_dir, analyzer=engine)

    # If an active profile exists, apply it
    if profile_manager.active_profile_name:
        profile_manager.apply_profile(profile_manager.active_profile_name)
    else:
        # Optional: fall back to a default profile if it exists
        default = "bass"
        if profile_manager.profile_exists(default):
            profile_manager.apply_profile(default)

    # ------------------------------------------------------------
    # 3. Create smoothing filters
    # ------------------------------------------------------------
    pitch_smoother = PitchSmoother()
    median_smoother = MedianSmoother()
    label_smoother = LabelSmoother()

    # ------------------------------------------------------------
    # 4. Create the LiveAnalyzer
    # ------------------------------------------------------------
    live_analyzer = LiveAnalyzer(
        engine,
        pitch_smoother,
        median_smoother,
        label_smoother,
    )

    # ------------------------------------------------------------
    # 5. Create the tuner window
    # ------------------------------------------------------------

    tuner = Tuner(voice_type="bass", profiles_dir="profiles")
    win = TunerWindow(tuner)
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
