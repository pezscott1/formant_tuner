from analysis.engine import FormantAnalysisEngine
from tuner.window import TunerWindow
from tuner.controller import Tuner
from PyQt5.QtWidgets import QApplication
import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"


if __name__ == "__main__":
    app = QApplication([])

    # 1. Create ONE shared engine
    engine = FormantAnalysisEngine(voice_type="bass", debug=False, use_hybrid=True,)

    # 2. Create ONE shared tuner (owns LiveAnalyzer + ProfileManager)
    tuner = Tuner(engine=engine, voice_type="bass", profiles_dir="profiles")

    # 3. Apply active or default profile
    pm = tuner.profile_manager
    if pm.active_profile_name:
        pm.apply_profile(None)
    # else:
    #    default = "bass"
    #    if pm.profile_exists(default):
    #        pm.apply_profile(default)

    # 4. Create ONE tuner window using the tuner
    win = TunerWindow(tuner)
    win.show()

    app.exec_()
