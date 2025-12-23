# tests/test_engine_singleton_wiring.py
from analysis.engine import FormantAnalysisEngine
from tuner.controller import Tuner
from tuner.window import TunerWindow


def test_engine_singleton_across_components(qtbot):
    engine = FormantAnalysisEngine(voice_type="bass")
    tuner = Tuner(engine=engine, voice_type="bass", profiles_dir="profiles")

    window = TunerWindow(tuner)
    qtbot.addWidget(window)

    # All components must share the same engine instance
    assert tuner.engine is engine
    assert tuner.live_analyzer.engine is engine
    assert tuner.profile_manager.analyzer is engine
    assert window.analyzer is engine
