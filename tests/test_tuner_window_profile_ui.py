# tests/test_tuner_window_profile_ui.py
from tuner.window import TunerWindow
from tuner.controller import Tuner
from analysis.engine import FormantAnalysisEngine


def test_set_active_profile_updates_label(qtbot, tmp_path):
    engine = FormantAnalysisEngine()
    tuner = Tuner(engine=engine, profiles_dir=str(tmp_path))
    window = TunerWindow(tuner)
    qtbot.addWidget(window)

    window._set_active_profile("bass")
    assert "bass" in window.active_label.text()


def test_apply_profile_updates_engine(qtbot, tmp_path):
    f = tmp_path / "bass_profile.json"
    f.write_text('{"voice_type": "bass", "i": {"f1": 300, "f2": 2500, "f3": 100}}')

    engine = FormantAnalysisEngine()
    tuner = Tuner(engine=engine, profiles_dir=str(tmp_path))
    window = TunerWindow(tuner)
    qtbot.addWidget(window)

    window._apply_profile_base("bass")
    assert engine.calibrated_profile is not None
