# tests/test_profile_activation_behavior.py
from tuner.window import TunerWindow
from tuner.controller import Tuner
from analysis.engine import FormantAnalysisEngine


def test_no_profile_auto_load(qtbot, tmp_path):
    engine = FormantAnalysisEngine()
    tuner = Tuner(engine=engine, profiles_dir=str(tmp_path))

    window = TunerWindow(tuner)
    qtbot.addWidget(window)

    # No profiles exist → Active: None
    assert tuner.profile_manager.active_profile_name is None
    assert "None" in window.active_label.text()


def test_single_click_applies_profile(qtbot, tmp_path):
    # Create a fake profile
    p = tmp_path / "test_profile_profile.json"
    p.write_text('{"voice_type": "bass", "i": {"f1": 300, "f2": 2500, "f3": 100}}')

    engine = FormantAnalysisEngine()
    tuner = Tuner(engine=engine, profiles_dir=str(tmp_path))
    window = TunerWindow(tuner)
    qtbot.addWidget(window)

    window._populate_profiles()
    item = window.profile_list.item(1)  # first real profile

    window._apply_selected_profile_item(item)

    assert tuner.profile_manager.active_profile_name == "test_profile"
    assert engine.calibrated_profile is not None
