from unittest.mock import MagicMock
from tuner.controller import Tuner


def test_load_profile_sets_active_and_updates_engine_voice_type(tmp_path):
    # Mock ProfileManager.apply_profile to return a base name
    tuner = Tuner(voice_type="tenor", profiles_dir=str(tmp_path))
    tuner.profile_manager.apply_profile = MagicMock(return_value="bass")

    result = tuner.load_profile("bass")

    assert result == "bass"
    assert tuner.active_profile == "bass"
    # load_profile should restore engine.voice_type
    assert tuner.engine.voice_type == "tenor"


def test_load_profile_returns_non_string(tmp_path):
    # If apply_profile returns something non-string,
    # engine.voice_type should NOT be touched
    tuner = Tuner(voice_type="tenor", profiles_dir=str(tmp_path))
    tuner.profile_manager.apply_profile = MagicMock(return_value=None)

    result = tuner.load_profile("bass")

    assert result is None
    assert tuner.engine.voice_type == "tenor"  # unchanged


def test_start_and_stop_are_noops(tmp_path):
    tuner = Tuner(profiles_dir=str(tmp_path))

    # Should not raise or return anything meaningful
    assert tuner.start() is None
    assert tuner.stop() is None


def test_poll_latest_processed_returns_none_when_no_raw(tmp_path):
    tuner = Tuner(profiles_dir=str(tmp_path))

    tuner.engine.get_latest_raw = MagicMock(return_value=None)

    assert tuner.poll_latest_processed() is None


def test_poll_latest_processed_calls_live_analyzer(tmp_path):
    tuner = Tuner(profiles_dir=str(tmp_path))

    fake_raw = {"f0": 120}
    tuner.engine.get_latest_raw = MagicMock(return_value=fake_raw)
    tuner.live_analyzer.process_raw = MagicMock(return_value={"processed": True})

    out = tuner.poll_latest_processed()

    tuner.live_analyzer.process_raw.assert_called_once_with(fake_raw)
    assert out == {"processed": True}
