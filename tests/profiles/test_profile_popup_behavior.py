from unittest.mock import MagicMock
from tuner.controller import Tuner
from tuner.profile_controller import ProfileManager
from calibration.session import CalibrationSession


def test_calibration_does_not_trigger_popups(tmp_path):
    # Create tuner with mocked window
    tuner = Tuner(profiles_dir=str(tmp_path))
    tuner.window = MagicMock()
    tuner.window.show_profile_set_popup = MagicMock()

    # Replace profile manager and attach mocked window
    tuner.profile_manager = ProfileManager(str(tmp_path), analyzer=tuner.engine)
    tuner.profile_manager.window = tuner.window

    # Create calibration session
    session = CalibrationSession(
        profile_name="test",
        voice_type="bass",
        vowels=["a", "e"],
    )
    session.profile_manager = tuner.profile_manager

    # Simulate successful captures
    session.handle_result("a", 500, 1500, 120, confidence=1.0, stability=0.0)
    session.handle_result("e", 400, 2300, 130, confidence=1.0, stability=0.0)

    # Save profile
    base = session.save_profile()

    # Apply profile silently
    tuner.profile_manager.apply_profile(base)

    # Assert NO popup was shown
    tuner.window.show_profile_set_popup.assert_not_called()
