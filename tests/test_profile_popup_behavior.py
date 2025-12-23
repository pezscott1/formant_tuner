import json
from unittest.mock import MagicMock

from tuner.controller import Tuner
from tuner.profile_controller import ProfileManager
from calibration.session import CalibrationSession


def test_calibration_does_not_trigger_popups(tmp_path):
    """
    Calibration should save/apply profiles silently.
    No UI popup should be shown.
    """

    # Create tuner with mocked window
    tuner = Tuner(profiles_dir=str(tmp_path))
    tuner.window = MagicMock()
    tuner.window.show_profile_set_popup = MagicMock()

    # Replace profile manager with one using the same analyzer
    tuner.profile_manager = ProfileManager(str(tmp_path), analyzer=tuner.engine)

    # Create a calibration session (pure logic)
    session = CalibrationSession(
        profile_name="test",
        voice_type="bass",
        vowels=["a", "e"],
    )

    # Feed calibration results (simulate successful captures)
    session.handle_result(500, 1500, 120)  # /a/
    session.handle_result(400, 2300, 130)  # /e/

    # Save profile (this is where popups used to fire)
    base = session.save_profile()

    # Apply profile silently
    tuner.profile_manager.apply_profile(base)

    # Assert NO popup was shown
    tuner.window.show_profile_set_popup.assert_not_called()
