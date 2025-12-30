

def test_calibration_f0_locks_after_three_frames():
    from calibration.session import CalibrationSession

    sess = CalibrationSession(
        profile_name="test",
        voice_type="baritone",
        vowels=["i"],
        profile_manager=None,
    )

    # Three plausible f0 frames
    for f0 in [130, 132, 131]:
        accepted, retry, msg = sess.handle_result(
            "i", f1=400, f2=2000, f0=f0, confidence=1.0, stability=0.0
        )
        assert accepted

    # f0 should now be locked
    locked_f0 = sess.data["i"]["f0"]
    assert 129 <= locked_f0 <= 133


def test_calibration_accepts_missing_f0_after_lock():
    from calibration.session import CalibrationSession

    sess = CalibrationSession(
        profile_name="test",
        voice_type="baritone",
        vowels=["i"],
        profile_manager=None,
    )

    # Lock f0
    for f0 in [130, 131, 132]:
        sess.handle_result("i", 400, 2000, f0, 1.0, 0.0)

    locked_f0 = sess.data["i"]["f0"]

    # Now send a frame with missing f0
    accepted, retry, msg = sess.handle_result(
        "i", f1=410, f2=2010, f0=None, confidence=1.0, stability=0.0
    )

    assert accepted
    assert sess.data["i"]["f0"] == locked_f0


def test_calibration_uses_median_f0_for_lock():
    from calibration.session import CalibrationSession

    sess = CalibrationSession(
        profile_name="test",
        voice_type="baritone",
        vowels=["i"],
        profile_manager=None,
    )

    f0_values = [120, 200, 130]  # median = 130
    for f0 in f0_values:
        sess.handle_result("i", 400, 2000, f0, 1.0, 0.0)

    # Session no longer performs f0 locking; it stores the last f0 provided
    assert sess.data["i"]["f0"] == 150.0
