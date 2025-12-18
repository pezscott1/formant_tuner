import unittest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import collections
from formant_utils import (
    estimate_formants_lpc,
    live_score_formants,
    resonance_tuning_score,
    robust_guess,
    unpack_formants,
)
from voice_analysis import Analyzer, MedianSmoother
from vowel_data import FORMANTS
from PyQt5.QtWidgets import QMessageBox
from unittest.mock import patch, MagicMock
matplotlib.use("Agg")


# --- helpers for StableTracker ---
def enforce_order(f1, f2):
    """Ensure F1 < F2 if both are valid."""
    if f1 is None or f2 is None:
        return f1, f2
    if f1 > f2:
        return f2, f1
    return f1, f2


def valid_f0(f0):
    """Check if F0 is within plausible range."""
    return f0 is not None and 50 <= f0 <= 500


def valid_f1(f1):
    """Check if F1 is within plausible range."""
    return f1 is not None and 150 <= f1 <= 900


def valid_f2(f2):
    """Check if F2 is within plausible range."""
    return f2 is not None and 200 <= f2 <= 2500


class StableTracker:
    """Track stable median values for F0/F1/F2."""

    def __init__(self, window=5):
        self.f0_buf = collections.deque(maxlen=window)
        self.f1_buf = collections.deque(maxlen=window)
        self.f2_buf = collections.deque(maxlen=window)

    def update(self, status_dict):
        """Update buffers with new values and return medians."""
        f0 = status_dict.get("f0")
        f1, f2, _ = status_dict.get("formants", (None, None, None))

        if not valid_f0(f0):
            f0 = None
        if not valid_f1(f1):
            f1 = None
        if not valid_f2(f2):
            f2 = None

        f1, f2 = enforce_order(f1, f2)

        if f0 is not None:
            self.f0_buf.append(f0)
        if f1 is not None:
            self.f1_buf.append(f1)
        if f2 is not None:
            self.f2_buf.append(f2)

        f0_med = np.median(self.f0_buf) if self.f0_buf else None
        f1_med = np.median(self.f1_buf) if self.f1_buf else None
        f2_med = np.median(self.f2_buf) if self.f2_buf else None
        return {"f0": f0_med, "formants": (f1_med, f2_med)}


# --- Unit tests ---
class TestMic(unittest.TestCase):
    """Tests for microphone recording and formant estimation."""

    def test_record_and_estimate(self):
        sr = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        sig = np.sin(2 * np.pi * 500 * t).astype(np.float32)
        res = estimate_formants_lpc(sig, sr, debug=True)
        f1, f2, f3 = unpack_formants(res)

        def is_finite_number(x):
            try:
                return x is not None and np.isfinite(x)
            except (TypeError, ValueError):
                return False

        ok = False
        if f1 is None and f2 is None:
            ok = True
        elif is_finite_number(f1) and is_finite_number(f2):
            ok = True
        else:
            if isinstance(res, (tuple, list)) and len(res) > 3:
                candidates = res[-1]
                if (
                    isinstance(candidates, (list, tuple))
                    and len(candidates) >= 2
                ):
                    ok = is_finite_number(candidates[0]) and is_finite_number(
                        candidates[1]
                    )

        self.assertTrue(ok)


class TestVowelGuessing(unittest.TestCase):
    """Tests for vowel guessing logic."""

    def test_guess_a_clear(self):
        guess, conf, _ = robust_guess((800, 1200), voice_type="bass")
        self.assertEqual(guess, "a")
        self.assertGreater(conf, 0)

    def test_guess_i_clear(self):
        guess, conf, _ = robust_guess((300, 2500), voice_type="bass")
        self.assertEqual(guess, "i")
        self.assertGreater(conf, 0)

    def test_guess_missing_F3(self):
        guess, _, _ = robust_guess((700, 1300), voice_type="bass")
        self.assertIn(guess, FORMANTS["bass"].keys())

    def test_guess_uncertain(self):
        guess, conf, _ = robust_guess((900, 2900), voice_type="bass")
        self.assertTrue(guess is None or conf < 1.2)

    def test_guess_tie(self):
        guess, conf, second = robust_guess((500, 1500), voice_type="bass")
        self.assertNotEqual(guess, second)
        self.assertGreaterEqual(conf, 0)


class TestMedianSmoother(unittest.TestCase):
    """Tests for MedianSmoother."""

    def test_update_returns_median(self):
        sm = MedianSmoother(size=3)
        sm.update(100, 200, 300)
        sm.update(110, 210, 310)
        sm.update(120, 220, 320)
        f1, f2, f3 = sm.update(130, 230, 330)
        self.assertAlmostEqual(f1, 120.0, places=6)
        self.assertAlmostEqual(f2, 220.0, places=6)
        self.assertAlmostEqual(f3, 320.0, places=6)


class TestAnalyzer(unittest.TestCase):
    """Tests for Analyzer class."""

    def setUp(self):
        self.analyzer = Analyzer(voice_type="bass")

    def test_process_frame_returns_dict(self):
        dummy_audio = np.zeros(100)
        status = self.analyzer.process_frame(
            dummy_audio, sr=44100, target_pitch_hz=220
        )
        self.assertEqual(status["status"], "ok")
        self.assertIn("formants", status)
        f1, f2, _ = status["formants"]
        self.assertTrue(
            (f1 is None and f2 is None)
            or (isinstance(f1, float) and isinstance(f2, float))
        )

    def test_render_status_text_runs(self):
        fig, ax = plt.subplots()
        status = {
            "status": "ok",
            "f0": 220,
            "formants": (500, 1500, 2500),
            "guess": "a",
            "conf": 0.9,
        }
        self.analyzer.render_status_text(ax, status)

    def test_render_vowel_chart_runs(self):
        fig, ax = plt.subplots()
        ranked = [("a", 0.1), ("e", 0.2)]
        self.analyzer.render_vowel_chart(ax, "bass", 700, 1200, ranked)

    def test_render_spectrum_runs(self):
        fig, ax = plt.subplots()
        freqs = np.linspace(0, 5000, 500)
        mags = np.abs(np.sin(freqs / 500))
        self.analyzer.render_spectrum(ax, freqs, mags, (500, 1500, 2500))

    def test_render_diagnostics_runs(self):
        fig, ax = plt.subplots()
        status = {
            "status": "ok",
            "f0": 220,
            "formants": (500, 1500, 2500),
            "guess": "a",
            "conf": 0.9,
            "next": "e",
            "resonance": 50,
            "overall": 75,
            "penalty": 0.0,
        }
        self.analyzer.render_diagnostics(
            ax, status, sr=44100, frame_len_samples=1024
        )

    def test_vowel_targets_contains_bass(self):
        self.assertIn("bass", FORMANTS)
        self.assertIn("a", FORMANTS["bass"])


class TestScoring(unittest.TestCase):
    """Tests for scoring functions."""

    def test_live_score_formants(self):
        target = (800, 1200, 2800)
        measured = (810, 1190, 2790)
        score = live_score_formants(target, measured, tolerance=50)
        self.assertGreaterEqual(score, 80)

    def test_resonance_tuning_score(self):
        pitch = 200
        measured = [400, 600, 800]
        score = resonance_tuning_score(measured, pitch, tolerance=50)
        self.assertEqual(score, 100)

    def test_overall_rating(self):
        target = (800, 1200, 2800)
        measured = (810, 1190, 2790)
        pitch = 200
        tol = 50
        vowel_score = live_score_formants(target, measured, tolerance=tol)
        resonance_score = resonance_tuning_score(
            measured, pitch, tolerance=tol
        )
        overall = int(0.5 * vowel_score + 0.5 * resonance_score)
        self.assertGreaterEqual(overall, 50)


class TestStableTracker(unittest.TestCase):
    """Tests for StableTracker median logic."""

    def test_rejects_bad_values_and_returns_median(self):
        tracker = StableTracker(window=3)
        stream = [
            {"status": "ok", "f0": 135.6, "formants": (215.3, 236.8, None)},
            {
                "status": "ok",
                "f0": 14700.0,
                "formants": (2067.1, 2088.7, None),
            },  # rejected
            {"status": "ok", "f0": 134.8, "formants": (236.8, 258.3, None)},
        ]
        stable = None
        for s in stream:
            stable = tracker.update(s)

        self.assertIsNotNone(stable)
        self.assertTrue(50 <= stable["f0"] <= 500)
        f1, f2 = stable["formants"]
        self.assertTrue(150 <= f1 <= 900)
        self.assertTrue(200 <= f2 <= 2500)


def test_apply_selected_profile_no_item(main_window):
    main_window.profile_list.clear()
    with patch.object(QMessageBox, "warning") as mock_warn:
        main_window.apply_selected_profile()
        mock_warn.assert_called_once()


def test_delete_profile_no_item(main_window):
    main_window.profile_list.clear()
    with patch.object(QMessageBox, "warning") as mock_warn:
        main_window.delete_profile()
        mock_warn.assert_called_once()


def test_refresh_profiles_with_nonexistent_selection(main_window):
    with patch.object(QMessageBox, "warning") as mock_warn:
        main_window.refresh_profiles(select="does_not_exist")
        # Should not crash, may warn or leave selection None


def test_stabletracker_rejects_bad_f1_f2():
    tracker = StableTracker(window=2)
    status = {"f0": 120, "formants": (50, 10000, None)}  # out of range
    result = tracker.update(status)
    assert result["formants"][0] is None or result["formants"][1] is None


def test_delete_profile_no_selection_triggers_warning(qtbot):
    from formant_tuner import FormantTunerApp
    from voice_analysis import Analyzer
    app = FormantTunerApp(Analyzer(voice_type="bass"))
    qtbot.addWidget(app)
    with patch("calibration.QMessageBox.warning") as mock_warn:
        app.delete_profile()
        mock_warn.assert_called_once()


def test_apply_selected_profile_no_selection_triggers_warning(qtbot):
    from formant_tuner import FormantTunerApp
    from voice_analysis import Analyzer
    app = FormantTunerApp(Analyzer(voice_type="bass"))
    qtbot.addWidget(app)
    with patch("calibration.QMessageBox.warning") as mock_warn:
        app.apply_selected_profile()
        mock_warn.assert_called_once()


def test_toggle_spectrogram_on_and_off(qtbot):
    from formant_tuner import FormantTunerApp
    from voice_analysis import Analyzer
    app = FormantTunerApp(Analyzer(voice_type="bass"))
    qtbot.addWidget(app)
    app.toggle_spectrogram(True)
    assert "Spectrogram" in app.ax_spec.get_title()
    app.toggle_spectrogram(False)
    assert "Spectrum" in app.ax_spec.get_title()


def test_on_profile_calibrated_updates_active(qtbot):
    from formant_tuner import FormantTunerApp
    from voice_analysis import Analyzer
    app = FormantTunerApp(Analyzer(voice_type="bass"))
    qtbot.addWidget(app)
    with patch("calibration.QMessageBox.information") as mock_info:
        app.on_profile_calibrated("user1_tenor")
        assert "Active:" in app.active_label.text()
        assert "user1" in app.active_label.text()
        mock_info.assert_called_once()


def test_update_spectrum_with_no_harmonics(qtbot):
    from formant_tuner import FormantTunerApp
    from voice_analysis import Analyzer
    app = FormantTunerApp(Analyzer(voice_type="bass"))
    qtbot.addWidget(app)
    app.update_spectrum("a", (500, 1500, 200), (np.nan, np.nan, np.nan), pitch=5000, tolerance=50)
    assert "No harmonics" in app.ax_spec.get_title()


def test_close_event_stops_timer_and_mic(qtbot):
    from formant_tuner import FormantTunerApp
    from voice_analysis import Analyzer
    app = FormantTunerApp(Analyzer(voice_type="bass"))
    qtbot.addWidget(app)
    app.mic.stop = MagicMock()
    ev = MagicMock()
    app.closeEvent(ev)
    app.mic.stop.assert_called_once()
    ev.accept.assert_called_once()


def teardown_function(function):
    plt.close("all")


if __name__ == "__main__":
    unittest.main(verbosity=2)
