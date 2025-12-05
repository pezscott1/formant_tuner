
import sounddevice as sd
import numpy as np
import unittest
from formant_tuner import (
    estimate_formants_lpc,
    live_score_formants,
    resonance_tuning_score,
    overall_rating
)


class TestRatings(unittest.TestCase):
    def test_resonance_tuning_score_perfect(self):
        # Pitch at 200 Hz, harmonics at 200, 400, 600...
        pitch = 200
        measured = [400, 600, 800]  # exactly on harmonics
        score = resonance_tuning_score(measured, pitch, tolerance=50)
        print("Resonance score (perfect):", score)
        self.assertEqual(score, 100)

    def test_resonance_tuning_score_off(self):
        pitch = 200
        measured = [410, 615, 830]  # slightly off harmonics
        score = resonance_tuning_score(measured, pitch, tolerance=50)
        print("Resonance score (off):", score)
        self.assertLess(score, 100)
        self.assertGreater(score, 0)

    def test_overall_rating(self):
        target = (800, 1200, 2800)
        measured = (810, 1190, 2790)
        pitch = 200
        tol = 50
        vowel_score, resonance_score, overall = overall_rating(measured, target, pitch, tol)
        print("Vowel:", vowel_score, "Resonance:", resonance_score, "Overall:", overall)
        self.assertGreaterEqual(overall, 50)

class TestFormants(unittest.TestCase):
    def test_sine_wave(self):
        import numpy as np
        sr = 44100
        t = np.linspace(0, 1, sr, endpoint=False)
        sig = np.sin(2*np.pi*500*t)
        formants = estimate_formants_lpc(sig, sample_rate=sr)
        print("Formants for sine:", formants)
        # Expect empty or very few formants
        self.assertTrue(len(formants) <= 3)

    def test_score(self):
        target = (800, 1200, 2800)
        measured = (810, 1190, 2790)
        score = live_score_formants(target, measured, tolerance=50)
        print("Score:", score)
        self.assertGreaterEqual(score, 80)

class TestMic(unittest.TestCase):
    def test_record_and_estimate(self):
        sr = 44100
        duration = 2.0
        print("Speak a vowel now...")
        audio = sd.rec(int(duration*sr), samplerate=sr, channels=1)
        sd.wait()
        self.assertTrue(np.max(audio) > 0.01, "No sound captured")
        formants = estimate_formants_lpc(audio[:,0], sample_rate=sr)
        print("Measured formants:", formants)
        self.assertTrue(len(formants) > 0, "No formants detected")

if __name__ == "__main__":
    unittest.main()
