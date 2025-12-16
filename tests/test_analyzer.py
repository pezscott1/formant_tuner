from voice_analysis import Analyzer
from tests.conftest import synth_vowel

# tests/test_analyzer.py


def test_process_frame_end_to_end():
    a = Analyzer(voice_type="tenor", smoothing=True, smooth_size=3)
    y = synth_vowel([300, 2300], 16000, dur=0.3, f0=220.0)
    status = a.process_frame(y, 16000, target_pitch_hz=220.0, debug=True)
    assert status["status"] == "ok"


def test_analyzer(analyzer_fixture, sr=16000):
    vowels = {
        "i": (120, 300, 2300),
        "a": (120, 700, 1200),
        "u": (120, 350, 900),
    }
    for name, (f0, f1, f2) in vowels.items():
        seg = synth_vowel([f1, f2], sr, dur=1.0, f0=f0)
        status = analyzer_fixture.process_frame(seg, sr)
        assert status["status"] == "ok"


if __name__ == "__main__":
    analyzer = Analyzer()          # create analyzer directly
    test_analyzer(analyzer)
