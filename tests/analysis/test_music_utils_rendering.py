import matplotlib.pyplot as plt

from utils.music_utils import (
    hz_to_midi,
    freq_to_note_name,
    render_piano,
)


# -------------------------
# hz_to_midi tests
# -------------------------

def test_hz_to_midi_none():
    assert hz_to_midi(None) is None


def test_hz_to_midi_nonpositive():
    assert hz_to_midi(0) is None
    assert hz_to_midi(-5) is None


def test_hz_to_midi_basic_notes():
    # A4 = 440 Hz → MIDI 69
    assert hz_to_midi(440) == 69
    # A5 = 880 Hz → MIDI 81
    assert hz_to_midi(880) == 81


# -------------------------
# freq_to_note_name tests
# -------------------------

def test_freq_to_note_name_invalid():
    assert freq_to_note_name(None) == "N/A"
    assert freq_to_note_name(0) == "N/A"
    assert freq_to_note_name(-10) == "N/A"


def test_freq_to_note_name_basic():
    # 440 Hz = A4
    assert freq_to_note_name(440) == "A4"
    # 261.63 Hz ≈ C4
    assert freq_to_note_name(261.63) == "C4"


def test_freq_to_note_name_out_of_range():
    # MIDI < 0
    assert freq_to_note_name(0.1) == "N/A"
    # MIDI >= 128
    assert freq_to_note_name(20000) == "N/A"


# -------------------------
# render_piano tests
# -------------------------

def test_render_piano_runs_without_error():
    fig, ax = plt.subplots()
    render_piano(ax, midi_note=None)
    # Should not raise, and axis should be off
    assert ax.get_xlim() == (0, 14)  # 2 octaves * 7 white keys
    assert ax.get_ylim() == (0, 1)


def test_render_piano_highlights_white_key():
    fig, ax = plt.subplots()
    # C4 = MIDI 60 → base_octave=3 → octave index = 60//12 - 3 = 2
    # But octaves=2, so this should NOT highlight anything
    render_piano(ax, 60)
    # Now choose something inside the range: C3 = MIDI 48
    ax.clear()
    render_piano(ax, 48)
    # The first white key should be yellow
    patches = [p for p in ax.patches if p.get_facecolor()[:3] == (1.0, 1.0, 0.0)]
    assert len(patches) == 1


def test_render_piano_highlights_black_key():
    fig, ax = plt.subplots()
    # C#3 = MIDI 49 → black key
    render_piano(ax, 49)
    patches = [p for p in ax.patches if p.get_facecolor()[:3] == (1.0, 1.0, 0.0)]
    assert len(patches) == 1
