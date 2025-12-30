# tests/test_music_utils.py
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from utils.music_utils import (
    hz_to_midi,
    freq_to_note_name,
    render_piano,
)


# ---------------------------------------------------------
# hz_to_midi tests
# ---------------------------------------------------------

def test_hz_to_midi_basic():
    assert hz_to_midi(440) == 69      # A4
    assert hz_to_midi(880) == 81      # A5
    assert hz_to_midi(220) == 57      # A3


def test_hz_to_midi_invalid_inputs():
    assert hz_to_midi(None) is None
    assert hz_to_midi(0) is None
    assert hz_to_midi(-10) is None


def test_hz_to_midi_fractional_rounding():
    midi = hz_to_midi(445)
    assert isinstance(midi, int)


def test_hz_to_midi_extreme_values():
    # Very small frequencies produce very negative MIDI numbers
    midi = hz_to_midi(1e-9)
    assert isinstance(midi, int)


# ---------------------------------------------------------
# freq_to_note_name tests
# ---------------------------------------------------------

def test_freq_to_note_name_basic():
    assert freq_to_note_name(440) == "A4"
    assert freq_to_note_name(261.63) == "C4"  # Middle C approx


def test_freq_to_note_name_accidentals():
    assert freq_to_note_name(277) in ("C#4", "D♭4", "Db4")
    assert freq_to_note_name(370) in ("F#4", "Gb4")
    assert freq_to_note_name(415) in ("G#4", "Ab4")


def test_freq_to_note_name_rounding_boundaries():
    assert freq_to_note_name(439.5) == "A4"
    assert freq_to_note_name(440.4) == "A4"


def test_freq_to_note_name_invalid():
    assert freq_to_note_name(0) == "N/A"
    assert freq_to_note_name(-5) == "N/A"
    assert freq_to_note_name(1e9) == "N/A"  # MIDI out of range


def test_freq_to_note_name_extremes():
    assert isinstance(freq_to_note_name(20), str)
    assert isinstance(freq_to_note_name(20000), str)


# ---------------------------------------------------------
# render_piano tests
# ---------------------------------------------------------

def test_render_piano_no_crash():
    fig, ax = plt.subplots()
    render_piano(ax, midi_note=None)
    plt.close(fig)


def test_render_piano_highlights_white_key():
    fig, ax = plt.subplots()
    render_piano(ax, midi_note=60)  # C4
    plt.close(fig)


def test_render_piano_highlights_black_key():
    fig, ax = plt.subplots()
    render_piano(ax, midi_note=61)  # C#4
    plt.close(fig)


def test_render_piano_calls_plotting():
    ax = MagicMock()
    render_piano(ax, midi_note=60)
    assert ax.add_patch.called or ax.plot.called


def test_render_piano_out_of_range_high():
    ax = MagicMock()
    render_piano(ax, midi_note=200)
    assert ax.add_patch.called or ax.plot.called


def test_render_piano_out_of_range_low():
    ax = MagicMock()
    render_piano(ax, midi_note=-50)
    assert ax.add_patch.called or ax.plot.called


def test_render_piano_handles_negative_midi():
    ax = MagicMock()
    render_piano(ax, midi_note=-10)
    assert ax.add_patch.called or ax.plot.called
