# tests/test_music_utils.py
import numpy as np
from utils.music_utils import hz_to_midi, render_piano
from unittest.mock import MagicMock


def test_hz_to_midi_basic():
    assert hz_to_midi(440) == 69  # A4


def test_hz_to_midi_none_or_zero():
    assert hz_to_midi(None) is None
    assert hz_to_midi(0) is None


def test_hz_to_midi_negative():
    assert hz_to_midi(-100) is None


def test_render_piano_runs_without_error():
    ax = MagicMock()
    render_piano(ax, midi_note=60)  # Middle C
    # Just ensure it calls plotting methods
    assert ax.plot.called or ax.add_patch.called


def test_hz_to_midi_standard():
    assert hz_to_midi(440) == 69


def test_hz_to_midi_none_zero_negative():
    assert hz_to_midi(None) is None
    assert hz_to_midi(0) is None
    assert hz_to_midi(-50) is None


def test_hz_to_midi_fractional():
    midi = hz_to_midi(445)
    assert isinstance(midi, int)


def test_render_piano_calls_plotting():
    ax = MagicMock()
    render_piano(ax, midi_note=60)
    assert ax.add_patch.called or ax.plot.called


def test_render_piano_handles_out_of_range():
    ax = MagicMock()
    render_piano(ax, midi_note=200)  # outside piano range
    assert ax.add_patch.called or ax.plot.called
