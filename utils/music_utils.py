from analysis.vowel_data import NOTE_NAMES
import numpy as np
import matplotlib.pyplot as plt
# -------------------------
# Pitch to MIDI + piano rendering
# -------------------------


def hz_to_midi(f0):
    """Convert frequency in Hz to MIDI note number."""
    if f0 is None or f0 <= 0:
        return None
    return int(round(69 + 12 * np.log2(f0 / 440.0)))


def render_piano(ax, midi_note, octaves=2, base_octave=3):
    """Render a piano keyboard and highlight a given MIDI note."""
    ax.clear()
    white_keys = []
    for i in range(octaves * 7):
        rect = plt.Rectangle(
            (i, 0), 1, 1, facecolor="white", edgecolor="black", zorder=0
        )
        ax.add_patch(rect)
        white_keys.append(rect)
    black_offsets = [0.7, 1.7, 3.7, 4.7, 5.7]
    for octave in range(octaves):
        for offset in black_offsets:
            x = octave * 7 + offset
            rect = plt.Rectangle(
                (x, 0.5),
                0.6,
                0.5,
                facecolor="black",
                edgecolor="black",
                zorder=1,
            )
            ax.add_patch(rect)
    if midi_note is not None:
        key_index = midi_note % 12
        octave = (midi_note // 12) - base_octave
        if 0 <= octave < octaves:
            white_map = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}
            if key_index in white_map:
                idx = white_map[key_index] + octave * 7
                white_keys[idx].set_facecolor("yellow")
            else:
                black_map = {1: 0.7, 3: 1.7, 6: 3.7, 8: 4.7, 10: 5.7}
                if key_index in black_map:
                    x = octave * 7 + black_map[key_index]
                    rect = plt.Rectangle(
                        (x, 0.5),
                        0.6,
                        0.5,
                        facecolor="yellow",
                        edgecolor="black",
                        zorder=2,
                    )
                    ax.add_patch(rect)
    ax.set_xlim(0, octaves * 7)
    ax.set_ylim(0, 1)
    ax.axis("off")


def freq_to_note_name(freq: float) -> str:
    if not freq or freq <= 0:
        return "N/A"
    midi = int(round(69 + 12 * np.log2(freq / 440.0)))
    if midi < 0 or midi >= 128:
        return "N/A"
    name = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{name}{octave}"
