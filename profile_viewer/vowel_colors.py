# profile_viewer/vowel_colors.py

def vowel_color_for(vowel: str) -> str:
    if vowel in VOWEL_COLORS:
        return VOWEL_COLORS[vowel]
    else:
        palette = [
            "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
            "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
            "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
            "#aaffc3", "#808000", "#ffd8b1", "#000075",
        ]
        # Stable hash-based assignment
        index = abs(hash(vowel)) % len(palette)
        return palette[index]


VOWEL_COLORS = {
    "i":  "#FF1744",  # neon red
    "ɪ":  "#00E5FF",  # electric cyan
    "e":  "#D500F9",  # ultraviolet magenta
    "ɛ":  "#76FF03",  # radioactive green
    "æ":  "#FF6D00",  # molten orange
    "a":  "#651FFF",  # deep indigo
    "ɑ":  "#00E676",  # mint neon
    "ʌ":  "#FFEA00",  # laser yellow
    "ɔ":  "#00BFA5",  # teal-green
    "o":  "#F50057",  # hot pink
    "ʊ":  "#AEEA00",  # chartreuse
    "u":  "#AA00FF",  # electric purple
}
