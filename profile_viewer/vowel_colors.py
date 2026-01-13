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
    "i":  "#FF1744",
    "ɪ":  "#00E5FF",
    "e":  "#D500F9",
    "ɛ":  "#76FF03",
    "æ":  "#FF6D00",
    "a":  "#651FFF",
    "ɑ":  "#00E676",
    "ʌ":  "#FFEA00",
    "ɔ":  "#00BFA5",
    "o":  "#F50057",
    "ʊ":  "#AEEA00",
    "u":  "#AA00FF",
}
