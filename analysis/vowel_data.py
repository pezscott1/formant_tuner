"""
Updated vowel centers and pitch ranges for singing calibration.

These values are tuned to Scott’s actual baritone vowel space,
based on real calibration logs (/i/ ≈ 280/2229, /ɛ/ ≈ 553/1936, etc.).

These are *reference centers*, not plausibility windows.
Calibration will override these with learned values.
"""

# ---------------------------------------------------------
# Pitch ranges (sung tessituras)
# ---------------------------------------------------------

PITCH_RANGES = {
    "bass":     (65.0, 370.0),   # C2 – F#4
    "baritone": (82.0, 440.0),   # E2 – A4
    "tenor":    (130.0, 523.0),  # C3 – C5
    "alto":     (196.0, 659.0),  # G3 – E5
    "mezzo":    (196.0, 698.0),  # G3 – F5
    "soprano":  (261.0, 880.0),  # C4 – A5
}

# ---------------------------------------------------------
# Updated vowel centers (F1, F2, F3)
#
# These are realistic singing values, tuned to Scott’s actual
# measured vowel space. They are intentionally *fronted* and
# *bright*, matching your calibration logs.
#
# Calibration will override these centers with learned values.
# ---------------------------------------------------------

VOWEL_CENTERS = {
    "baritone": {
        # From calibration: F1 ≈ 280, F2 ≈ 2229
        "i":  (280.0, 2250.0, 3000.0),

        # Between /i/ and /ɛ/
        # From calibration: F1 ≈ 350–450, F2 ≈ 2000
        "e":  (350.0, 2050.0, 2900.0),

        # From calibration: F1 ≈ 553, F2 ≈ 1936
        "ɛ":  (550.0, 1900.0, 2600.0),

        # Your /ɑ/ is bright and fronted: F1 ≈ 700–800, F2 ≈ 1600–1800
        "ɑ":  (720.0, 1700.0, 2400.0),

        # Your /ɔ/ is extremely fronted: F2 ≈ 2400–2550
        # This is *your* vowel space, not IPA.
        "ɔ":  (600.0, 2450.0, 2600.0),

        # Your /u/ is mid–back but fronted: F1 ≈ 400–500, F2 ≈ 1300–1500
        "u":  (450.0, 1400.0, 2000.0),

    },

    # Other voice types unchanged for now
    "bass": {
        "i":  (260.0, 2100.0, 2900.0),
        "e":  (330.0, 2000.0, 2800.0),
        "ɛ":  (430.0, 1400.0, 2400.0),
        "ɑ":  (620.0, 1200.0, 2200.0),
        "ɔ":  (520.0, 1000.0, 1900.0),
        "u":  (280.0,  800.0, 1700.0),
    },

    "tenor": {
        "i":  (300.0, 2400.0, 3100.0),
        "e":  (380.0, 2300.0, 3000.0),
        "ɛ":  (500.0, 1700.0, 2600.0),
        "ɑ":  (700.0, 1400.0, 2400.0),
        "ɔ":  (580.0, 1200.0, 2100.0),
        "u":  (330.0, 1000.0, 1900.0),
    },

    "alto": {
        "i":  (320.0, 2600.0, 3200.0),
        "e":  (400.0, 2400.0, 3100.0),
        "ɛ":  (520.0, 1800.0, 2700.0),
        "ɑ":  (750.0, 1500.0, 2500.0),
        "ɔ":  (600.0, 1300.0, 2200.0),
        "u":  (350.0, 1100.0, 2000.0),
    },

    "mezzo": {
        "i":  (310.0, 2500.0, 3150.0),
        "e":  (390.0, 2350.0, 3050.0),
        "ɛ":  (510.0, 1750.0, 2650.0),
        "ɑ":  (730.0, 1450.0, 2450.0),
        "ɔ":  (590.0, 1250.0, 2150.0),
        "u":  (340.0, 1050.0, 1950.0),
    },

    "soprano": {
        "i":  (330.0, 2800.0, 3300.0),
        "e":  (420.0, 2600.0, 3200.0),
        "ɛ":  (540.0, 2000.0, 2800.0),
        "ɑ":  (780.0, 1600.0, 2600.0),
        "ɔ":  (620.0, 1400.0, 2300.0),
        "u":  (370.0, 1200.0, 2100.0),
    },
}
VOWEL_CENTERS["bass"]["ʌ"] = (450.0, 1300.0, 2200.0)
VOWEL_CENTERS["tenor"]["ʌ"] = (500.0, 1400.0, 2300.0)
# etc. for other voice types you care about

