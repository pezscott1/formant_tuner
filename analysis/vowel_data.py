# analysis/vowel_data.py
from typing import Dict, Tuple

# FORMANTS:
#   voice_type -> vowel -> (F1, F2, F3)
#
# These are *reference* values only. Calibration will adapt them to the user.
# The bass row is tuned toward Scott's observed values (bright /ɛ/, higher-F2 /ɑ/).

FORMANTS: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "bass": {
        # i: fairly low F1, high F2
        "i":  (300.0, 2400.0, 3200.0),

        # ɛ: bright, relatively high F1/F2 for a bass (from logs: ~600–800 / 2300–2600)
        "ɛ":  (650.0, 2350.0, 3000.0),

        # ɑ: back-ish but with higher F2 than
        # “classical” tables (logs: ~800–900 / 1600–1800)
        "ɑ":  (800.0, 1700.0, 2600.0),

        # ɔ: mid F1, low-mid F2
        "ɔ":  (550.0, 1100.0, 2400.0),

        # u: low F1, low F2
        "u":  (350.0, 900.0, 2300.0),
    },

    # The others are conservative defaults; you can refine later with more data.
    "baritone": {
        "i":  (310.0, 2500.0, 3300.0),
        "ɛ":  (600.0, 2200.0, 3000.0),
        "ɑ":  (750.0, 1500.0, 2500.0),
        "ɔ":  (700.0, 2700.0, 3000.0),   # center values based on your real data
        "u":  (600.0, 2650.0, 3000.0),
    },
    "tenor": {
        "i":  (300.0, 2600.0, 3400.0),
        "ɛ":  (550.0, 2300.0, 3000.0),
        "ɑ":  (700.0, 1400.0, 2500.0),
        "ɔ":  (500.0, 900.0, 2300.0),
        "u":  (320.0, 800.0, 2200.0),
    },
    "alto": {
        "i":  (320.0, 2700.0, 3400.0),
        "ɛ":  (600.0, 2400.0, 3100.0),
        "ɑ":  (750.0, 1500.0, 2600.0),
        "ɔ":  (550.0, 1000.0, 2400.0),
        "u":  (350.0, 900.0, 2300.0),
    },
    "soprano": {
        "i":  (330.0, 2800.0, 3500.0),
        "ɛ":  (650.0, 2500.0, 3200.0),
        "ɑ":  (800.0, 1600.0, 2700.0),
        "ɔ":  (550.0, 1100.0, 2500.0),
        "u":  (360.0, 950.0, 2400.0),
    },
}

# Global pitch plausibility ranges by voice type (Hz)
PITCH_RANGES: Dict[str, Tuple[float, float]] = {
    "bass":     (60.0, 260.0),
    "baritone": (70.0, 280.0),
    "tenor":    (80.0, 320.0),
    "alto":     (100.0, 500.0),
    "soprano":  (120.0, 700.0),
}
