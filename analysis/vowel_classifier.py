# analysis.vowel_classifier.py
"""
Uses:
  - hybrid formants when available
  - distance-based scoring
  - confidence metric
  - fallback to LPC if hybrid unavailable
"""

import numpy as np
from analysis.vowel_data import VOWEL_CENTERS


def classify_vowel(f1, f2, centers=None, voice_type=None):
    if centers is None:
        vt = (voice_type or "baritone").lower()
        centers = VOWEL_CENTERS.get(vt)
    if centers is None:
        return None, 0.0, None

    # Missing or invalid F1
    if f1 is None or not np.isfinite(f1):
        return None, 0.0, {"reason": "missing_f1"}

    # Full 2-D classification requires valid F2
    if f2 is None or not np.isfinite(f2):
        return None, 0.0, None

    scores = {}
    for vowel, (t1, t2, _) in centers.items():
        dist = np.sqrt((f1 - t1)**2 + (f2 - t2)**2)
        scores[vowel] = dist

    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1])
    best, best_dist = sorted_scores[0]
    second, second_dist = sorted_scores[1]
    confidence = second_dist / (best_dist + 1e-6)
    return best, float(confidence), second
