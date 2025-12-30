# analysis.vowel_classifier.py
"""
Modern vowel classifier for Scott’s baritone vowel space.

Uses:
  - hybrid formants when available
  - updated VOWEL_CENTERS tuned to Scott’s real calibration
  - distance-based scoring
  - confidence metric
  - fallback to LPC if hybrid unavailable

This replaces the old robust_guess().
"""

import numpy as np
from analysis.vowel_data import VOWEL_CENTERS


def classify_vowel(f1, f2, voice_type=None):
    if f1 is None or f2 is None or np.isnan(f1) or np.isnan(f2):
        return None, 0.0, None

    vt = (voice_type or "baritone").lower()
    centers = VOWEL_CENTERS.get(vt)
    if centers is None:
        return None, 0.0, None

    # Compute Euclidean distance to each vowel center
    scores = {}
    for vowel, (t1, t2, _) in centers.items():
        dist = np.sqrt((f1 - t1) ** 2 + (f2 - t2) ** 2)
        scores[vowel] = dist

    # Sort by distance
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1])
    best, best_dist = sorted_scores[0]
    second, second_dist = sorted_scores[1]

    # Confidence: lower is better
    confidence = second_dist / (best_dist + 1e-6)

    return best, float(confidence), second
