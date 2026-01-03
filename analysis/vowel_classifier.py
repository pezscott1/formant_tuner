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


def classify_vowel(f1, f2, centers=None, voice_type=None):
    if centers is None:
        vt = (voice_type or "baritone").lower()
        centers = VOWEL_CENTERS.get(vt)
    if f1 is None or not np.isfinite(f1):
        return None, 0.0, {"reason": "missing_f1"}
    if centers is None:
        return None, 0.0, None
    print("CLASSIFIER RUNNING:", f1, f2)
    # ---------------------------------------------------------
    # Case 1: F2-only classification (classical /i/, /u/, /ɔ/)
    # ---------------------------------------------------------
    if f2 is not None and (f1 is None or np.isnan(f1)):
        # Use only F2 distance
        scores = {}
        for vowel, (t1, t2, _) in centers.items():
            dist = abs(f2 - t2)
            scores[vowel] = dist

        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1])
        best, best_dist = sorted_scores[0]
        second, second_dist = sorted_scores[1]
        confidence = second_dist / (best_dist + 1e-6)

        res = (best, float(confidence), second)
        print("CLASSIFIER RESULT case 1:", res)
        return res

    # ---------------------------------------------------------
    # Case 2: Full 2‑D classification (F1 + F2)
    # ---------------------------------------------------------
    if f1 is None or f2 is None or np.isnan(f1) or np.isnan(f2):
        return None, 0.0, None

    scores = {}
    for vowel, (t1, t2, _) in centers.items():
        dist = np.sqrt((f1 - t1) ** 2 + (f2 - t2) ** 2)
        scores[vowel] = dist

    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1])
    best, best_dist = sorted_scores[0]
    second, second_dist = sorted_scores[1]
    confidence = second_dist / (best_dist + 1e-6)
    res = (best, float(confidence), second)
    print("CLASSIFIER RESULT case 2:", res)
    return res
