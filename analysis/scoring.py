# analysis/scoring.py
import numpy as np


def directional_feedback(measured_formants, user_formants, vowel, tolerance):
    """Provide feedback on whether F1/F2 should be raised or lowered."""
    entry = user_formants.get(vowel, {})
    target_f1, target_f2 = entry.get("f1"), entry.get("f2")
    f1, f2 = measured_formants[0], measured_formants[1]
    fb_f1 = fb_f2 = None
    if target_f1 and f1:
        if f1 < target_f1 - tolerance:
            fb_f1 = "↑ raise F1"
        elif f1 > target_f1 + tolerance:
            fb_f1 = "↓ lower F1"
    if target_f2 and f2:
        if f2 < target_f2 - tolerance:
            fb_f2 = "↑ raise F2"
        elif f2 > target_f2 + tolerance:
            fb_f2 = "↓ lower F2"
    return fb_f1, fb_f2


# -------------------------
# Candidate selection and scoring
# -------------------------


def plausibility_score(f1, f2):
    """Score plausibility of F1/F2 separation."""
    if f1 is None or f2 is None:
        return 0.0
    sep = max(0.0, f2 - f1)
    score = sep - abs(500 - f1) * 0.01 - abs(1500 - f2) * 0.001
    return float(score)


def choose_best_candidate(initial, retakes):
    """Choose the best candidate dict by plausibility_score."""
    best = initial
    best_score = plausibility_score(initial.get("f1"), initial.get("f2"))
    for r in retakes:
        sc = plausibility_score(r.get("f1"), r.get("f2"))
        if sc > best_score:
            best, best_score = r, sc
    return best


def live_score_formants(target_formants, measured_formants, tolerance=50):
    """Score how close measured formants are to target formants."""
    score = 0
    count = 0
    for m, t in zip(measured_formants, target_formants):
        if m is None or t is None:
            continue
        dist = abs(m - t)
        if dist <= tolerance:
            score += (1 - dist / tolerance) * 100
        count += 1
    return int(score / count) if count > 0 else 0


def resonance_tuning_score(formants, pitch, tolerance=50):
    """
    Score how well measured formants align with harmonics of the given pitch.
    Returns an integer score (0–100).
    """
    score = 0
    count = 0
    if pitch is None or np.isnan(pitch):
        return 0
    harmonics = [n * pitch for n in range(1, 10)]
    for f in formants:
        if f is None:
            continue
        closest = min(harmonics, key=lambda h: abs(h - f))
        dist = abs(closest - f)
        if dist <= tolerance:
            score += 10
        count += 1
    return int((score / count) * 10) if count > 0 else 0
