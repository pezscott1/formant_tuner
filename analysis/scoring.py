import numpy as np


# ---------------------------------------------------------
# Directional feedback
# ---------------------------------------------------------
def directional_feedback(measured_formants, user_formants, vowel, tolerance):
    """
    Provide feedback on whether F1/F2 should be raised or lowered.
    Uses relative deviation and avoids contradictory arrows.
    """
    entry = user_formants.get(vowel, {})
    target_f1, target_f2 = entry.get("f1"), entry.get("f2")
    f1, f2 = measured_formants[0], measured_formants[1]

    fb_f1 = fb_f2 = None

    if target_f1 and f1:
        diff = f1 - target_f1
        if diff < -tolerance:
            fb_f1 = "↑ raise F1"
        elif diff > tolerance:
            fb_f1 = "↓ lower F1"

    if target_f2 and f2:
        diff = f2 - target_f2
        if diff < -tolerance:
            fb_f2 = "↑ raise F2"
        elif diff > tolerance:
            fb_f2 = "↓ lower F2"

    return fb_f1, fb_f2


# ---------------------------------------------------------
# Plausibility scoring
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Formant scoring (0–100)
# ---------------------------------------------------------
def live_score_formants(target_formants, measured_formants, tolerance=50):
    """
    Gaussian distance scoring for F1/F2/F3.
    Much smoother and more meaningful than linear scoring.
    """
    score = 0
    count = 0

    for m, t in zip(measured_formants, target_formants):
        if m is None or t is None:
            continue

        dist = abs(m - t)

        # Gaussian falloff
        s = np.exp(-(dist ** 2) / (2 * (tolerance ** 2)))
        score += s * 100
        count += 1

    return int(score / count) if count > 0 else 0


# ---------------------------------------------------------
# Resonance tuning score (0–100)
# ---------------------------------------------------------
def resonance_tuning_score(formants, pitch, tolerance=50):
    """
    Score how well measured formants align with harmonics of the given pitch.
    Uses smooth harmonic proximity weighting.
    """
    if pitch is None or np.isnan(pitch):
        return 0

    harmonics = np.array([n * pitch for n in range(1, 12)])
    score = 0
    count = 0

    for f in formants:
        if f is None:
            continue

        # Distance to nearest harmonic
        d = np.min(np.abs(harmonics - f))

        # Gaussian harmonic proximity
        s = np.exp(-(d ** 2) / (2 * (tolerance ** 2)))
        score += s * 100
        count += 1

    return int(score / count) if count > 0 else 0
