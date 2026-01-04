# analysis/scoring.py
import numpy as np


# ---------------------------------------------------------
# Plausibility scoring
# ---------------------------------------------------------

def plausibility_score(f1, f2):
    """
    Score plausibility of F1/F2 separation.
    Penalizes swapped formants, too-close spacing, and mic ridge.
    """
    if f1 is None or f2 is None:
        return 0.0
    if f1 > f2:
        return -100.0

    sep = f2 - f1
    if sep < 200:
        return -50.0

    ridge_penalty = abs(f2 - 2600) * 0.02
    score = sep - ridge_penalty
    return float(score)


# ---------------------------------------------------------
# Formant scoring (0–100)
# ---------------------------------------------------------

def live_score_formants(target_formants, measured_formants, tolerance=50):
    """
    Gaussian distance scoring for F1/F2/F3.
    Missing F3 is ignored.
    """
    score = 0.0
    count = 0

    for m, t in zip(measured_formants, target_formants):
        if m is None or t is None:
            continue
        if not np.isfinite(m) or not np.isfinite(t):
            continue

        dist = abs(m - t)
        s = np.exp(-(dist ** 2) / (2 * (tolerance ** 2)))
        score += float(s) * 100.0
        count += 1

    if count == 0:
        return 0

    avg = score / count
    return int(avg) if np.isfinite(avg) else 0


# ---------------------------------------------------------
# Resonance tuning score (0–100)
# ---------------------------------------------------------

def resonance_tuning_score(formants, pitch, tolerance=50):
    """
    Score how well measured formants align with harmonics of the pitch.
    """
    if pitch is None:
        return 0

    if hasattr(pitch, "f0"):
        pitch = pitch.f0

    if not isinstance(pitch, (int, float, np.floating)):
        return 0
    if not np.isfinite(pitch):
        return 0

    harmonics = np.array([n * pitch for n in range(1, 12)], dtype=float)
    score = 0.0
    count = 0

    for f in formants:
        if f is None or not np.isfinite(f):
            continue

        d = float(np.min(np.abs(harmonics - f)))
        s = np.exp(-(d ** 2) / (2 * (tolerance ** 2)))
        score += float(s) * 100.0
        count += 1

    if count == 0:
        return 0

    avg = score / count
    return int(avg) if np.isfinite(avg) else 0
