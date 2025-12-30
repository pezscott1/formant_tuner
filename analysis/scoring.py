# analysis/scoring.py
import numpy as np

# ---------------------------------------------------------
# Directional feedback
# ---------------------------------------------------------


def directional_feedback(measured_formants, user_formants, vowel, tolerance):
    """
    Provide feedback on whether F1/F2 should be raised or lowered.
    Back vowels: F2 is the anchor, F1 drift allowed.
    """
    entry = user_formants.get(vowel)
    if not entry:
        return None, None

    target_f1 = entry["f1"]
    target_f2 = entry["f2"]

    f1, f2 = measured_formants[0], measured_formants[1]

    if f1 is None or f2 is None or target_f1 is None or target_f2 is None:
        return None, None

    fb_f1 = fb_f2 = None

    # Back vowels: F2 is the anchor
    if vowel in ("ɔ", "u"):
        diff2 = f2 - target_f2
        if diff2 < -tolerance:
            fb_f2 = "↑ raise F2"
        elif diff2 > tolerance:
            fb_f2 = "↓ lower F2"

        diff1 = f1 - target_f1
        if abs(diff1) > 2 * tolerance:
            fb_f1 = "adjust F1 (drift)"

        return fb_f1, fb_f2

    # Standard vowels
    diff1 = f1 - target_f1
    diff2 = f2 - target_f2

    if diff1 < -tolerance:
        fb_f1 = "↑ raise F1"
    elif diff1 > tolerance:
        fb_f1 = "↓ lower F1"

    if diff2 < -tolerance:
        fb_f2 = "↑ raise F2"
    elif diff2 > tolerance:
        fb_f2 = "↓ lower F2"

    return fb_f1, fb_f2


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


def choose_best_candidate(initial, retakes):
    """
    Choose the best candidate using plausibility_score.
    Never switch unless a retake is clearly better.
    """
    base_f1 = initial.get("f1")
    base_f2 = initial.get("f2")

    def score(c):
        return plausibility_score(c.get("f1"), c.get("f2"))

    best = initial
    best_score = score(initial)

    for r in retakes:
        s = score(r)

        if (
            abs(r.get("f1", base_f1) - base_f1) <= 20
            and abs(r.get("f2", base_f2) - base_f2) <= 20
        ):
            continue

        if s > best_score:
            best = r
            best_score = s

    return best


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
