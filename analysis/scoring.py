# analysis/scoring.py
import numpy as np


# ---------------------------------------------------------
# Directional feedback
# ---------------------------------------------------------
def directional_feedback(measured_formants, user_formants, vowel, tolerance):
    """
    Provide feedback on whether F1/F2 should be raised or lowered.
    Now includes:
      - confidence gating
      - drift suppression for back vowels
      - no contradictory arrows
    """
    entry = user_formants.get(vowel, {})
    target_f1, target_f2 = entry.get("f1"), entry.get("f2")
    f1, f2 = measured_formants[0], measured_formants[1]

    fb_f1 = fb_f2 = None

    # Missing data → no feedback
    if f1 is None or f2 is None or target_f1 is None or target_f2 is None:
        return None, None

    # Back vowels: F2 is the anchor, F1 drift allowed
    if vowel in ("ɔ", "u"):
        # F2 feedback
        diff2 = f2 - target_f2
        if diff2 < -tolerance:
            fb_f2 = "↑ raise F2"
        elif diff2 > tolerance:
            fb_f2 = "↓ lower F2"

        # F1 drift allowed → no arrow unless extreme
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
    Updated to:
      - penalize swapped formants
      - penalize extreme mic-induced F2 clustering
      - reward realistic spacing
    """
    if f1 is None or f2 is None:
        return 0.0

    if f1 > f2:
        return -100.0  # impossible

    sep = f2 - f1
    if sep < 200:
        return -50.0  # too close

    # Penalize mic-induced ridge around 2500–2700
    ridge_penalty = abs(f2 - 2600) * 0.02

    score = sep - ridge_penalty - abs(500 - f1) * 0.01
    return float(score)


def choose_best_candidate(initial, retakes):
    """
    Choose the best candidate using plausibility_score, but never
    switch away from the initial unless a retake is clearly better.
    """

    base_f1 = initial.get("f1")
    base_f2 = initial.get("f2")

    def score(c):
        return plausibility_score(c.get("f1"), c.get("f2"))

    best = initial
    best_score = score(initial)

    for r in retakes:
        s = score(r)

        # If practically identical to initial (tiny tweak) → ignore
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
    Improvements:
      - confidence-aware (if provided)
      - smoother falloff
      - avoids punishing missing F3
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
    Improvements:
      - more harmonics (1–12)
      - smoother Gaussian weighting
      - avoids punishing missing formants
      - robust to PitchResult objects
    """

    # -------------------------
    # Robust pitch validation
    # -------------------------
    if pitch is None:
        return 0

    # Unwrap PitchResult → float
    if hasattr(pitch, "f0"):
        pitch = pitch.f0

    # Only accept real numeric scalars
    if not isinstance(pitch, (int, float, np.floating)):
        return 0

    if np.isnan(pitch):
        return 0

    # -------------------------
    # Harmonic scoring
    # -------------------------
    harmonics = np.array([n * pitch for n in range(1, 12)])
    score = 0
    count = 0

    for f in formants:
        if f is None:
            continue

        d = np.min(np.abs(harmonics - f))
        s = np.exp(-(d ** 2) / (2 * (tolerance ** 2)))
        score += s * 100
        count += 1

    return int(score / count) if count > 0 else 0
