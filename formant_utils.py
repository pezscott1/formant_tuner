#!/usr/bin/env python3
import json
import logging
import os
import tempfile
from collections import deque
from datetime import datetime, timezone
from typing import Tuple, List, Optional, Mapping, Any
import librosa
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import lstsq
from scipy.signal import lfilter, find_peaks
from numpy.typing import NDArray
from vowel_data import FORMANTS, VOWEL_MAP, PITCH_RANGES

logger = logging.getLogger(__name__)

PROFILES_DIR = "profiles"
os.makedirs(PROFILES_DIR, exist_ok=True)

FORMANT_TOLERANCE = 0.25  # ±25% tolerance around reference values


# -------------------------
# Plausibility helpers
# -------------------------


def get_vowel_ranges(voice_type, vowel):
    """Return plausible F1/F2 ranges for a given voice type and vowel."""
    vt = voice_type.lower()
    if vt not in FORMANTS:
        vt = "tenor"
    ref = FORMANTS[vt].get(vowel)
    if not ref:
        return None
    f1, f2 = ref[0], ref[1]
    f1_low, f1_high = f1 * (1 - FORMANT_TOLERANCE), f1 * (
        1 + FORMANT_TOLERANCE
    )
    f2_low, f2_high = f2 * (1 - FORMANT_TOLERANCE), f2 * (
        1 + FORMANT_TOLERANCE
    )
    return f1_low, f1_high, f2_low, f2_high


def is_plausible_formants(f1, f2, voice_type="tenor", vowel=None):
    """Check plausibility of F1/F2 values for a given voice type and vowel."""
    if f1 is None or f2 is None:
        return False, "missing formant"
    if f1 > f2:
        return False, "f1 > f2 (swapped)"
    ranges = get_vowel_ranges(voice_type, vowel)
    if not ranges:
        return True, "ok"
    f1_low, f1_high, f2_low, f2_high = ranges
    if not (f1_low <= f1 <= f1_high):
        return False, f"f1 out of range ({f1:.0f} Hz)"
    if not (f2_low <= f2 <= f2_high):
        return False, f"f2 out of range ({f2:.0f} Hz)"
    return True, "ok"


def is_plausible_pitch(f0, voice_type="tenor"):
    """Check plausibility of pitch F0 for a given voice type."""
    if f0 is None or np.isnan(f0):
        return False, "missing pitch"
    vt = voice_type.lower()
    low, high = PITCH_RANGES.get(vt, (100, 600))
    if not (low <= f0 <= high):
        return False, f"f0 out of range ({f0:.0f} Hz)"
    return True, "ok"


def guess_vowel(f1, f2, voice_type="bass", last_guess=None):
    """
    Guess the closest vowel based on measured F1/F2 and reference formants.
    Falls back to last_guess if inputs are missing or implausible.
    """
    if f1 is None or f2 is None:
        return last_guess
    if f2 - f1 < 500:
        return last_guess
    best_vowel, best_dist = None, float("inf")
    ref_map = FORMANTS.get(voice_type, VOWEL_MAP)
    for vowel, formants in ref_map.items():
        t1, t2 = formants[0], formants[1]
        dist = abs(f1 - t1) + abs(f2 - t2)
        if dist < best_dist:
            best_vowel, best_dist = vowel, dist
    return best_vowel or last_guess


# -------------------------
# Pitch estimator
# -------------------------
def _extract_peak_data(
    env: np.ndarray, freqs: np.ndarray, mask: np.ndarray, peak_thresh: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared helper to extract peak frequencies and heights."""
    peaks, _ = find_peaks(env[mask], height=np.max(env[mask]) * peak_thresh)

    # Convert to a plain Python list[int],
    # so the IDE/type checker accepts it as Iterable/Sized
    idx_list: List[int] = [int(p) for p in np.asarray(peaks).ravel()]

    if not idx_list:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # Use list indexing (NumPy accepts list of ints)
    peak_freqs = np.asarray(
        np.round(freqs[mask][idx_list], 1), dtype=np.float64
    )
    heights = np.asarray(np.round(env[mask][idx_list], 2), dtype=np.float64)
    return peak_freqs, heights


def estimate_pitch(frame, sr):
    """Estimate pitch F0 from a frame using autocorrelation."""
    frame = np.asarray(frame, dtype=float)
    if frame.size == 0:
        return None
    frame = frame - np.mean(frame)
    corr = np.correlate(frame, frame, mode="full")
    corr = corr[len(corr) // 2:]
    d = np.diff(corr)
    pos = np.where(d > 0)[0]
    if pos.size == 0:
        return None
    start = pos[0]
    peak = np.argmax(corr[start:]) + start
    if peak == 0:
        return None
    return sr / peak


# -------------------------
# LPC envelope + cepstral fallbacks
# -------------------------


def lpc_envelope_peaks(
    frame, sr, order=14, nfft=8192, low=50, high=4000, peak_thresh=0.02
):
    try:
        frame = lfilter([1, -0.97], 1, frame)
        frame = frame * np.hamming(len(frame))
        R = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
        order = min(order, max(8, len(R) - 1))
        Rm = np.array([R[i: i + order] for i in range(order)])
        rv = -R[order: order + order]
        a, *_ = lstsq(Rm, rv, rcond=None)
        a = np.concatenate(([1.0], a))
        w = np.linspace(0, np.pi, nfft // 2)
        h = 1.0 / np.polyval(a, np.exp(-1j * w))
        freqs = (w / (2 * np.pi)) * sr
        env = 20 * np.log10(np.abs(h) + 1e-12)
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return np.array([]), np.array([])
        return _extract_peak_data(env, freqs, mask, peak_thresh)

    except Exception as e:  # noqa: E722
        logger.exception("lpc_envelope_peaks failed: %s", e)
        return np.array([]), np.array([])


def smoothed_spectrum_peaks(
    frame: NDArray[np.float64],
    sr: int,
    lifter_cut: int = 60,
    nfft: int = 8192,
    low: int = 50,
    high: int = 4000,
    peak_thresh: float = 0.02,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Cepstral lifter smoothing to extract spectral-envelope peaks.
    Returns (peak_freqs_array, heights_array).
    """
    try:
        win = frame * np.hamming(len(frame))
        X = np.abs(np.fft.rfft(win, n=nfft))
        logX = np.log(X + 1e-12)
        cep = np.fft.irfft(logX)
        lifter_cut = max(1, lifter_cut)
        cep[lifter_cut:-lifter_cut] = 0
        smooth_log = np.fft.rfft(cep, n=nfft).real
        env = np.exp(smooth_log)
        freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return np.array([]), np.array([])
        return _extract_peak_data(env, freqs, mask, peak_thresh)

    except Exception as e:  # noqa: E722
        logger.exception("smoothed_spectrum_peaks failed: %s", e)
        return np.array([]), np.array([])


# -------------------------
# Main formant estimator
# -------------------------


def pick_formants(candidates):
    """
    Select F1 and F2 from candidate frequencies.
    Heuristic:
      - F1: lowest candidate in 200–1000 Hz
      - F2: lowest candidate in 800–3000 Hz that is > F1
      - Prefer mid-band (700–1200 Hz) if F2 is very high or F1 is low
      - Fallback: two lowest plausible candidates
    """
    if not candidates:
        return None, None

    candidates = np.array(sorted(candidates))
    f1_range = (200.0, 1000.0)
    f2_range = (800.0, 3000.0)

    f1_candidates = [f for f in candidates if f1_range[0] <= f <= f1_range[1]]
    f1 = f1_candidates[0] if f1_candidates else None

    f2 = None
    if f1 is not None:
        f2_candidates = [
            f for f in candidates if f2_range[0] <= f <= f2_range[1] and f > f1
        ]
        if f2_candidates:
            f2 = f2_candidates[0]
            mid = [f for f in f2_candidates if 700.0 <= f <= 1200.0]
            if (f2 > 1500.0 and mid) or (f1 < 350.0 and mid):
                f2 = mid[0]
    else:
        plausible = [f for f in candidates if 90.0 < f < 5000.0]
        if len(plausible) >= 2:
            f1, f2 = plausible[0], plausible[1]
        elif len(plausible) == 1:
            f1, f2 = plausible[0], None

    return float(f1) if f1 is not None else None, (
        float(f2) if f2 is not None else None
    )


def estimate_formants_lpc(
    y, sr, order=None, win_len_ms=30, pre_emph=0.97, debug=False
):
    """
    Robust LPC-based formant estimator.
    Returns (f1, f2, f3) or (f1, f2, f3, candidates) if debug=True.
    """
    y = np.asarray(y, dtype=float).flatten()
    if y.size == 0:
        return (None, None, None) if not debug else (None, None, None, [])

    energy = np.mean(y**2)
    if energy < 1e-6:
        if debug:
            print("[estimate_formants_lpc] low energy, skipping")
        return (None, None, None) if not debug else (None, None, None, [])

    y = np.append(y[0], y[1:] - pre_emph * y[:-1])

    win_len = int(sr * (win_len_ms / 1000.0))
    win_len = max(win_len, 64)
    if y.size < win_len:
        segment = y
    else:
        start = max(0, (y.size - win_len) // 2)
        segment = y[start: start + win_len]

    segment = segment * np.hamming(len(segment))

    if order is None:
        order = int(2 + sr / 1000)
    order = max(8, min(order, 40))

    try:
        A = librosa.lpc(segment, order=order)
    except Exception as e:  # noqa: E722
        if debug:
            print("[estimate_formants_lpc] librosa.lpc failed:", e)
        return (None, None, None) if not debug else (None, None, None, [])

    roots = np.roots(A)
    roots = [r for r in roots if np.imag(r) > 1e-6 and 0.7 < np.abs(r) < 1.3]

    freqs = np.array([np.angle(r) * (sr / (2.0 * np.pi)) for r in roots])
    freqs = np.sort(freqs[freqs > 0.0])
    candidates = [float(f) for f in freqs if 90.0 < f < min(5000.0, sr / 2.0)]

    if debug:
        print("[DEBUG] LPC candidates:", candidates)

    f1, f2 = pick_formants(candidates)
    f3 = None
    if f2 is not None:
        higher = [f for f in candidates if f > f2 + 100.0]
        if higher:
            f3 = higher[0]

    return (f1, f2, f3, candidates) if debug else (f1, f2, f3)


def unpack_formants(res):
    """Normalize return shapes into a 3‑tuple (f1, f2, f3)."""
    if res is None:
        return None, None, None
    if isinstance(res, (tuple, list)):
        if len(res) >= 3:
            return res[0], res[1], res[2]
        if len(res) == 2:
            return res[0], res[1], None
    return None, None, None


# -------------------------
# Core helpers
# -------------------------


def profile_path(profile_name: str) -> str:
    """Return the filesystem path for a given profile name."""
    safe_name = profile_name.replace(" ", "_")
    return os.path.join(PROFILES_DIR, f"{safe_name}_profile.json")


def _atomic_write_json(path, obj):
    """Atomically write JSON to a file."""
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=dirpath, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception as e:  # noqa: E722
                logger.debug("Failed to remove temp file: %s", e)


def set_active_profile(profile_name: str):
    """Persist the currently active profile so the rest of the app can apply it."""
    try:
        active_path = os.path.join(PROFILES_DIR, "active_profile.json")
        with open(active_path, "w", encoding="utf-8") as fh:
            json.dump({"active": profile_name}, fh)
        logger.info("Set active profile to %s", profile_name)
    except Exception as e:  # noqa: E722
        logger.exception("Failed to set active profile: %s", e)


def dump_live_profile(
    profile_name: str, profile_dict: dict, dirpath: str = "profiles"
) -> dict:
    """
    Save a snapshot of the current profile in three places:
      - timestamped file
      - latest file
      - stable file
    Returns a dict of the paths written.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ts_path = os.path.join(dirpath, f"{profile_name}_live_{ts}_profile.json")
    latest_path = os.path.join(dirpath, f"{profile_name}_latest_profile.json")
    stable_path = os.path.join(dirpath, f"{profile_name}_profile.json")

    _atomic_write_json(ts_path, profile_dict)
    _atomic_write_json(latest_path, profile_dict)
    _atomic_write_json(stable_path, profile_dict)

    return {
        "timestamped": ts_path,
        "latest": latest_path,
        "stable": stable_path,
    }


def normalize_profile_for_save(user_formants, retries_map=None):
    """
    Normalize a user_formants mapping (vowel -> (f1, f2, f0)) into
    a dict suitable for JSON saving.
    """
    out = {}
    retries_map = retries_map or {}
    if not isinstance(user_formants, dict):
        return out

    for vowel, vals in user_formants.items():
        f1 = f2 = f0 = None
        try:
            if isinstance(vals, (list, tuple)):
                if len(vals) > 0:
                    f1 = None if vals[0] is None else float(vals[0])
                if len(vals) > 1:
                    f2 = None if vals[1] is None else float(vals[1])
                if len(vals) > 2:
                    f0 = None if vals[2] is None else float(vals[2])
            elif isinstance(vals, dict):
                f1 = None if vals.get("f1") is None else float(vals.get("f1"))
                f2 = None if vals.get("f2") is None else float(vals.get("f2"))
                f0 = None if vals.get("f0") is None else float(vals.get("f0"))
        except Exception as e:  # noqa: E722
            logger.debug("normalize_profile_for_save failed: %s", e)
            f1, f2, f0 = None, None, None

        if f1 is not None and f2 is not None and f1 > f2:
            f1, f2 = f2, f1

        retries = int(retries_map.get(vowel, 0) or 0)
        ok, reason = is_plausible_formants(f1, f2)
        reason_text = "ok" if ok else reason

        out[vowel] = {
            "f1": None if f1 is None else float(f1),
            "f2": None if f2 is None else float(f2),
            "f0": None if f0 is None else float(f0),
            "retries": retries,
            "reason": reason_text,
            "saved_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "source": "calibration",
        }
    return out


# -------------------------
# Spectrogram and expected formants
# -------------------------


def safe_spectrogram(y, sr, n_fft=2048, hop_length=512):
    """Compute a safe spectrogram, returning freqs, times, and power."""
    if y is None or len(y) == 0:
        f = np.linspace(0, sr / 2, 128)
        t = np.array([0.0])
        Sxx = np.zeros((f.size, t.size))
        return f, t, Sxx

    if len(y) < n_fft:
        f = np.linspace(0, sr / 2, max(64, int(n_fft // 32)))
        t = np.array([0.0])
        Sxx = np.zeros((f.size, t.size))
        return f, t, Sxx

    try:
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        times = librosa.frames_to_time(
            np.arange(S.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft
        )
        return freqs, times, S
    except Exception as e:  # noqa: E722
        logger.debug("librosa spectrogram failed: %s", e)

    try:
        win = np.hanning(n_fft)
        frames = []
        for i in range(0, max(1, len(y) - n_fft + 1), hop_length):
            frame = y[i: i + n_fft] * win
            spec = np.abs(np.fft.rfft(frame)) ** 2
            frames.append(spec)
        S = (
            np.column_stack(frames)
            if frames
            else np.zeros((n_fft // 2 + 1, 1))
        )
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        times = np.arange(S.shape[1]) * (hop_length / float(sr))
        return freqs, times, S
    except Exception as e:  # noqa: E722
        logger.debug("fft spectrogram failed: %s", e)
        f = np.linspace(0, sr / 2, 128)
        t = np.array([0.0])
        Sxx = np.zeros((f.size, t.size))
        return f, t, Sxx


def get_expected_formants(
    voice_type_local: Optional[str],
    vowel: str,
    f0: Optional[float] = None,
    user_offsets_local: Optional[Mapping[str, Any]] = None,
):
    """Return expected F1/F2 for a given voice type and vowel.

    f0 and user_offsets_local are accepted for API compatibility but not used.
    """
    # Mark intentionally unused to satisfy linters
    _ = f0
    _ = user_offsets_local

    vt = voice_type_local.lower() if voice_type_local else "tenor"
    preset = FORMANTS.get(vt, FORMANTS.get("tenor"))
    base = preset.get(vowel)
    if not base:
        return None, None
    f1, f2 = float(base[0]), float(base[1])
    return int(round(f1)), int(round(f2))


# -------------------------
# Directional feedback
# -------------------------


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


# -------------------------
# Vowel guessing
# -------------------------


def robust_guess(measured_formants, voice_type="bass"):
    """Guess vowel robustly from measured formants."""
    if voice_type in FORMANTS:
        ref_map = {
            v: (f1, f2) for v, (f1, f2, *_) in FORMANTS[voice_type].items()
        }
    else:
        ref_map = VOWEL_MAP
    valid = [f for f in measured_formants if f is not None and not np.isnan(f)]
    if len(valid) < 2:
        return None, 0.0, None
    f1, f2 = sorted(valid)[:2]
    scores = {
        v: ((f1 - tf1) ** 2 + (f2 - tf2) ** 2) ** 0.5
        for v, (tf1, tf2) in ref_map.items()
    }
    best, second = sorted(scores.items(), key=lambda kv: kv[1])[:2]
    confidence = second[1] / (best[1] + 1e-6)
    return best[0], float(confidence), second[0]


# -------------------------
# Smoothing helpers
# -------------------------


class LabelSmoother:
    """Smooth label predictions over a sliding window."""

    def __init__(self, window=5, min_dwell=2):
        self.buf = deque(maxlen=window)
        self.current = None
        self.dwell = 0
        self.min_dwell = min_dwell

    def update(self, label):
        if label is None:
            return self.current
        if label == self.current:
            self.dwell += 1
            return self.current
        self.buf.append(label)
        if list(self.buf).count(label) >= self.min_dwell:
            self.current = label
            self.dwell = 1
        return self.current


class PitchSmoother:
    """Smooth pitch estimates over a sliding window."""

    def __init__(self, window=5):
        self.buf = deque(maxlen=window)

    def update(self, f0):
        if f0 is None or np.isnan(f0):
            return None
        self.buf.append(f0)
        return float(np.median(self.buf))


# -------------------------
# Pitch to MIDI + piano rendering
# -------------------------


def hz_to_midi(f0):
    """Convert frequency in Hz to MIDI note number."""
    if f0 is None or f0 <= 0:
        return None
    return int(round(69 + 12 * np.log2(f0 / 440.0)))


def render_piano(ax, midi_note, octaves=2, base_octave=3):
    """Render a piano keyboard and highlight a given MIDI note."""
    ax.clear()
    white_keys = []
    for i in range(octaves * 7):
        rect = plt.Rectangle(
            (i, 0), 1, 1, facecolor="white", edgecolor="black", zorder=0
        )
        ax.add_patch(rect)
        white_keys.append(rect)
    black_offsets = [0.7, 1.7, 3.7, 4.7, 5.7]
    for octave in range(octaves):
        for offset in black_offsets:
            x = octave * 7 + offset
            rect = plt.Rectangle(
                (x, 0.5),
                0.6,
                0.5,
                facecolor="black",
                edgecolor="black",
                zorder=1,
            )
            ax.add_patch(rect)
    if midi_note is not None:
        key_index = midi_note % 12
        octave = (midi_note // 12) - base_octave
        if 0 <= octave < octaves:
            white_map = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}
            if key_index in white_map:
                idx = white_map[key_index] + octave * 7
                white_keys[idx].set_facecolor("yellow")
            else:
                black_map = {1: 0.7, 3: 1.7, 6: 3.7, 8: 4.7, 10: 5.7}
                if key_index in black_map:
                    x = octave * 7 + black_map[key_index]
                    rect = plt.Rectangle(
                        (x, 0.5),
                        0.6,
                        0.5,
                        facecolor="yellow",
                        edgecolor="black",
                        zorder=2,
                    )
                    ax.add_patch(rect)
    ax.set_xlim(0, octaves * 7)
    ax.set_ylim(0, 1)
    ax.axis("off")
