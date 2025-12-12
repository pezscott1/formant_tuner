import numpy as np
from scipy.signal import lfilter, find_peaks
from scipy.signal.windows import hamming
from numpy.linalg import lstsq
import logging, os, tempfile, json
from datetime import datetime, timezone
from vowel_data import FORMANTS, VOWEL_MAP
logger = logging.getLogger(__name__)


def _atomic_write_json(path, obj):
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=dirpath, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass


def lpc_envelope_peaks(frame, sr, order=14, nfft=8192, low=50, high=4000, peak_thresh=0.02):
    try:
        frame = lfilter([1, -0.97], 1, frame)
        frame = frame * np.hamming(len(frame))
        R = np.correlate(frame, frame, mode='full')[len(frame)-1:]
        order = min(order, max(8, len(R)-1))
        Rm = np.array([R[i:i+order] for i in range(order)])
        rv = -R[order:order+order]
        a, *_ = lstsq(Rm, rv, rcond=None)
        a = np.concatenate(([1.0], a))
        w = np.linspace(0, np.pi, nfft//2)
        h = 1.0 / np.polyval(a, np.exp(-1j*w))
        freqs = (w / (2*np.pi)) * sr
        env = 20 * np.log10(np.abs(h) + 1e-12)
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return np.array([]), np.array([])
        peaks, props = find_peaks(env[mask], height=np.max(env[mask]) * peak_thresh)
        peak_freqs = freqs[mask][peaks] if peaks.size else np.array([])
        heights = env[mask][peaks] if peaks.size else np.array([])
        return np.round(peak_freqs, 1), np.round(heights, 2)
    except Exception:
        logger.exception("lpc_envelope_peaks failed")
        return np.array([]), np.array([])


def directional_feedback(measured_formants, user_formants, vowel, tolerance):
    entry = user_formants.get(vowel, {})
    target_f1, target_f2 = entry.get("f1"), entry.get("f2")
    f1, f2 = measured_formants[0], measured_formants[1]

    fb_f1 = None
    fb_f2 = None

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


# --- cepstral smoothed envelope peaks (robust fallback) ---
def smoothed_spectrum_peaks(frame, sr, lifter_cut=60, nfft=8192, low=50, high=4000, peak_thresh=0.02):
    """
    Cepstral lifter smoothing to extract spectral-envelope peaks.
    Returns (peak_freqs_array, heights_array).
    """
    try:
        win = frame * np.hamming(len(frame))
        X = np.abs(np.fft.rfft(win, n=nfft))
        logX = np.log(X + 1e-12)
        cep = np.fft.irfft(logX)
        if lifter_cut < 1:
            lifter_cut = 1
        cep[lifter_cut:-lifter_cut] = 0
        smooth_log = np.fft.rfft(cep, n=nfft).real
        env = np.exp(smooth_log)
        freqs = np.fft.rfftfreq(nfft, 1.0/sr)
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return np.array([]), np.array([])
        peaks, props = find_peaks(env[mask], height=np.max(env[mask]) * peak_thresh)
        peak_freqs = freqs[mask][peaks] if peaks.size else np.array([])
        heights = env[mask][peaks] if peaks.size else np.array([])
        return np.round(peak_freqs, 1), np.round(heights, 3)
    except Exception:
        logger.exception("smoothed_spectrum_peaks failed")
        return np.array([]), np.array([])


def is_plausible_formants(f1, f2, voice_type="bass"):
    """
    Return (ok:bool, reason:str). Enforce ordering and plausible ranges.
    These ranges are conservative; tweak for your user population.
    """
    # Accept None as not plausible
    if f1 is None or f2 is None:
        return False, "missing formant"
    # enforce ordering
    if f1 > f2:
        return False, "f1 > f2 (swapped)"
    # plausible ranges
    if not (50 <= f1 <= 1200):
        return False, f"f1 out of range ({f1:.0f} Hz)"
    if not (400 <= f2 <= 5000):
        return False, f"f2 out of range ({f2:.0f} Hz)"
    # relative spacing check
    if (f2 - f1) < 200:
        return False, "formants too close"
    return True, "ok"

 
def dump_live_profile(profile_name, profile_dict, dirpath="profiles"):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ts_path = os.path.join(dirpath, f"{profile_name}_live_{ts}_profile.json")
    latest_path = os.path.join(dirpath, f"{profile_name}_latest_profile.json")
    stable_path = os.path.join(dirpath, f"{profile_name}_profile.json")
    _atomic_write_json(ts_path, profile_dict)
    _atomic_write_json(latest_path, profile_dict)
    _atomic_write_json(stable_path, profile_dict)
    return {"timestamped": ts_path, "latest": latest_path, "stable": stable_path}
   

def normalize_profile_for_save(user_formants, retries_map=None):
    """
    Normalize a user_formants mapping (vowel -> (f1, f2) or list) into
    a dict suitable for JSON saving with fields:
      { vowel: {"f1": float|None, "f2": float|None, "f0": None, "retries": int, "reason": str, "saved_at": iso, "source": "calibration"} }
    retries_map: optional dict mapping vowel -> int (how many retries were used)
    """
    out = {}
    retries_map = retries_map or {}
    for v, vals in (user_formants.items() if isinstance(user_formants, dict) else []):
        # accept tuples/lists like (f1, f2) or (f1, f2, f0)
        f1 = None
        f2 = None
        f0 = None
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
            else:
                # unknown shape: try to coerce
                try:
                    f1 = float(vals)
                except Exception:
                    f1 = None
        except Exception:
            # defensive: leave as None on conversion error
            f1, f2, f0 = None, None, None

        # enforce ordering: f1 <= f2 when both present
        if f1 is not None and f2 is not None and f1 > f2:
            f1, f2 = f2, f1

        # retries default
        retries = int(retries_map.get(v, 0) or 0)

        # plausibility check using existing helper
        ok, reason = is_plausible_formants(f1, f2)
        reason_text = "ok" if ok else reason

        out[v] = {
            "f1": None if f1 is None else float(f1),
            "f2": None if f2 is None else float(f2),
            "f0": None if f0 is None else float(f0),
            "retries": retries,
            "reason": reason_text,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "source": "calibration"
        }
    return out



def simple_estimate_formants_lpc(signal, sample_rate=44100, lpc_order=12, fmin=100, fmax=4000):

    n = len(signal)
    if n < lpc_order + 1:
        return []

    # Pre-emphasis and window
    preemph = lfilter([1, -0.97], 1, signal)
    windowed = preemph * np.hamming(n)
    windowed = windowed - np.mean(windowed)

    # Normalize to unit peak to improve conditioning
    peak = np.max(np.abs(windowed))
    if peak > 0:
        windowed = windowed / peak

    # Autocorrelation
    r = np.correlate(windowed, windowed, mode='full')[n-1:]
    if len(r) < lpc_order + 1:
        return []

    # Yule–Walker Toeplitz system
    T = np.empty((lpc_order, lpc_order), dtype=np.float64)
    for i in range(lpc_order):
        for j in range(lpc_order):
            T[i, j] = r[abs(i - j)]
    rhs = -r[1:lpc_order+1]

    # Regularize and solve
    eps = 1e-6
    try:
        a = np.linalg.solve(T + eps * np.eye(lpc_order), rhs)
    except np.linalg.LinAlgError:
        return []
    a = np.concatenate(([1.0], a))

    # Roots and frequencies
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0]
    angs = np.angle(roots)
    freqs = angs * (sample_rate / (2 * np.pi))

    # Filter by pole magnitude and frequency range
    formants = [f for r, f in zip(roots, freqs) if np.abs(r) > 0.01 and fmin <= f <= fmax]

    print("All root freqs:", np.round(freqs, 2))
    print("Filtered formants:", np.round(formants, 2))

    # Return the lowest three
    return sorted(formants)[:3]
     

def align_formants_to_targets(measured_formants, target_formants):
    aligned = []
    for m, t in zip(measured_formants, target_formants):
        if m is None or t is None or (isinstance(m, float) and np.isnan(m)):
            aligned.append(None)
        else:
            aligned.append(abs(m - t))
    return aligned


def live_score_formants(target_formants, measured_formants, tolerance=50):
    score = 0
    count = 0
    for m, t in zip(measured_formants, target_formants):
        if m is None or t is None: continue
        dist = abs(m - t)
        if dist <= tolerance:
            score += (1 - dist/tolerance) * 100
        count += 1
    return int(score / count) if count > 0 else 0


def overall_rating(measured_formants, target_formants, pitch, tolerance=50):
    vowel_score = live_score_formants(target_formants, measured_formants, tolerance)
    resonance_score = resonance_tuning_score(measured_formants, pitch, tolerance)
    overall = int(0.5 * vowel_score + 0.5 * resonance_score)
    return vowel_score, resonance_score, overall


def resonance_tuning_score(formants, pitch, tolerance=50):
    score = 0; count = 0
    harmonics = [n*pitch for n in range(1,10)]
    for f in formants:
        if f is None: continue
        closest = min(harmonics, key=lambda h: abs(h-f))
        dist = abs(closest-f)
        if dist <= tolerance:
            score += 10
        count += 1
    return int((score/count)*10) if count>0 else 0


def robust_guess(measured_formants, voice_type="bass"):
    if voice_type in FORMANTS:
        ref_map = {v: (f1, f2) for v,(f1,f2,*_) in FORMANTS[voice_type].items()}
    else:
        ref_map = VOWEL_MAP
    valid = [f for f in measured_formants if f is not None and not np.isnan(f)]
    if len(valid) < 2:
        return None, 0.0, None
    f1, f2 = sorted(valid)[:2]
    scores = {v: ((f1-tf1)**2 + (f2-tf2)**2)**0.5 for v,(tf1,tf2) in ref_map.items()}
    best, second = sorted(scores.items(), key=lambda kv: kv[1])[:2]
    confidence = second[1] / (best[1] + 1e-6)
    return best[0], float(confidence), second[0]


# --- Robust LPC formant estimator with fallbacks ---
def estimate_formants_lpc(frame, sr, order_roots=12, order_env=14, nfft=8192, peak_thresh=0.02):
    """
    Robust LPC-based formant estimation:
    - Try root-based candidates (filtered by magnitude < 1.0)
    - Fallback to LPC spectral envelope peaks (choose two lowest strong peaks)
    - Final fallback to cepstral smoothed envelope peaks
    Returns (f1, f2) or (None, None)
    """
    try:
        # pre-emphasis + window
        frame = lfilter([1, -0.97], 1, frame)
        frame = frame * hamming(len(frame))

        # autocorrelation + LPC (for roots)
        R = np.correlate(frame, frame, mode='full')[len(frame)-1:]
        order = min(max(8, order_roots), len(R)-1)
        Rm = np.array([R[i:i+order] for i in range(order)])
        rv = -R[order:order+order]
        a, *_ = lstsq(Rm, rv, rcond=None)
        a = np.concatenate(([1.0], a))

        # root-based candidates (filter unstable roots)
        roots = np.roots(a)
        roots = [r for r in roots if np.imag(r) > 0.01 and np.abs(r) < 1.0]
        freqs = sorted((np.angle(r) * sr) / (2*np.pi) for r in roots)
        formants = [f for f in freqs if 50 <= f <= 6000]
        if len(formants) >= 2:
            return float(formants[0]), float(formants[1])

        # LPC spectral envelope fallback (prefer two lowest peaks)
        env_peaks, env_heights = lpc_envelope_peaks(frame, sr, order=order_env, nfft=nfft, low=50, high=4000, peak_thresh=peak_thresh)
        if env_peaks.size:
            peaks_sorted = np.sort(env_peaks)
            if peaks_sorted.size >= 2:
                return float(peaks_sorted[0]), float(peaks_sorted[1])
            elif peaks_sorted.size == 1:
                return float(peaks_sorted[0]), None

        # Cepstral envelope fallback
        cep_peaks, cep_heights = smoothed_spectrum_peaks(frame, sr, lifter_cut=60, nfft=nfft, low=50, high=4000, peak_thresh=peak_thresh)
        if cep_peaks.size:
            peaks_sorted = np.sort(cep_peaks)
            if peaks_sorted.size >= 2:
                return float(peaks_sorted[0]), float(peaks_sorted[1])
            elif peaks_sorted.size == 1:
                return float(peaks_sorted[0]), None

        return None, None
    except Exception:
        logger.exception("estimate_formants_lpc failed")
        return None, None


# --- scoring and candidate selection ---
def plausibility_score(f1, f2):
    if f1 is None or f2 is None:
        return 0.0
    sep = max(0.0, f2 - f1)
    score = sep - abs(500 - f1) * 0.01 - abs(1500 - f2) * 0.001
    return float(score)

def choose_best_candidate(initial, retakes):
    """
    initial: dict from analyze_segment
    retakes: list of dicts
    Returns best dict by plausibility_score
    """
    best = initial
    best_score = plausibility_score(initial.get("f1"), initial.get("f2"))
    for r in retakes:
        sc = plausibility_score(r.get("f1"), r.get("f2"))
        if sc > best_score:
            best, best_score = r, sc
    return best


def spectral_fallback_from_frames(frames_list, sr, low=600, high=3500, peak_thresh=0.08):
    try:
        for fallback_frame in frames_list:
            win = fallback_frame * hamming(len(fallback_frame))
            S = np.abs(np.fft.rfft(win))
            freqs = np.fft.rfftfreq(len(win), 1.0/sr)
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                S_mask = S[mask]
                peaks, props = find_peaks(S_mask, height=np.max(S_mask) * peak_thresh)
                if peaks.size:
                    # sort peaks by height
                    sorted_idx = np.argsort(props["peak_heights"])[::-1]
                    top_freqs = freqs[mask][peaks[sorted_idx]]
                    # take up to 2 peaks
                    if len(top_freqs) >= 2:
                        return float(top_freqs[0]), float(top_freqs[1])
                    else:
                        return float(top_freqs[0]), np.nan
    except Exception as e:
        logger.debug(f"spectral_fallback failed: {e}")
    return None, None


def guess_vowel(f1, f2, voice_type="bass", last_guess=None):
    if f1 is None or f2 is None:
        return last_guess

    # Guardrails
    if f2 - f1 < 500:
        return last_guess

    # Always define excluded
    excluded = set()
    if f2 < 1200:
        excluded = {"i", "e"}

    best_vowel, best_dist = None, float("inf")
    for vowel, (t1, t2) in FORMANTS[voice_type].items():
        if vowel in excluded:
            continue
        w1 = 1.0
        w2 = 2.0 if vowel in {"i", "e"} else 1.2
        dist = w1 * abs(f1 - t1) + w2 * abs(f2 - t2)
        if dist < best_dist:
            best_vowel, best_dist = vowel, dist

    return best_vowel or last_guess


# Simple autocorrelation pitch estimator
def estimate_pitch(frame, sr):
    frame = np.asarray(frame, dtype=float)
    if frame.size == 0:
        return None
    frame = frame - np.mean(frame)
    corr = np.correlate(frame, frame, mode='full')
    corr = corr[len(corr)//2:]
    d = np.diff(corr)
    pos = np.where(d > 0)[0]
    if pos.size == 0:
        return None
    start = pos[0]
    peak = np.argmax(corr[start:]) + start
    if peak == 0:
        return None
    return sr / peak

 

class LabelSmoother:
    def __init__(self, window=5, min_dwell=2):
        self.buf = deque(maxlen=window)
        self.current = None
        self.dwell = 0
        self.min_dwell = min_dwell

    def update(self, label):
        if label is None:
            return self.current  # hold
        if label == self.current:
            self.dwell += 1
            return self.current
        # Only switch if label appears consistently
        self.buf.append(label)
        if list(self.buf).count(label) >= self.min_dwell:
            self.current = label
            self.dwell = 1
        return self.current


def hz_to_midi(f0):
    if f0 <= 0:
        return None
    return int(round(69 + 12 * np.log2(f0 / 440.0)))

def render_piano(ax, midi_note, octaves=2, base_octave=3):
    ax.clear()

    # Draw white keys
    white_keys = []
    for i in range(octaves * 7):
        rect = plt.Rectangle((i, 0), 1, 1,
                             facecolor="white", edgecolor="black", zorder=0)
        ax.add_patch(rect)
        white_keys.append(rect)

    # Draw black keys (skip E and B in each octave)
    black_offsets = [0.7, 1.7, 3.7, 4.7, 5.7]  # positions relative to octave
    for octave in range(octaves):
        for offset in black_offsets:
            x = octave * 7 + offset
            rect = plt.Rectangle((x, 0.5), 0.6, 0.5,
                                 facecolor="black", edgecolor="black", zorder=1)
            ax.add_patch(rect)

    # Highlight detected note
    if midi_note is not None:
        key_index = midi_note % 12
        octave = (midi_note // 12) - base_octave
        if 0 <= octave < octaves:
            # White key mapping
            white_map = {0:0, 2:1, 4:2, 5:3, 7:4, 9:5, 11:6}
            if key_index in white_map:
                idx = white_map[key_index] + octave*7
                white_keys[idx].set_facecolor("yellow")
            else:
                # Black key mapping
                black_map = {1:0.7, 3:1.7, 6:3.7, 8:4.7, 10:5.7}
                if key_index in black_map:
                    x = octave*7 + black_map[key_index]
                    rect = plt.Rectangle((x, 0.5), 0.6, 0.5,
                                         facecolor="yellow", edgecolor="black", zorder=2)
                    ax.add_patch(rect)

    ax.set_xlim(0, octaves*7)
    ax.set_ylim(0, 1)
    ax.axis("off")
