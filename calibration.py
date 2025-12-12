
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import sounddevice as sd
from scipy.signal import lfilter, find_peaks, spectrogram
from scipy.signal.windows import hamming
from numpy.linalg import lstsq
from collections import deque
import json, math, joblib, os, threading, atexit, librosa, logging, queue, time, traceback, pprint
from sklearn.neighbors import KNeighborsClassifier
from dataclasses import dataclass
from datetime import datetime, timezone
import tempfile
from glob import glob
from voice_analysis import Analyzer
from formant_utils import (
    estimate_formants_lpc,
    estimate_pitch,
    guess_vowel,
    plausibility_score,
    choose_best_candidate,
    spectral_fallback_from_frames,
    normalize_profile_for_save,
    dump_live_profile,
)
from mic_analyzer import MicAnalyzer, results_queue
import sys
import tkinter as tk

# Configure root logger once at program start
logging.basicConfig(
    level=logging.INFO,               # default level; change to DEBUG to see more
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
results_queue = queue.Queue()
calib_queue = queue.Queue() 

PROFILE_DIR = "profiles"
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]
FORMANTS = {
    "bass": {
        "i": (300, 2200),
        "e": (500, 1900),
        "a": (700, 1100),
        "o": (500, 800),
        "u": (350, 600),
    }
}

# --- plausibility helpers and diagnostics ---
DIAGNOSTIC_DIR = "diagnostics"
os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
logger = logging.getLogger(__name__)


def save_diagnostic(vowel, sr, y, f1, f2, f0, profile_name):
    """Save WAV and JSON diagnostics for offline inspection."""
    try:
        import soundfile as sf
        ts = int(time.time())
        base = f"{profile_name}_{vowel}_{ts}"
        wav_path = os.path.join(DIAGNOSTIC_DIR, base + ".wav")
        json_path = os.path.join(DIAGNOSTIC_DIR, base + ".json")
        # write wav (mono)
        try:
            sf.write(wav_path, y.astype(np.float32), sr)
        except Exception:
            # fallback: try sounddevice write if soundfile not available
            pass
        diag = {"vowel": vowel, "f1": f1, "f2": f2, "f0": f0, "sr": sr, "wav": wav_path}
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(diag, jf, indent=2)
        logger.info("Saved diagnostic for %s to %s", vowel, json_path)
    except Exception:
        logger.exception("Failed to save diagnostic for %s", vowel)

# --- add this helper near sanitize_profile_dict / diagnostics helpers ---


def sanitize_profile_dict(d):
    out = {}
    for v, vals in d.items():
        f1 = vals[0] if vals and not (isinstance(vals[0], float) and math.isnan(vals[0])) else None
        f2 = vals[1] if len(vals) > 1 and not (isinstance(vals[1], float) and math.isnan(vals[1])) else None
        out[v] = (f1, f2)
    return out


def profile_summary_text(profile_dict):
    lines = []
    for v, (f1, f2) in profile_dict.items():
        if f1 is None and f2 is None:
            lines.append(f"{v}: no formants")
        elif f2 is None:
            lines.append(f"{v}: f1={f1:.1f} Hz; f2=missing")
        else:
            lines.append(f"{v}: f1={f1:.1f} Hz; f2={f2:.1f} Hz")
    return "\n".join(lines)


def hz_to_note_name(f0):
    if f0 <= 0:
        return None
    # MIDI note number formula
    midi = int(round(69 + 12 * np.log2(f0 / 440.0)))
    name = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{name}{octave}"


# --- JSON helper for diagnostics ---
def _json_default(o):
    try:
        return o.item()
    except Exception:
        try:
            return o.tolist()
        except Exception:
            return str(o)


def estimate_formants_lpc_on_segment(y, sr, cfg=None):
    """
    Robust wrapper used by older call sites.
    Uses getattr defaults so missing attributes won't raise AttributeError.
    Logs cfg type/summary when it's not the expected dataclass instance.
    """
    try:
        if cfg is None:
            cfg = CalibrationConfig(sr=sr)

        # defensive attribute access with defaults
        order_roots = getattr(cfg, "order_roots", 12)
        order_env   = getattr(cfg, "order_env", 14)
        peak_thresh = getattr(cfg, "peak_thresh", 0.02)
        mid_len_sec = getattr(cfg, "mid_len_seconds", 1.5)
        frame_ms    = getattr(cfg, "frame_ms", 60)

        # quick runtime check and log if cfg looks wrong
        if not hasattr(cfg, "__dataclass_fields__") and not isinstance(cfg, CalibrationConfig):
            logger.debug("estimate_formants_lpc_on_segment: cfg is unexpected type %s; attrs: %s",
                         type(cfg), {k: getattr(cfg, k, None) for k in ("order_roots","order_env","peak_thresh","mid_len_seconds","frame_ms")})

        total = len(y)
        mid_len = min(int(mid_len_sec * sr), total)
        mid_start = max(0, total//2 - mid_len//2)
        mid = y[mid_start:mid_start+mid_len].astype(float)

        frame_len = max(32, int(sr * (frame_ms/1000.0)))
        center_frame = mid[max(0, len(mid)//2 - frame_len//2): max(0, len(mid)//2 - frame_len//2) + frame_len]

        return estimate_formants_lpc(center_frame, sr,
                                     order_roots=order_roots,
                                     order_env=order_env,
                                     nfft=8192,
                                     peak_thresh=peak_thresh)
    except Exception:
        logger.exception("estimate_formants_lpc_on_segment failed")
        return None, None


# --- Small calibration config and capture helpers ---
@dataclass
class CalibrationConfig:
    sr: int = 44100
    duration: float = 3.0
    prep_seconds: float = 3.0
    mid_len_seconds: float = 1.5
    frame_ms: int = 60
    max_retries: int = 2
    auto_retake: int = 1
    # new fields required by the formant helpers
    order_roots: int = 12
    order_env: int = 14
    peak_thresh: float = 0.02


def capture_segment(sr, duration):
    """
    Capture `duration` seconds from default input device.
    Returns numpy array of shape (n_samples,).
    """
    frames = int(duration * sr)
    try:
        with sd.InputStream(samplerate=sr, channels=1) as s:
            data, _ = s.read(frames)
            return data[:, 0].astype(float).copy()
    except Exception:
        logger.exception("capture_segment failed")
        return np.zeros(frames, dtype=float)


def analyze_segment(y, sr, cfg=None):
    if cfg is None:
        cfg = CalibrationConfig(sr=sr)

    # choose middle steady region
    total = len(y)
    mid_len = min(int(cfg.mid_len_seconds * sr), total)
    mid_start = max(0, total//2 - mid_len//2)
    mid = y[mid_start:mid_start+mid_len].astype(float)

    # choose a representative frame (center)
    frame_len = max(32, int(sr * (cfg.frame_ms/1000.0)))
    center_frame = mid[max(0, len(mid)//2 - frame_len//2): max(0, len(mid)//2 - frame_len//2) + frame_len]

    # LPC/cepstral estimation
    f1, f2 = estimate_formants_lpc(center_frame, sr,
                                   order_roots=cfg.order_roots,
                                   order_env=cfg.order_env,
                                   nfft=8192,
                                   peak_thresh=cfg.peak_thresh)

    # robust pitch estimate with YIN
    f0 = None
    try:
        f0_series = librosa.yin(mid, fmin=50, fmax=500, sr=sr)
        f0 = float(np.nanmean(f0_series))
    except Exception:
        f0 = None

    # frames_list for potential spectral fallback (split mid into frames)
    frames_list = []
    hop = frame_len // 2
    for i in range(0, max(1, len(mid) - frame_len + 1), hop):
        frames_list.append(mid[i:i+frame_len])

    return {"f1": f1, "f2": f2, "f0": f0, "frames_list": frames_list, "steady": mid}
    

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


# Matplotlib UI

fig, (ax_status, ax_piano) = plt.subplots(2, 1, figsize=(8, 4))
ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03])
pitch_slider = Slider(ax_slider, "Pitch (Hz)", 80, 1000, valinit=220)
ax_radio = plt.axes([0.05, 0.15, 0.15, 0.25])

# Create Start/Stop buttons
ax_start = plt.axes([0.7, 0.02, 0.1, 0.05])   # [left, bottom, width, height]
btn_start = Button(ax_start, "Start")

ax_stop = plt.axes([0.82, 0.02, 0.1, 0.05])
btn_stop = Button(ax_stop, "Stop")

# Pitch slider
ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03])
pitch_slider = Slider(ax_slider, "Pitch (Hz)", 80, 1000, valinit=220)

# Tolerance slider
ax_tol = plt.axes([0.25, 0.01, 0.65, 0.03])
tol_slider = Slider(ax_tol, "Tolerance (Hz)", 10, 200, valinit=50)

analyzer = Analyzer() 

mic = MicAnalyzer(
    vowel_provider=lambda: current_formants,
    tol_provider=lambda: tol_slider.val,
    pitch_provider=lambda: pitch_slider.val,
    sample_rate=44100,
    frame_ms=80,
    analyzer=analyzer
)


# Attach callbacks
btn_start.on_clicked(lambda e: mic.start())
btn_stop.on_clicked(lambda e: mic.stop())


current_vowel_name = None
current_formants = (None, None, None)

def load_selected_profile(label):
    global current_vowel_name, current_formants
    profile_path = os.path.join(PROFILE_DIR, f"{label}_profile.json")
    model_path = os.path.join(PROFILE_DIR, f"{label}_model.pkl")
    analyzer.load_profile(profile_path, model_path)

    # update current vowel state
    current_vowel_name = label
    entry = analyzer.user_formants.get(label, {"f1": None, "f2": None})
    current_formants = (entry.get("f1"), entry.get("f2"), None)

    ax_status.clear()
    ax_status.text(0.05, 0.8, f"Loaded profile: {label}", transform=ax_status.transAxes, color="blue")
    fig.canvas.draw_idle()


def refresh_profiles():
    PROFILE_DIR = "profiles"
    os.makedirs(PROFILE_DIR, exist_ok=True)
    profiles = [f.split("_profile.json")[0] for f in os.listdir(PROFILE_DIR) if f.endswith("_profile.json")]
    ax_radio.clear()
    if profiles:
        radio = RadioButtons(ax_radio, profiles)
        radio.on_clicked(load_selected_profile)
    else:
        ax_status.text(0.05, 0.8, "No profiles found", transform=ax_status.transAxes, color="red")
    fig.canvas.draw_idle()
    return radio if profiles else None


# If you want a GUI text box for profile name, create it earlier:

ax_textbox = plt.axes([0.25, 0.10, 0.3, 0.05])
profile_box = TextBox(ax_textbox, "Profile Name", initial="user1")

# Shared state
_calib_thread = None
_calib_result = {"done": False, "success": False, "profile": None}
_calibrating = False
# Safe button creation / rebind


def run_calibration(event):
    global _calib_thread, _calib_result, _calibrating

    # double-check guard
    if _calibrating and _calib_thread and getattr(_calib_thread, "is_alive", lambda: False)():
        logger.info("run_calibration: already running, returning")
        return

    # prepare state
    profile_name = profile_box.text.strip() if 'profile_box' in globals() else "user1"
    _calib_result = {"done": False, "success": False, "profile": profile_name, "prompt": f"Starting calibration for {profile_name}"}

    # ensure mic stopped (safe)
    mic_obj = globals().get("mic", None)
    if mic_obj is not None:
        try:
            if getattr(mic_obj, "active", False):
                mic_obj.stop()
                logger.info("mic stopped for calibration")
                time.sleep(0.05)
        except Exception:
            logger.exception("Failed to stop mic (continuing)")

    # prevent duplicate thread start
    if _calib_thread and getattr(_calib_thread, "is_alive", lambda: False)():
        logger.info("run_calibration: thread already alive; not starting another")
        return

    # create and assign thread to global before starting
    t = threading.Thread(
        target=_calibration_worker,
        args=(profile_name, getattr(mic_obj, "samplerate", 44100), 2.0, ["i","e","a","o","u"]),
        daemon=True
    )
    _calib_thread = t
    logger.info("run_calibration: assigned global _calib_thread, starting thread")
    t.start()

# Non-overlapping position (adjust if needed)
btn_pos = [0.05, 0.02, 0.15, 0.06]

# Create or recreate the button
if "calib_button" not in globals() or calib_button is None:
    ax_button = plt.axes(btn_pos)
    calib_button = Button(ax_button, "Calibrate")
    calib_button.on_clicked(run_calibration)
else:
    try:
        axb = calib_button.ax
        calib_button = Button(axb, "Calibrate")
        calib_button.on_clicked(run_calibration)
    except Exception:
        calib_button.on_clicked(run_calibration)

fig.canvas.draw_idle()


def choose_best_candidate(initial, retakes):
    """
    initial is dict from analyze_segment. retakes is list of analyze_segment results.
    Returns best dict.
    """
    best = initial
    best_score = plausibility_score(initial.get("f1"), initial.get("f2"))
    for r in retakes:
        sc = plausibility_score(r.get("f1"), r.get("f2"))
        if sc > best_score:
            best, best_score = r, sc
    return best


def _calibration_worker(profile_name, sr, duration, vowels):
    user_formants = {}
    for v in vowels:
        success = False
        retries = 0
        while not success and retries < 2:  # allow one retry
            # prep countdown
            for t in range(3, 0, -1):
                calib_queue.put({"stage": "countdown", "prompt": f"Sing /{v}/ in {t}..."})
                time.sleep(1.0)

            def show_live_spectrogram(y, sr):
                f, t, Sxx = spectrogram(y, fs=sr, nperseg=1024, noverlap=512)
                plt.clf()
                plt.pcolormesh(t, f, 10*np.log10(Sxx+1e-12), shading='gouraud')
                plt.ylim(0, 4000)
                plt.title("Live Spectrogram")
                plt.pause(0.01)
            # capture with live countdown
            y = []
            remaining = int(duration)
            while remaining > 0:
                chunk = capture_segment(sr, 1.0)
                y.extend(chunk)
                show_live_spectrogram(np.array(y), sr)
                remaining -= 1
                if remaining > 0:
                    calib_queue.put({"stage": "countdown", "prompt": f"Keep singing /{v}/... {remaining}s left"})

            # normalize amplitude
            y = np.array(y, dtype=float)
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))

            # analyze with YIN pitch
            result = analyze_segment(y, sr)
            try:
                f0_series = librosa.yin(y, fmin=50, fmax=500, sr=sr)
                result["f0"] = float(np.nanmean(f0_series))
            except Exception:
                result["f0"] = None

            # fallback if LPC failed
            if result.get("f1") is None or result.get("f2") is None:
                f1_fb, f2_fb = spectral_fallback_from_frames(result["frames_list"], sr)
                if result.get("f1") is None:
                    result["f1"] = f1_fb
                if result.get("f2") is None:
                    result["f2"] = f2_fb

            # plausibility check
            reason = "ok"
            if result.get("f1") is None or result.get("f2") is None:
                reason = "missing formant"
            elif result["f1"] > 2000:
                reason = f"f1 out of range ({int(result['f1'])} Hz)"

            result["reason"] = reason
            result["retries"] = retries

            calib_queue.put({"stage": "analysis", "vowel": v, "result": result})

            if reason == "ok":
                success = True
                user_formants[v] = (result["f1"], result["f2"], result["f0"])
            else:
                retries += 1
                if retries < 2:
                    calib_queue.put({"stage": "countdown", "prompt": f"Retry /{v}/, formants unstable"})

    # finished
    calib_queue.put({"stage": "done", "profile": profile_name})

    # persist profile
    try:
        profile_dict = normalize_profile_for_save(user_formants)
        dump_live_profile(profile_name, profile_dict)
        logger.info("Saved calibration profile for %s", profile_name)
        global _calib_result
        _calib_result["done"] = True
        _calib_result["success"] = True
    except Exception:
        logger.exception("Failed to save calibration profile")
        _calib_result["done"] = True
        _calib_result["success"] = False


def _check_calibration_completion():
    """Call this from poll_queue at the top to detect when calibration finishes."""
    global _calib_thread, _calib_result, _calibrating

    if not (_calib_thread and isinstance(_calib_result, dict) and _calib_result.get("done")):
        return

    try:
        if _calib_result.get("success"):
            close_window()
            
            if hasattr(ax_status, "clear"):
                ax_status.clear()
            ax_status.text(0.05, 0.8, f"Calibration complete: {_calib_result.get('profile')}", transform=ax_status.transAxes, color="green")
        else:
            if hasattr(ax_status, "clear"):
                ax_status.clear()
            ax_status.text(0.05, 0.8, "Calibration failed. See console.", transform=ax_status.transAxes, color="red")

        fig.canvas.draw_idle()

        # auto-select the new profile if present
        try:
            rb = refresh_profiles()  # modify refresh_profiles to return the RadioButtons instance
            if rb is not None:
                labels = [lab.get_text() for lab in rb.labels]
                if _calib_result.get("profile") in labels:
                    rb.set_active(labels.index(_calib_result["profile"]))
        except Exception:
            logger.exception("Auto-select profile failed")

       # robust mic restart: create a persistent InputStream and store it in globals
        try:
            mic_obj = globals().get("mic", None)
            if mic_obj is None:
                mic_obj = sd.InputStream(callback=audio_callback, channels=1, samplerate=44100, blocksize=2048)
                mic_obj.start()
                globals()["mic"] = mic_obj
                logger.info("Created and started persistent mic after calibration")
            else:
                if not getattr(mic_obj, "active", False):
                    try:
                        mic_obj.start()
                        logger.info("Restarted existing mic after calibration")
                    except Exception:
                        # if restart fails, recreate
                        mic_obj = sd.InputStream(callback=audio_callback, channels=1, samplerate=44100, blocksize=2048)
                        mic_obj.start()
                        globals()["mic"] = mic_obj
                        logger.info("Recreated and started mic after calibration")
        except Exception:
            logger.exception("Failed to ensure persistent mic after calibration")
  
        # Re-enable button visually and reset guard
        try:
            calib_button.eventson = True
            calib_button.label.set_text("Calibrate")
            calib_button.ax.set_facecolor("0.95")
        except Exception:
            logger.exception("Failed to re-enable calib_button UI")

    finally:
        # clear thread reference and guard
        try:
            _calib_thread = None
            _calibrating = False
        except Exception:
            logger.exception("Failed to reset calibration state")


analyzer = Analyzer()
smoother = LabelSmoother(window=5, min_dwell=2)


def close_window():
    try:
        root = tk._default_root
        if root:
            root.destroy()
    except Exception:
        pass
    sys.exit(0)

def poll_calib_queue(_frame):
    _check_calibration_completion()
    updated = False
    while not calib_queue.empty():
        msg = calib_queue.get_nowait()
        if not isinstance(msg, dict):
            continue
        stage = msg.get("stage")
        if stage == "countdown":
            poll_calib_queue._prompt_text.set_text(msg.get("prompt", ""))
            poll_calib_queue._prompt_text.set_color("orange")
            updated = True
        elif stage == "analysis":
            result = msg.get("result", {})
            f1, f2, f0 = result.get("f1"), result.get("f2"), result.get("f0")
            f1_str = f"{f1:.0f}" if f1 is not None else "?"
            f2_str = f"{f2:.0f}" if f2 is not None else "?"
            f0_str = f"{f0:.0f}" if f0 is not None else "?"
            vowel = msg.get("vowel", "?")
            poll_calib_queue._status_text.set_text(f"/{vowel}/ F1={f1_str} F2={f2_str} F0={f0_str}")
            poll_calib_queue._status_text.set_color("green")
            updated = True
        elif stage == "done":
            profile = msg.get("profile", "")
            poll_calib_queue._status_text.set_text(f"Calibration complete: {profile}")
            poll_calib_queue._status_text.set_color("blue")
            # clear the countdown prompt at completion
            poll_calib_queue._prompt_text.set_text("")
            updated = True
    if updated:
        fig.canvas.draw_idle()


def poll_mic_queue():
    updated = False
    while not results_queue.empty():
        status = results_queue.get_nowait()

        # Pitch slider synced to f0
        if status.get("f0"):
            try:
                pitch_slider.set_val(float(status["f0"]))
            except Exception:
                pass

        # Measured formants overlay on chart
        measured = status.get("formants", (np.nan, np.nan, np.nan))
        for a in getattr(ax_chart, "_measured_overlay", []):
            try:
                a.remove()
            except Exception:
                pass
        ax_chart._measured_overlay = []
        p = ax_chart.scatter(measured[1], measured[0], c="red", s=60, zorder=10)
        ax_chart._measured_overlay.append(p)

        # Update info text color based on overall score
        overall = status.get("overall", None)
        if overall is None:
            fig._info_text.set_color("purple")
        else:
            fig._info_text.set_color("green" if overall >= 80 else
                                     "orange" if overall >= 50 else "red")

        # Minimal info line
        f1, f2, f3 = measured
        mf_disp = [f"{int(x)}" if (x is not None and not np.isnan(x)) else "?" for x in [f1, f2, f3]]
        fig._info_text.set_text(
            f"Voice type={voice_type}, Target=/{current_vowel_name}/\n"
            f"F1={mf_disp[0]}, F2={mf_disp[1]}, F3={mf_disp[2]}\n"
            f"Score={status.get('vowel_score', 0)}, "
            f"Resonance={status.get('resonance_score', 0)}, "
            f"Overall={overall if overall is not None else '--'}"
        )

        updated = True

    if updated:
        fig.canvas.draw_idle()

    # Re-arm the Tk timer (every 100 ms)
    fig.canvas.get_tk_widget().after(100, poll_calib_queue)
    fig.canvas.get_tk_widget().after(100, poll_mic_queue)


# Create persistent text artists once, stacked neatly
poll_calib_queue._prompt_text = ax_status.text(
    0.05, 0.95, "", transform=ax_status.transAxes,
    color="orange", va="top"
)   # prep + live countdown (top)

poll_calib_queue._status_text = ax_status.text(
    0.05, 0.80, "", transform=ax_status.transAxes,
    color="green", va="top"
)   # analysis results (middle)

poll_calib_queue._profile_text = ax_status.text(
    0.05, 0.65, "", transform=ax_status.transAxes,
    color="brown", va="top"
)   # profile summary (below analysis)

poll_calib_queue._note_text = ax_status.text(
    0.05, 0.50, "", transform=ax_status.transAxes,
    color="blue", va="top"
)   # optional note/pitch info

poll_calib_queue._vowel_text = ax_status.text(
    0.05, 0.35, "", transform=ax_status.transAxes,
    color="purple", va="top"
)   # optional vowel label

ax_status.axis("off")
fig.canvas.draw_idle()

print("_calib_thread in globals:", "_calib_thread" in globals())
print("_calib_thread:", globals().get("_calib_thread"))
print("is_alive:", getattr(globals().get("_calib_thread"), "is_alive", lambda: False)())
pprint.pprint(_calib_result)
print("mic exists:", "mic" in globals(), "mic.active:", getattr(globals().get("mic", None), "active", None))

_cleanup_done = False

def cleanup():
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    try:
        # Stop calibration thread if running (daemon threads exit with process; we avoid join here)
        # Stop and close mic
        if 'mic' in globals():
            try:
                if getattr(mic, "active", False):
                    mic.stop()
            except Exception:
                pass
            try:
                mic.close()
            except Exception:
                pass
        sd.stop()
        print("Audio stream closed.")
    except Exception as e:
        print("Cleanup error:", e)

atexit.register(cleanup)
fig.canvas.mpl_connect('close_event', lambda event: cleanup())

ani = animation.FuncAnimation(fig, poll_calib_queue, interval=200, cache_frame_data=False)
plt.show()


if __name__ == "__main__":
    run_calibration(None)
