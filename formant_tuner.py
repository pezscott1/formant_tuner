import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider, Button
from matplotlib.patches import Ellipse
import os
import sounddevice as sd
from scipy.signal import spectrogram
import queue
from collections import deque
from vowel_data import FORMANTS, NOTE_NAMES
from voice_analysis import Analyzer
from formant_utils import robust_guess, is_plausible_formants, plausibility_score, choose_best_candidate
import subprocess
import glob
import tkinter as tk
from tkinter import ttk
from mic_analyzer import MicAnalyzer, results_queue


analyzer = Analyzer(voice_type="bass", smoothing=True, smooth_size=5)

PROFILE_PATH = os.path.join("profiles", "user1_profile.json")
MODEL_PATH = os.path.join("profiles", "user1_model.pkl")
if os.path.exists(PROFILE_PATH):
    analyzer.load_profile(PROFILE_PATH, model_path=MODEL_PATH)


recent_results = []

def process_status(status):
    f1, f2 = status["formants"][:2]
    ok, reason = is_plausible_formants(f1, f2)
    if not ok:
        return None  # skip implausible frame

    recent_results.append(status)
    if len(recent_results) >= 5:  # smooth over 5 frames
        best = choose_best_candidate(recent_results[0], recent_results[1:])
        recent_results.clear()
        return best
    return None


def harmonic_series(f0, num_harmonics=12):
    return np.arange(1, num_harmonics + 1) * f0


def spectral_envelope(formants, freqs, bandwidth=100):
    envelope = np.zeros_like(freqs)
    for f in formants:
        envelope += np.exp(-0.5 * ((freqs - f) / bandwidth) ** 2)
    return envelope


def score_vowel_pitch(formants, pitch, tolerance=50, max_score=100):
    harmonics = harmonic_series(pitch, num_harmonics=12)
    score = 0
    total_possible = len(formants) * 10
    for f in formants:
        closest = min(harmonics, key=lambda h: abs(h - f))
        distance = abs(closest - f)
        if distance <= tolerance:
            points = max(0, 10 - (distance / tolerance) * 10)
            score += points
    return int((score / total_possible) * max_score)


def score_against_profile(measured_formants, user_profile, vowel, tolerance=50, max_score=100):
    if not user_profile or vowel not in user_profile:
        return None
    entry = user_profile[vowel]
    target_f1 = entry.get("f1")
    target_f2 = entry.get("f2")
    measured_f1, measured_f2 = measured_formants[0], measured_formants[1]
    score = 0
    total_possible = 20
    if target_f1 and measured_f1:
        dist = abs(measured_f1 - target_f1)
        if dist <= tolerance:
            score += max(0, 10 - (dist / tolerance) * 10)
    if target_f2 and measured_f2:
        dist = abs(measured_f2 - target_f2)
        if dist <= tolerance:
            score += max(0, 10 - (dist / tolerance) * 10)
    return int((score / total_possible) * max_score)


def directional_feedback(measured_formants, user_profile, vowel, tolerance=50):
    if not user_profile or vowel not in user_profile:
        return None, None
    target_f1 = user_profile[vowel].get("f1")
    target_f2 = user_profile[vowel].get("f2")
    measured_f1, measured_f2 = measured_formants[0], measured_formants[1]
    fb_f1 = None
    fb_f2 = None
    if measured_f1 and target_f1:
        diff = measured_f1 - target_f1
        if abs(diff) > tolerance:
            fb_f1 = "lower F1" if diff > 0 else "raise F1"
    if measured_f2 and target_f2:
        diff = measured_f2 - target_f2
        if abs(diff) > tolerance:
            fb_f2 = "lower F2" if diff > 0 else "raise F2"
    return fb_f1, fb_f2


def freq_to_note_name(freq):
    if freq <= 0:
        return "N/A"
    midi = int(round(69 + 12 * np.log2(freq / 440.0)))
    name = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{name}{octave}"


def play_pitch(frequency, duration=2.0, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 0.2 * np.sin(2 * np.pi * frequency * t)
    sd.play(waveform, sample_rate)
    sd.wait()


def interactive_vowel_chart(initial_pitch=261.63, voice_type='bass', headless=False):
    analyzer.voice_type = voice_type
    vowels = FORMANTS[voice_type]

    # Main figure and two panels
    fig, (ax_chart, ax_spec) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(left=0.36, bottom=0.25)

    # Persistent info text
    info_text = fig.text(0.02, 0.02,
                         "Measured: F1=?, F2=?, F3=?\nVowel=--, Resonance=--, Overall=--",
                         ha='left', va='bottom', fontsize=10, color='purple')
    fig._info_text = info_text

    # Vowel chart points
    points = {}
    ax_chart._measured_overlay = []
    for v, (F1, F2, F3, FS) in vowels.items():
        pt = ax_chart.scatter(F2, F1, label=f"/{v}/", picker=True, s=100, c='blue')
        points[pt] = (v, (F1, F2, F3, FS))
        ax_chart.text(F2 + 50, F1 + 30, f"/{v}/", fontsize=10)
        ax_chart.add_patch(Ellipse((F2, F1), width=200, height=100, alpha=0.2, color='blue'))

    ax_chart.set_title(f"Vowel Chart ({voice_type})")
    ax_chart.set_xlabel("F2 (Hz)")
    ax_chart.set_ylabel("F1 (Hz)")
    ax_chart.invert_xaxis()
    ax_chart.invert_yaxis()

    # Spectrum state
    freq_axis = np.linspace(0, 4000, 1000)
    current_vowel_name = 'a' if 'a' in vowels else next(iter(vowels))
    current_formants = vowels[current_vowel_name]
    formant_history = deque(maxlen=7)

    # Control panel
    control_left = 0.02
    control_width = 0.28
    control_height = 0.05
    label_offset = 1.12

    ax_pitch_ctrl = plt.axes([control_left, 0.72, control_width, control_height])
    pitch_slider = Slider(ax_pitch_ctrl, '', 100, 600, valinit=initial_pitch, valstep=0.1)
    ax_pitch_ctrl.text(0.0, label_offset, "Pitch (Hz)", transform=ax_pitch_ctrl.transAxes,
                       fontsize=9, ha='left', va='bottom')

    ax_tol_ctrl = plt.axes([control_left, 0.65, control_width, control_height])
    tol_slider = Slider(ax_tol_ctrl, '', 10, 200, valinit=50, valstep=1)
    ax_tol_ctrl.text(0.0, label_offset, "Tolerance (Hz)", transform=ax_tol_ctrl.transAxes,
                     fontsize=9, ha='left', va='bottom')

    PROFILE_DIR = "profiles"
    try:
        profile_files = [f for f in os.listdir(PROFILE_DIR) if f.endswith("_profile.json")]
    except FileNotFoundError:
        profile_files = []

    if not profile_files:
        profile_files = ["(none)"]

    # Add special option for calibration
    profile_files.append("setup profile")


    def on_profile_select(label):
        if label != "(none)":
            profile_path = os.path.join(PROFILE_DIR, label)
            model_path = profile_path.replace("_profile.json", "_model.pkl")
            analyzer.load_profile(profile_path, model_path=model_path)
            print(f"Loaded profile {label}")
            # FIX: add measured_formants placeholder
            update_spectrum(current_vowel_name, current_formants,
                            (np.nan, np.nan, np.nan),
                            pitch_slider.val, tol_slider.val)


    def add_profile_dropdown(fig, profiles, on_select):
        canvas = fig.canvas.get_tk_widget()
        root = canvas.winfo_toplevel()

        label = ttk.Label(root, text="User Profile:", font=("Arial", 14))
        label.place(x=20, y=60, width=200, height=30)

        combo = ttk.Combobox(root, values=profiles, state="readonly", font=("Arial", 14))
        combo.place(x=20, y=95, width=280, height=35)  # lower and wider
        combo.current(0)

        def handler(event):
            on_select(combo.get())

        combo.bind("<<ComboboxSelected>>", handler)
        return combo

                
    # Place dropdown in upper-left corner of the window
    profile_dropdown = add_profile_dropdown(fig, profile_files, on_profile_select)

    # Buttons
    ax_btn_start = plt.axes([control_left, 0.55, control_width, control_height])
    ax_btn_stop = plt.axes([control_left, 0.48, control_width, control_height])
    ax_btn_play = plt.axes([control_left, 0.41, control_width, control_height])
    ax_btn_spec = plt.axes([control_left, 0.34, control_width, control_height])

    btn_start = Button(ax_btn_start, 'Start Mic')
    btn_stop = Button(ax_btn_stop, 'Stop Mic')
    btn_play = Button(ax_btn_play, 'Play Pitch')
    btn_spec = Button(ax_btn_spec, 'Show Spectrogram')

    ax_btn_calib = plt.axes([control_left, 0.27, control_width, control_height])
    btn_calib = Button(ax_btn_calib, 'Calibrate')


    def launch_calibration(event):
        # spawn calibration.py in a new Python process
        proc = subprocess.Popen(["python", "calibration.py"])
        proc.wait()  # block until calibration.py exits

        # once calibration.py exits, reload newest profile
        files = glob.glob(os.path.join(PROFILE_DIR, "*_profile.json"))
        if files:
            newest = max(files, key=os.path.getmtime)
            model_path = newest.replace("_profile.json", "_model.pkl")
            analyzer.load_profile(newest, model_path=model_path)
            print(f"Calibration complete. Loaded {os.path.basename(newest)}")
            profile_dropdown.set(os.path.basename(newest))


    btn_calib.on_clicked(launch_calibration)

    mic = MicAnalyzer(
        vowel_provider=lambda: current_formants,
        tol_provider=lambda: tol_slider.val,
        pitch_provider=lambda: pitch_slider.val,
        sample_rate=44100,  
        frame_ms=80,        
        analyzer=analyzer 
    )

    # Event handlers
    def on_pick(event):
        if event.artist in points:
            vname, formants = points[event.artist]
            nonlocal current_vowel_name, current_formants
            current_vowel_name = vname
            current_formants = formants
            update_spectrum(vname, formants, pitch_slider.val, tol_slider.val)
            # No fake measured yet
            fig._info_text.set_color("purple")
            fig.canvas.draw_idle()

    def freq_to_midi(f):
        return 69.0 + 12.0 * np.log2(f / 440.0)

    def midi_to_freq(m):
        return 440.0 * 2.0 ** ((m - 69.0) / 12.0)

    def on_pitch_change(val):
        m_cont = freq_to_midi(val)
        m_round = int(round(m_cont))
        f_snap = midi_to_freq(m_round)
        if abs(f_snap - pitch_slider.val) > 1e-6:
            pitch_slider.eventson = False
            pitch_slider.set_val(f_snap)
            pitch_slider.eventson = True
        update_spectrum(current_vowel_name, current_formants, (np.nan, np.nan, np.nan), f_snap, tol_slider.val)

    def update_voice(label):
        nonlocal vowels, points, current_formants, voice_type
        voice_type = label
        analyzer.voice_type = label
        vowels = FORMANTS[label]

        ax_chart.clear()
        points.clear()
        for v, (F1, F2, F3, FS) in vowels.items():
            pt = ax_chart.scatter(F2, F1, label=f"/{v}/", picker=True, s=100, c='blue')
            points[pt] = (v, (F1, F2, F3, FS))
            ax_chart.text(F2 + 50, F1 + 30, f"/{v}/", fontsize=10)
            ax_chart.add_patch(Ellipse((F2, F1), width=200, height=100, alpha=0.2, color='blue'))
        ax_chart.set_title(f"Vowel Chart ({label})")
        ax_chart.set_xlabel("F2 (Hz)")
        ax_chart.set_ylabel("F1 (Hz)")
        ax_chart.invert_xaxis()
        ax_chart.invert_yaxis()

        if current_vowel_name not in vowels:
            current_vowel_name = 'a' if 'a' in vowels else next(iter(vowels))
            current_formants = vowels[current_vowel_name]
        update_spectrum(current_vowel_name, current_formants, pitch_slider.val, tol_slider.val)
        fig.canvas.draw_idle()

    def on_profile_select(label):
        if label != "(none)":
            profile_path = os.path.join(PROFILE_DIR, label)
            model_path = profile_path.replace("_profile.json", "_model.pkl")
            analyzer.load_profile(profile_path, model_path=model_path)
            print(f"Loaded profile {label}")
            update_spectrum(current_vowel_name, current_formants, pitch_slider.val, tol_slider.val)

    def show_spectrogram(event=None):
        if mic is None or len(mic.buffer) < 1000:
            print("Not enough audio captured for spectrogram.")
            return
        sig = np.array(mic.buffer)
        f, t, Sxx = spectrogram(sig, fs=mic.sample_rate, nperseg=1024, noverlap=512)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        pcm = ax2.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud')
        ax2.set_ylim(0, 4000)
        ax2.set_title("Spectrogram of captured audio")
        fig2.colorbar(pcm, ax=ax2, label='Power (dB)')
        plt.show()


    pitch_slider.on_changed(on_pitch_change)
    tol_slider.on_changed(lambda val: update_spectrum(current_vowel_name, current_formants, pitch_slider.val, val))
    btn_start.on_clicked(lambda e: mic.start())
    btn_stop.on_clicked(lambda e: mic.stop())
    btn_play.on_clicked(lambda e: play_pitch(pitch_slider.val))
    btn_spec.on_clicked(show_spectrogram)
    fig.canvas.mpl_connect('pick_event', on_pick)

    # Initial spectrum
    def update_spectrum(vowel, target_formants, measured_formants, pitch, tolerance):
        ax_spec.cla()
        target = (target_formants[0], target_formants[1], target_formants[2])
        hs = [h for h in harmonic_series(pitch, num_harmonics=12) if 0 <= h <= 4000]
        if not hs:
            ax_spec.set_xlim(0, 4000)
            ax_spec.set_ylim(0, 1)
            ax_spec.set_title("No harmonics in range")
            fig.canvas.draw_idle()
            return
        amplitudes = []
        for h in hs:
            boost = 1.0
            for f in target:
                if abs(h - f) <= tolerance:
                    boost += 2.0
            amplitudes.append(boost)
        amplitudes = np.array(amplitudes)
        ax_spec.stem(hs, amplitudes, linefmt='gray', markerfmt='o', basefmt=" ")
        ax_spec.plot(freq_axis := np.linspace(0, 4000, 1000),
                     spectral_envelope(target, freq_axis), 'r-', linewidth=2, label="Filter Envelope")
        for f in target:
            ax_spec.axvline(f, color='blue', linestyle='--', alpha=0.5)
        score = score_vowel_pitch(target, pitch, tolerance)
        note_name = freq_to_note_name(pitch)
        ax_spec.set_xlim(0, 4000)
        ax_spec.set_ylim(0, max(amplitudes) + 1)
        ax_spec.set_title(f"Spectrum /{vowel}/ ({voice_type}, {note_name} {pitch:.2f} Hz, Score={score})")
        ax_spec.set_xlabel("Frequency (Hz)")
        ax_spec.set_ylabel("Amplitude (a.u.)")
        ax_spec.legend(loc='upper right')
        # reset all markers to blue, then highlight current vowel by score/profile
        for pt, (vname, _) in points.items():
            try:
                pt.set_facecolor('blue')
            except Exception:
                pt.set_facecolors(['blue'])
        for pt, (vname, _) in points.items():
            if vname == vowel:
                if analyzer.user_formants:
                    profile_score = score_against_profile(
                        measured_formants=(measured_formants[0], measured_formants[1]),
                        user_profile=analyzer.user_formants,
                        vowel=vowel,
                        tolerance=tolerance
                    )

                    if profile_score is not None:
                        color = 'green' if profile_score >= 80 else 'yellow' if profile_score >= 50 else 'red'
                    else:
                        color = 'gray'
                else:
                    color = 'green' if score >= 80 else 'yellow' if score >= 50 else 'red'
                try:
                    pt.set_facecolor(color)
                except Exception:
                    pt.set_facecolors([color])
        fig.canvas.draw_idle()


    # Poll queue in main thread via Tk timer
    def poll_queue():
        updated = False
        while not results_queue.empty():
            raw_status = results_queue.get_nowait()
            stable_status = process_status(raw_status)
            if not stable_status:
                continue  # skip implausible/noisy frame

            # Use stable_status consistently
            f0 = stable_status.get("f0")
            f1, f2, f3 = stable_status.get("formants", (np.nan, np.nan, np.nan))

            # Sync pitch slider
            if f0:
                try:
                    pitch_slider.set_val(float(f0))
                except Exception:
                    pass

            # Measured formants overlay on chart
            measured = (f1, f2, f3)
            update_spectrum(
                current_vowel_name,
                current_formants,          # chart tuple
                measured,                  # mic values from queue
                pitch_slider.val,
                tol_slider.val
            )
            for a in getattr(ax_chart, "_measured_overlay", []):
                try:
                    a.remove()
                except Exception:
                    pass
            ax_chart._measured_overlay = []
            p = ax_chart.scatter(measured[1], measured[0], c="red", s=60, zorder=10)
            ax_chart._measured_overlay.append(p)

            # Update info text color based on overall
            overall = stable_status.get("overall", None)
            if overall is None:
                fig._info_text.set_color("purple")
            else:
                fig._info_text.set_color(
                    "green" if overall >= 80 else
                    "orange" if overall >= 50 else
                    "red"
                )

            # Minimal info line
            mf_disp = [f"{int(x)}" if (x is not None and not np.isnan(x)) else "?"
                    for x in measured]
            fb_f1 = stable_status.get("fb_f1", "")
            fb_f2 = stable_status.get("fb_f2", "")

            fig._info_text.set_text(
                f"Voice type={voice_type}, Target=/{current_vowel_name}/\n"
                f"F1={mf_disp[0]} {fb_f1}, F2={mf_disp[1]} {fb_f2}, F3={mf_disp[2]}\n"
                f"Score={stable_status.get('vowel_score', 0)}, "
                f"Resonance={stable_status.get('resonance_score', 0)}, "
                f"Overall={overall if overall is not None else '--'}"
            )

            updated = True

        if updated:
            fig.canvas.draw_idle()

        # Re-arm the Tk timer (every 100 ms)
        fig.canvas.get_tk_widget().after(100, poll_queue)

    # Kick off polling loop once
    fig.canvas.get_tk_widget().after(100, poll_queue)

    if headless:
        return fig, mic, fig._info_text, None

    plt.show()


if __name__ == "__main__":
    interactive_vowel_chart()
