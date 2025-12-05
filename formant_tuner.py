
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider, Button
from matplotlib.patches import Ellipse
import sounddevice as sd
from scipy.signal import lfilter, spectrogram
from scipy.signal.windows import hamming
from numpy.linalg import lstsq
import matplotlib.animation as animation
import queue



FORMANTS = { 'tenor': { 'i': (320, 2290, 3000, 3500), 'e': (580, 2000, 3000, 3300), 'a': (800, 1200, 2800, 3200), 'o': (640, 1000, 2700, 3100), 'u': (350, 950, 2600, 3000), 'æ': (700, 1800, 2800, 3200), 'ʌ': (680, 1300, 2700, 3100), 'ɔ': (540, 800, 2600, 3000) }, 'soprano': { 'i': (400, 2700, 3500, 3800), 'e': (650, 2300, 3300, 3600), 'a': (950, 1500, 3100, 3400), 'o': (750, 1200, 3000, 3300), 'u': (420, 1100, 2900, 3200), 'æ': (800, 2000, 3100, 3400), 'ʌ': (780, 1450, 3000, 3300), 'ɔ': (600, 950, 2900, 3200) } }

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']



def harmonic_series(f0, num_harmonics=12):
    return np.arange(1, num_harmonics + 1) * f0

def spectral_envelope(formants, freqs, bandwidth=100):
    envelope = np.zeros_like(freqs)
    for f in formants: envelope += np.exp(-0.5 * ((freqs - f) / bandwidth) ** 2)
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


def estimate_formants_lpc(signal, sample_rate=44100, lpc_order=12, fmin=200, fmax=4000):
    preemph = lfilter([1, -0.97], 1, signal)
    win = preemph * hamming(len(preemph))
    win = win - np.mean(win)
    r = np.correlate(win, win, mode='full')[len(win) - 1:]
    R = np.array([r[i:i + lpc_order] for i in range(lpc_order)])
    rhs = -r[1:lpc_order + 1]
    a, _, _, _ = lstsq(R, rhs, rcond=None)
    a = np.concatenate(([1.0], a))
    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0]
    angs = np.arctan2(np.imag(roots), np.real(roots))
    formants = sorted(angs * (sample_rate / (2 * np.pi)))
    formants = [f for f in formants if fmin < f < fmax]

    # Enforce minimal spacing
    filtered = []
    for f in formants:
        if not filtered or min(abs(f - g) for g in filtered) > 150:
            filtered.append(f)
    return filtered[:3]


def align_formants_to_targets(measured, target):
    measured = list(measured or [])  # tolerate None -> []
    assigned = [None, None, None]
    used = set()
    for ti, t in enumerate(target):
        best_idx, best_d = None, float('inf')
        for mi, m in enumerate(measured):
            if mi in used:
                continue
            d = abs(m - t)
            if d < best_d:
                best_idx, best_d = mi, d
        if best_idx is not None:
            assigned[ti] = measured[best_idx]
            used.add(best_idx)
    return assigned


def live_score_formants(target_formants, measured_formants, tolerance=50, weights=(0.4, 0.4, 0.2)):
    aligned = align_formants_to_targets(measured_formants, target_formants)
    score = 0.0
    total = sum(weights)
    for i, w in enumerate(weights):
        m = aligned[i]
        if m is None:
            continue
        d = abs(m - target_formants[i])
        if d <= tolerance:
            s = max(0.0, 1.0 - d / tolerance)
            score += w * s
    return int(100 * (score / total))


def resonance_tuning_score(measured_formants, pitch, tolerance=50):
    harmonics = harmonic_series(pitch, num_harmonics=12)
    scores = []
    for f in measured_formants or []:
        closest = min(harmonics, key=lambda h: abs(h - f))
        d = abs(closest - f)
        if d <= tolerance:
            s = max(0, 100 - (d / tolerance) * 100)
        else:
            s = 0
        scores.append(s)
    return int(np.mean(scores)) if scores else 0


def overall_rating(measured_formants, target_formants, pitch, tolerance=50):
    vowel_score = live_score_formants(target_formants, measured_formants, tolerance)
    resonance_score = resonance_tuning_score(measured_formants, pitch, tolerance)
    return vowel_score, resonance_score, int(0.5 * vowel_score + 0.5 * resonance_score)


results_queue = queue.Queue()


class MicAnalyzer:
    def __init__(self, vowel_provider, tol_provider, pitch_provider,
                 sample_rate=44100, frame_ms=80):
        self.get_vowel = vowel_provider
        self.get_tol = tol_provider
        self.get_pitch = pitch_provider
        self.sample_rate = sample_rate
        self.frame_len = int(sample_rate * frame_ms / 1000.0)
        self.stream = None
        self.buffer = []

    def audio_callback(self, indata, frames, time, status):
        if status:
            return
        mono = indata[:, 0]
        if np.max(np.abs(mono)) < 2e-3:
            return
        try:
            self.buffer.extend(mono.tolist())
            measured = estimate_formants_lpc(mono, sample_rate=self.sample_rate)  # always list (possibly empty)
            F1, F2, F3, FS = self.get_vowel()
            target = (F1, F2, F3)
            tol = self.get_tol()
            pitch = self.get_pitch()

            # If no formants detected in this frame, still push a safe payload
            vowel_score, resonance_score, overall = overall_rating(measured, target, pitch, tol)
            results_queue.put((measured, vowel_score, resonance_score, overall))
        except Exception as e:
            # Optional: log minimal info; never crash the audio thread
            print(f"Audio callback error: {e}")
            return

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def start(self):
        if self.stream is not None:
            return
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.frame_len,
            callback=self.audio_callback
        )
        self.stream.start()


def interactive_vowel_chart(initial_pitch=261.63, voice_type='tenor', headless=False):
    vowels = FORMANTS[voice_type]
    fig, (ax_chart, ax_spec) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Vowel chart ---
    points = {}
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

    # --- Spectrum state ---
    freq_axis = np.linspace(0, 4000, 1000)
    current_vowel_name = 'a'
    current_formants = vowels[current_vowel_name]

    # --- Live text overlay on the spectrum axis ---
    live_text = ax_spec.text(
        0.01, 0.95,
        "Measured: F1=?, F2=?, F3=? |\nVowel=--, Resonance=--, Overall=--",
        transform=ax_spec.transAxes,
        fontsize=10,
        color='purple',
        verticalalignment='top',
        wrap=True
    )
    ax_spec._measured_lines = []
    fig.canvas.draw_idle()

    def current_target_formants():
        F1, F2, F3, FS = current_formants
        return (F1, F2, F3)

    # --- Spectrum redraw ---
    def update_spectrum(vowel, formants, pitch, tolerance):
        ax_spec.cla()
        target = (formants[0], formants[1], formants[2])
        # compute harmonics and keep only those inside the x-axis limits
        harmonics = harmonic_series(pitch, num_harmonics=12)
        hs = [h for h in harmonics if 0 <= h <= 4000]

        # if no harmonics fall in range, avoid plotting mismatch
        if not hs:
            ax_spec.cla()
            ax_spec.set_xlim(0, 4000)
            ax_spec.set_ylim(0, 1)
            ax_spec.set_title("No harmonics in range")
            return

        # compute amplitudes aligned with hs
        amplitudes = []
        for h in hs:
            boost = 1.0
            for f in target:
                if abs(h - f) <= tolerance:
                    boost += 2.0
            amplitudes.append(boost)

        amplitudes = np.array(amplitudes)

        # plot stems using hs and amplitudes (same length)
        ax_spec.stem(hs, amplitudes, linefmt='gray', markerfmt='o', basefmt=" ")

        ax_spec.plot(freq_axis, spectral_envelope(target, freq_axis), 'r-', linewidth=2, label="Filter Envelope")
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

        nonlocal live_text
        prev_msg = getattr(live_text, 'get_text', lambda: None)() or \
                   "Measured: F1=?, F2=?, F3=? |\n Vowel=--, Resonance=--, Overall=--"
        prev_color = getattr(live_text, 'get_color', lambda: 'purple')()
        live_text = ax_spec.text(0.05, 0.85, prev_msg, transform=ax_spec.transAxes,
                                 fontsize=10, color=prev_color, verticalalignment='top')
        ax_spec._measured_lines = []

        # reset all markers to default blue
        for pt, (vname, _) in points.items():
            try:
                pt.set_facecolor('blue')
            except Exception:
                pt.set_facecolors(['blue'])

        # then highlight the selected vowel
        for pt, (vname, _) in points.items():
            if vname == vowel:
                color = 'green' if score >= 80 else 'yellow' if score >= 50 else 'red'
                try:
                    pt.set_facecolor(color)
                except Exception:
                    pt.set_facecolors([color])

        for ax in (ax_pitch_ctrl, ax_tol_ctrl):
            for t in ax.texts:
                t.set_fontsize(9)

        fig.canvas.draw_idle()

    # --- Live UI updater ---
    def update_live_ui(measured_formants, vowel_score, resonance_score, overall_score):
        for line in getattr(ax_spec, "_measured_lines", []):
            try:
                line.remove()
            except Exception:
                pass
        ax_spec._measured_lines = []
        for f in measured_formants:
            ax_spec._measured_lines.append(ax_spec.axvline(f, color='red', linestyle='-', alpha=0.7))

        aligned = align_formants_to_targets(measured_formants, current_target_formants())
        mf_disp = [f"{int(x)}" if x is not None else "?" for x in aligned]

        live_text.set_text(
            f"Measured: F1={mf_disp[0]}, F2={mf_disp[1]}, F3={mf_disp[2]} | \n"
            f"Vowel={vowel_score}, Resonance={resonance_score}, Overall={overall_score}"
        )
        live_text.set_color("green" if overall_score >= 80 else "orange" if overall_score >= 50 else "red")
        fig.canvas.draw_idle()

    def update_voice(label):
        # we reassign these outer variables, so declare nonlocal
        nonlocal vowels, points, current_vowel_name, current_formants, voice_type
        voice_type = label
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

        # sensible default after switching voice type
        current_vowel_name = 'a' if 'a' in vowels else next(iter(vowels))
        current_formants = vowels[current_vowel_name]
        update_spectrum(current_vowel_name, current_formants, pitch_slider.val, tol_slider.val)
        fig.canvas.draw_idle()


    # --- Control panel layout (single authoritative block) ---
    plt.subplots_adjust(left=0.36, bottom=0.25)
    control_left = 0.02
    control_width = 0.28
    control_height = 0.05

    # Voice type radio buttons
    rax = plt.axes([control_left, 0.80, control_width, 0.15])
    radio = RadioButtons(rax, list(FORMANTS.keys()))

    control_height = 0.05
    label_offset = 1.12  # label y-position in axis coords (slightly above the slider)

    # Pitch slider (no built-in label)
    ax_pitch_ctrl = plt.axes([control_left, 0.72, control_width, control_height])
    pitch_slider = Slider(ax_pitch_ctrl, '', 100, 600, valinit=initial_pitch, valstep=0.1)
    # place the visible label above the slider
    ax_pitch_ctrl.text(0.0, label_offset, "Pitch (Hz)", transform=ax_pitch_ctrl.transAxes,
                       fontsize=9, ha='left', va='bottom')

    # Tolerance slider (no built-in label)
    ax_tol_ctrl = plt.axes([control_left, 0.65, control_width, control_height])
    tol_slider = Slider(ax_tol_ctrl, '', 10, 200, valinit=50, valstep=1)
    ax_tol_ctrl.text(0.0, label_offset, "Tolerance (Hz)", transform=ax_tol_ctrl.transAxes,
                     fontsize=9, ha='left', va='bottom')

    # Pitch <-> MIDI helpers and snapping (single definitions)
    def freq_to_midi(f):
        return 69.0 + 12.0 * np.log2(f / 440.0)

    def midi_to_freq(m):
        return 440.0 * 2.0 ** ((m - 69.0) / 12.0)

    def snap_pitch_to_semitone(val):
        m_cont = freq_to_midi(val)
        m_round = int(round(m_cont))
        f_snap = midi_to_freq(m_round)
        pitch_slider.eventson = False
        pitch_slider.set_val(f_snap)
        pitch_slider.eventson = True
        update_spectrum(current_vowel_name, current_formants, f_snap, tol_slider.val)

    def show_note_name(val):
        m = int(round(freq_to_midi(val)))
        note = freq_to_note_name(midi_to_freq(m))
        ax_spec.set_title(f"Spectrum /{current_vowel_name}/ ({voice_type}, {note} {val:.2f} Hz, Score=--)")
        fig.canvas.draw_idle()

    # Wire pitch slider callbacks (snap first, then update title)
    pitch_slider.on_changed(snap_pitch_to_semitone)
    pitch_slider.on_changed(show_note_name)

    # Optional: draw MIDI tick markers once
    m_min = int(np.floor(freq_to_midi(pitch_slider.valmin)))
    m_max = int(np.ceil(freq_to_midi(pitch_slider.valmax)))
    tick_freqs = [midi_to_freq(m) for m in range(m_min, m_max + 1)]
    for f in tick_freqs:
        # position in axis coordinates (0..1)
        pos = (f - pitch_slider.valmin) / (pitch_slider.valmax - pitch_slider.valmin)
        ax_pitch_ctrl.axvline(pos, color='k', alpha=0.12, linewidth=0.4)

    # Instantiate mic AFTER sliders exist
    mic = MicAnalyzer(
        vowel_provider=lambda: current_formants,
        tol_provider=lambda: tol_slider.val,
        pitch_provider=lambda: pitch_slider.val
    )

    def show_spectrogram(event=None):
        # mic is instantiated later; guard if not ready
        if 'mic' not in locals() and 'mic' not in globals():
            print("Mic not instantiated yet.")
            return
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

    # Buttons (use clear names for axes)
    ax_btn_start = plt.axes([control_left, 0.55, control_width, control_height])
    btn_start = Button(ax_btn_start, 'Start Mic')
    btn_start.on_clicked(lambda e: mic.start())

    ax_btn_stop = plt.axes([control_left, 0.48, control_width, control_height])
    btn_stop = Button(ax_btn_stop, 'Stop Mic')
    btn_stop.on_clicked(lambda e: mic.stop())

    ax_btn_play = plt.axes([control_left, 0.41, control_width, control_height])
    btn_play = Button(ax_btn_play, 'Play Pitch')
    btn_play.on_clicked(lambda e: play_pitch(pitch_slider.val))

    ax_btn_spec = plt.axes([control_left, 0.34, control_width, control_height])
    btn_spec = Button(ax_btn_spec, 'Show Spectrogram')
    btn_spec.on_clicked(show_spectrogram)

    # Single slider-driven spectrum updates (no duplicates)
    pitch_slider.on_changed(
        lambda val: update_spectrum(current_vowel_name, current_formants, pitch_slider.val, tol_slider.val))
    tol_slider.on_changed(
        lambda val: update_spectrum(current_vowel_name, current_formants, pitch_slider.val, tol_slider.val))

    # Single radio handler
    radio.on_clicked(update_voice)

    # Animation loop: consume results from queue in main thread
    def poll_queue(_frame):
        while not results_queue.empty():
            measured, vowel_score, resonance_score, overall = results_queue.get()
            update_live_ui(measured, vowel_score, resonance_score, overall)
    # --- Diagnostics: run quick checks and optionally LPC debug ---
    def run_diagnostics(lpc_debug=False):
        # uses outer-scope names: mic, live_text, ax_spec, pitch_slider
        nonlocal mic, live_text

        # guard: ensure we have audio
        if mic is None:
            msg = "Diagnostics: mic not instantiated."
            live_text.set_text(msg)
            live_text.set_color("orange")
            fig.canvas.draw_idle()
            return

        sig = np.array(mic.buffer)
        if sig.size == 0:
            msg = "Diagnostics: no audio captured yet."
            live_text.set_text(msg)
            live_text.set_color("orange")
            fig.canvas.draw_idle()
            return

        sr = mic.sample_rate
        duration = sig.size / sr
        sig_min, sig_max = float(sig.min()), float(sig.max())

        # spectrogram dominant frequency (quick)
        try:
            f, t, Sxx = spectrogram(sig, fs=sr, nperseg=1024, noverlap=512)
            idx = np.argmax(Sxx, axis=0)
            dominant_freqs = f[idx]
            dom_median = float(np.median(dominant_freqs))
            dom_first = float(dominant_freqs[0]) if dominant_freqs.size else 0.0
        except Exception as e:
            dom_median = None
            dom_first = None
            spect_err = str(e)
        else:
            spect_err = None

        # LPC quick test on a steady slice (100-500 ms)
        lpc_result = None
        lpc_err = None
        if lpc_debug:
            try:
                start, end = int(0.1 * sr), int(0.5 * sr)
                frame = sig[start:end] if sig.size >= end else sig

                # use a safer LPC variant: regularized solve + magnitude filter
                def estimate_formants_safe(signal, sample_rate=sr, lpc_order=12, fmin=50, fmax=5000):
                    preemph = lfilter([1, -0.97], 1, signal)
                    win = preemph * hamming(len(preemph))
                    win = win - np.mean(win)
                    r = np.correlate(win, win, mode='full')[len(win) - 1:]
                    R = np.array([r[i:i + lpc_order] for i in range(lpc_order)])
                    rhs = -r[1:lpc_order + 1]
                    eps = 1e-6 * np.eye(R.shape[0])
                    a, _, _, _ = lstsq(R + eps, rhs, rcond=None)
                    a = np.concatenate(([1.0], a))
                    roots = np.roots(a)
                    roots = roots[np.imag(roots) > 0]
                    angs = np.angle(roots)
                    formants = angs * (sample_rate / (2 * np.pi))
                    # magnitude filter and f-range clamp
                    formants = [f for r, f in zip(roots, formants) if np.abs(r) > 0.01 and fmin < f < fmax]
                    return sorted(formants)

                lpc_result = estimate_formants_safe(frame, sample_rate=sr, lpc_order=12)
            except Exception as e:
                lpc_err = str(e)

        # Build a compact multi-line diagnostic message
        lines = []
        lines.append(f"SR={sr}  len={sig.size}  dur={duration:.2f}s  min={sig_min:.3g} max={sig_max:.3g}")
        if spect_err:
            lines.append(f"Spectrogram: ERROR: {spect_err}")
        else:
            lines.append(f"Spectrogram dominant median={dom_median:.1f} Hz  first={dom_first:.1f} Hz")
        if lpc_debug:
            if lpc_err:
                lines.append(f"LPC: ERROR: {lpc_err}")
            else:
                if lpc_result:
                    fdisp = ", ".join(str(int(x)) for x in lpc_result[:3])
                    lines.append(f"LPC formants (top3): {fdisp}")
                else:
                    lines.append("LPC formants: none detected")
        else:
            lines.append("LPC debug: off (click Diagnostics ▶ to enable)")

        # Update the live_text overlay (wrap to two lines if long)
        diag_text = " | ".join(lines)
        # If too long, break into two lines for readability
        if len(diag_text) > 120:
            # split roughly in half at a separator
            half = len(diag_text) // 2
            sep = diag_text.rfind(" | ", 0, half)
            if sep == -1:
                sep = diag_text.find(" | ", half)
            if sep != -1:
                diag_text = diag_text[:sep] + "\n" + diag_text[sep + 3:]
        live_text.set_text(diag_text)
        live_text.set_color("green" if (lpc_result and len(lpc_result) >= 2) else "orange")
        fig.canvas.draw_idle()

        # default: run quick checks without heavy LPC debug; hold Shift-click to enable LPC debug
        def _diag_click(event):
            # If user wants LPC debug, call with lpc_debug=True; here we always run quick checks.
            run_diagnostics(lpc_debug=False)

        btn_diag.on_clicked(_diag_click)

        live_text.set_text(diag_text)
        live_text.set_color("green" if (lpc_result and len(lpc_result) >= 2) else "orange")
        fig.canvas.draw_idle()
        return diag_text  # <-- return for tests

    ax_btn_diag = plt.axes([control_left, 0.27, control_width, control_height])
    btn_diag = Button(ax_btn_diag, 'Diagnostics ▶')
    btn_diag.on_clicked(lambda e: run_diagnostics(lpc_debug=False))

    ax_btn_diag_full = plt.axes([control_left, 0.20, control_width, control_height])
    btn_diag_full = Button(ax_btn_diag_full, 'Diagnostics (LPC)')
    btn_diag_full.on_clicked(lambda e: run_diagnostics(lpc_debug=True))

    if not headless:
        ani = animation.FuncAnimation(fig, poll_queue, interval=100, cache_frame_data=False)

    # Initial spectrum
    update_spectrum(current_vowel_name, current_formants, pitch_slider.val, tol_slider.val)
    if headless:
        # return the key objects tests need
        return fig, mic, live_text, run_diagnostics

    plt.show()
    return None



if __name__ == "__main__":
    interactive_vowel_chart(261.63, 'tenor')


