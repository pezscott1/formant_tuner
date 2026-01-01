import numpy as np
from utils.music_utils import freq_to_note_name


# ============================================================
# Spectrum Plot (hybrid-aware, dict-based)
# ============================================================

def _normalize_formants(x):
    """
    Accept either dict {"f1":..,"f2":..,"f3":..} or
    tuple/list (f1,f2,f3) and normalize to dict.
    """
    if isinstance(x, dict):
        return x
    if isinstance(x, (tuple, list)):
        vals = list(x) + [None, None, None]
        f1, f2, f3 = vals[:3]
        return {"f1": f1, "f2": f2, "f3": f3}
    return {"f1": None, "f2": None, "f3": None}


def update_spectrum(window, vowel, target_formants,
                    measured_formants, pitch, _tolerance):
    """
    Modernized spectrum plot:
      - Uses dict-based formants
      - Uses hybrid metadata when available
      - Draws target + measured formants
      - Shows hybrid/LPC method + confidence
    """

    ax = window.ax_chart
    ax.clear()

    # ---------------- Extract dict-based formants ----------------
    target_formants = _normalize_formants(target_formants)
    measured_formants = _normalize_formants(measured_formants)

    f1_t = target_formants.get("f1")
    f2_t = target_formants.get("f2")
    f3_t = target_formants.get("f3")

    f1_m = measured_formants.get("f1")
    f2_m = measured_formants.get("f2")
    f3_m = measured_formants.get("f3")

    # ====== Pull latest audio + hybrid/LPC metadata ======
    raw = None
    if getattr(window, "analyzer", None) is not None:
        try:
            raw = window.analyzer.get_latest_raw()
        except Exception:
            raw = None

    segment = raw.get("segment") if raw else None

    # Prefer hybrid metadata
    if raw and "hybrid_method" in raw:
        method = raw["hybrid_method"]
        conf = float(raw.get("confidence", 0.0))
    else:
        method = raw.get("method", "none") if raw else "none"
        conf = float(raw.get("confidence", 0.0)) if raw else 0.0

    # ====== Plot FFT ======
    if segment is not None:
        seg = np.asarray(segment, dtype=float).flatten()
        if seg.size > 0:
            fft = np.abs(np.fft.rfft(seg))
            freqs = np.fft.rfftfreq(len(seg), 1.0 / window.sample_rate)
            ax.plot(freqs, fft, color="black", linewidth=1.0)

    # ====== Target formant lines ======
    for f in (f1_t, f2_t, f3_t):
        if f is not None and np.isfinite(f):
            ax.axvline(f, color="blue", linestyle="--", alpha=0.7)

    # ====== Measured formant lines ======
    if conf > 0.25:
        for f in (f1_m, f2_m, f3_m):
            if f is not None and np.isfinite(f):
                ax.axvline(f, color="red", linestyle=":", alpha=0.8)

    # ====== Title ======
    if pitch and pitch > 0:
        note = freq_to_note_name(pitch)
        ax.set_title(
            f"Spectrum /{vowel}/ — {note} ({pitch:.1f} Hz)  "
            f"[{method}, conf={conf:.2f}]"
        )
    else:
        ax.set_title(f"Spectrum /{vowel}/  [{method}, conf={conf:.2f}]")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")

    window.canvas.draw_idle()


# ============================================================
# Vowel Chart (hybrid-aware, dict-based)
# ============================================================

def update_vowel_chart(
    window,
    vowel,
    target_formants,
    measured_formants,
    vowel_score,
    resonance_score,
    overall,
):
    """
    Modernized vowel chart:
      - Uses dict-based formants (but accepts tuples)
      - Uses stability + confidence gating
      - Draws measured point + line to target
      - Title shows scores
    """

    ax = window.ax_vowel

    # ---------------- Remove previous artists ----------------
    had_previous = getattr(window, "vowel_measured_artist", None) is not None

    old_point = getattr(window, "vowel_measured_artist", None)
    old_line = getattr(window, "vowel_line_artist", None)

    if old_point is not None:
        try:
            old_point.remove()
        except Exception:
            pass
    if old_line is not None:
        try:
            old_line.remove()
        except Exception:
            pass

    window.vowel_measured_artist = None
    window.vowel_line_artist = None

    # ---------------- Extract stability + confidence ----------------
    analyzer = getattr(window, "analyzer", None)
    latest_raw = getattr(window, "latest_raw", None)
    if latest_raw is None and analyzer is not None:
        try:
            latest_raw = analyzer.get_latest_raw()
        except Exception:
            latest_raw = None
    if latest_raw is None:
        latest_raw = {}

    conf = float(latest_raw.get("confidence", 1.0))

    formant_smoother = getattr(analyzer, "formant_smoother", None) if analyzer else None
    stable = getattr(formant_smoother, "formants_stable", True) \
        if formant_smoother else True

    # ---------------- Extract dict-based formants ----------------
    target_formants = _normalize_formants(target_formants)
    measured_formants = _normalize_formants(measured_formants)

    tf1 = target_formants.get("f1")
    tf2 = target_formants.get("f2")

    mf1 = measured_formants.get("f1")
    mf2 = measured_formants.get("f2")

    def valid(x):
        return x is not None and np.isfinite(x)

    # Accept F2-only frames for classical /i/ and similar vowels
    if valid(mf2) and not valid(mf1):
        measured_valid = True
    else:
        measured_valid = valid(mf1) and valid(mf2)

    target_valid = valid(tf2)  # allow target F2-only too

    if (not measured_valid) or (not stable) or (conf < 0.25):
        title = (
            f"/{vowel}/  Overall={overall:.2f}  "
            f"(Vowel={vowel_score:.2f}, Resonance={resonance_score:.2f})"
        )
        ax.set_title(title)
        window.canvas.draw_idle()
        return

    # ---------------- Measured point ----------------
    try:
        # If F1 is missing, place the point at a neutral F1 height (e.g., 300 Hz)
        y = mf1 if valid(mf1) else 300.0
        point = ax.scatter([mf2], [y], c="red", s=40)
    except Exception:
        point = None

    if not had_previous:
        window.vowel_measured_artist = point

    # ---------------- Line to target ----------------
    line = None
    if target_valid:
        try:
            y1 = tf1 if valid(tf1) else y
            y2 = mf1 if valid(mf1) else y
            line = ax.plot([tf2, mf2], [y1, y2], ...)
        except Exception:
            line = None

    if not had_previous:
        window.vowel_line_artist = line

    # ---------------- Title ----------------
    title = (
        f"/{vowel}/  Overall={overall:.2f}  "
        f"(Vowel={vowel_score:.2f}, Resonance={resonance_score:.2f})"
    )
    ax.set_title(title)

    # ---------------- Redraw ----------------
    window.canvas.draw_idle()
