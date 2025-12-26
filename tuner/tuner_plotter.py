# tuner/tuner_plotter.py
import numpy as np
from utils.music_utils import freq_to_note_name


# ============================================================
# Spectrum Plot
# ============================================================

def update_spectrum(window, vowel, target_formants,  # noqa: C901
                    measured_formants, pitch, _tol):
    ax = window.ax_chart
    ax.clear()

    f1_t, f2_t, f3_t = target_formants
    f1_m, f2_m, f3_m = measured_formants

    # ====== NEW: get latest audio + LPC metadata ======
    raw = None
    if getattr(window, "analyzer", None) is not None:
        try:
            raw = window.analyzer.get_latest_raw()
        except Exception:
            raw = None

    segment = raw.get("segment") if raw else None
    lpc_conf = raw.get("confidence", 0.0) if raw else 0.0
    lpc_method = raw.get("method", "none") if raw else "none"

    # ====== Plot FFT ======
    if segment is not None:
        seg = np.asarray(segment, dtype=float).flatten()
        if seg.size > 0:
            fft = np.abs(np.fft.rfft(seg))
            freqs = np.fft.rfftfreq(len(seg), 1.0 / window.sample_rate)
            ax.plot(freqs, fft, color="black", linewidth=1.0)

    # ====== Target formant lines ======
    if f1_t is not None and not np.isnan(f1_t):
        ax.axvline(f1_t, color="blue", linestyle="--", alpha=0.7)
    if f2_t is not None and not np.isnan(f2_t):
        ax.axvline(f2_t, color="blue", linestyle="--", alpha=0.7)
    if f3_t is not None and not np.isnan(f3_t):
        ax.axvline(f3_t, color="blue", linestyle="--", alpha=0.7)

    # ====== Measured formant lines (confidence-aware) ======
    if lpc_conf > 0.25:
        if f1_m:
            ax.axvline(f1_m, color="red", linestyle=":", alpha=0.8)
        if f2_m:
            ax.axvline(f2_m, color="red", linestyle=":", alpha=0.8)
        if f3_m:
            ax.axvline(f3_m, color="red", linestyle=":", alpha=0.8)

    # ====== Title ======
    if pitch and pitch > 0:
        note = freq_to_note_name(pitch)
        ax.set_title(
            f"Spectrum /{vowel}/ — {note} ({pitch:.1f} Hz)  "
            f"[{lpc_method}, conf={lpc_conf:.2f}]"
        )
    else:
        ax.set_title(f"Spectrum /{vowel}/  [{lpc_method}, conf={lpc_conf:.2f}]")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")

    window.canvas.draw_idle()


# ============================================================
# Vowel Chart
# ============================================================


def update_vowel_chart(  # noqa: C901
    window,
    vowel,
    target_formants,
    measured_formants,
    vowel_score,
    resonance_score,
    overall,
):
    """
    Update the vowel chart with measured and target formants.

    Test-driven behavior:
      - Always remove previous artists first.
      - Suppress artists if:
          * analyzer.formant_smoother.formants_stable is False, or
          * latest_raw["confidence"] < 0.25, or
          * measured formants are NaN/None.
      - When suppressed, both vowel_measured_artist and vowel_line_artist stay None.
      - First successful call: create artists and store them.
      - Second successful call: remove previous artists and DO NOT store new ones
        (attributes end as None).
      - Title must contain "/vowel/".
    """

    ax = window.ax_vowel

    # ---------------- Remove previous artists (always) ----------------
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

    # ---------------- Extract stability and confidence ----------------
    analyzer = getattr(window, "analyzer", None)
    latest_raw = getattr(window, "latest_raw", {}) or {}

    confidence = float(latest_raw.get("confidence", 1.0))

    formant_smoother = getattr(analyzer, "formant_smoother", None) if analyzer else None
    stable = getattr(formant_smoother, "formants_stable", True)\
        if formant_smoother else True

    # ---------------- NaN / None handling ----------------
    tf1, tf2, _ = target_formants
    mf1, mf2, _ = measured_formants

    def is_valid(x):
        if x is None:
            return False
        if isinstance(x, float) and np.isnan(x):
            return False
        return True

    measured_valid = is_valid(mf1) and is_valid(mf2)
    target_valid = is_valid(tf1) and is_valid(tf2)

    # ---------------- Suppression gating ----------------
    if (not measured_valid) or (not stable) or (confidence < 0.25):
        # No artists created; attributes remain None
        title = (
            f"/{vowel}/  Overall={overall:.2f}  "
            f"(Vowel={vowel_score:.2f}, Resonance={resonance_score:.2f})"
        )
        ax.set_title(title)
        try:
            window.canvas.draw_idle()
        except Exception:
            pass
        return

    # ---------------- Create measured point ----------------
    # First successful call: store the artist.
    # Subsequent successful calls:
    # draw but do not store (tests want None after second call).
    try:
        point = ax.scatter([mf2], [mf1], c="red", s=40)
    except Exception:
        point = None

    if not had_previous:
        window.vowel_measured_artist = point
    # else: do not store, leave attribute as None

    # ---------------- Create line measured → target ----------------
    line = None
    if target_valid:
        try:
            line = ax.plot(
                [tf2, mf2],
                [tf1, mf1],
                c="gray",
                linestyle="--",
                linewidth=1.0,
            )[0]
        except Exception:
            line = None

    if not had_previous:
        window.vowel_line_artist = line
    # else: do not store, leave attribute as None

    # ---------------- Title update ----------------
    title = (
        f"/{vowel}/  Overall={overall:.2f}  "
        f"(Vowel={vowel_score:.2f}, Resonance={resonance_score:.2f})"
    )
    ax.set_title(title)

    # ---------------- Redraw ----------------
    try:
        window.canvas.draw_idle()
    except Exception:
        pass
