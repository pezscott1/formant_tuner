# tuner/tuner_plotter.py

import numpy as np
from utils.music_utils import freq_to_note_name


# ============================================================
# Spectrum Plot
# ============================================================

def update_spectrum(window, vowel, target_formants, measured_formants, pitch, _tol):
    ax = window.ax_chart
    ax.clear()

    f1_t, f2_t, f3_t = target_formants
    f1_m, f2_m, f3_m = measured_formants

    # ====== NEW: get latest audio segment ======
    raw = None
    if getattr(window, "analyzer", None) is not None:
        try:
            raw = window.analyzer.get_latest_raw()
        except Exception:
            raw = None

    segment = raw.get("segment") if raw else None
    if segment is not None:
        seg = np.asarray(segment, dtype=float).flatten()
        if seg.size > 0:
            # Compute FFT
            fft = np.abs(np.fft.rfft(seg))
            freqs = np.fft.rfftfreq(len(seg), 1.0 / window.sample_rate)

            ax.plot(freqs, fft, color="black", linewidth=1.0)

    # ====== Existing formant lines ======
    if f1_t is not None and not np.isnan(f1_t):
        ax.axvline(f1_t, color="blue", linestyle="--", alpha=0.7)
    if f2_t:
        ax.axvline(f2_t, color="blue", linestyle="--", alpha=0.7)
    if f3_t:
        ax.axvline(f3_t, color="blue", linestyle="--", alpha=0.7)

    if f1_m:
        ax.axvline(f1_m, color="red", linestyle=":", alpha=0.8)
    if f2_m:
        ax.axvline(f2_m, color="red", linestyle=":", alpha=0.8)
    if f3_m:
        ax.axvline(f3_m, color="red", linestyle=":", alpha=0.8)

    # Title
    if pitch and pitch > 0:
        note = freq_to_note_name(pitch)
        ax.set_title(f"Spectrum /{vowel}/ — {note} ({pitch:.1f} Hz)")
    else:
        ax.set_title(f"Spectrum /{vowel}/")

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
    Update the vowel chart with:
      - measured point
      - straight line measured → target
      - updated title with scores
    """
    f1_t, f2_t, _ = target_formants
    f1_m, f2_m, _ = measured_formants

    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    # Sanitize all formants
    f1_t = _safe_float(f1_t)
    f2_t = _safe_float(f2_t)
    f1_m = _safe_float(f1_m)
    f2_m = _safe_float(f2_m)

    # ---------------- Clear old measured point ----------------
    if getattr(window, "vowel_measured_artist", None) is not None:
        try:
            window.vowel_measured_artist.remove()
        except Exception:
            pass
        window.vowel_measured_artist = None

    # ---------------- Clear old measured→target line ----------------
    if getattr(window, "vowel_line_artist", None) is not None:
        try:
            window.vowel_line_artist.remove()
        except Exception:
            pass
        window.vowel_line_artist = None

    # ---------------- Plot measured point ----------------
    if not np.isnan(f1_m) and not np.isnan(f2_m):
        try:
            window.vowel_measured_artist = window.ax_vowel.scatter(
                [f1_m], [f2_m], color="red", s=60, zorder=5
            )
        except Exception:
            window.vowel_measured_artist = None

    # ---------------- Straight line measured → target ----------------
    if (
        not np.isnan(f1_m) and not np.isnan(f2_m)
        and not np.isnan(f1_t) and not np.isnan(f2_t)
    ):
        try:
            window.vowel_line_artist = window.ax_vowel.plot(
                [f1_m, f1_t],
                [f2_m, f2_t],
                color="gray",
                linestyle="-",
                linewidth=1.2,
                alpha=0.7,
                zorder=4,
            )[0]
        except Exception:
            window.vowel_line_artist = None

    # ---------------- Update title with scores ----------------
    title = (f"/{vowel}/  Overall={overall} "
             f"(Vowel={vowel_score}, Resonance={resonance_score})")
    window.ax_vowel.set_title(title)

    # ---------------- Redraw ----------------
    try:
        window.canvas.draw_idle()
    except Exception:
        pass
