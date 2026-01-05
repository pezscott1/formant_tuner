import numpy as np
from LEGACY.utils.music_utils import freq_to_note_name


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
    # Recreate spectrum status text if needed
    if (not hasattr(window, "spec_status_text")
            or window.spec_status_text.axes is not ax):
        window.spec_status_text = ax.text(
            0.02, 0.95, "",
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=12, fontweight="bold", color="#CC0000"
        )

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

    # ====== Static title + per-axis status text ======
    ax.set_title("Spectrum")

    # Update spectrum status text
    if hasattr(window, "spec_status_text"):
        if pitch and pitch > 0:
            note = freq_to_note_name(pitch)
            status = f"/{vowel}/ — {note} ({pitch:.1f} Hz)  [{method}, conf={conf:.2f}]"
        else:
            status = f"/{vowel}/  [{method}, conf={conf:.2f}]"

        window.spec_status_text.set_text(status)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")

    window.canvas.draw_idle()


def update_vowel_chart(
    window,
    vowel,
    target_formants,
    measured_formants,
    vowel_score,
    resonance_score,
    overall,
):
    ax = window.ax_vowel

    # ------------------------------------------------------------
    # Remove previous artists
    # ------------------------------------------------------------
    had_previous = getattr(window, "vowel_measured_artist", None) is not None
    _remove_old_artists(window)

    # ------------------------------------------------------------
    # Extract stability + confidence
    # ------------------------------------------------------------
    conf, stable = _extract_confidence_and_stability(window)

    # ------------------------------------------------------------
    # Normalize formants
    # ------------------------------------------------------------
    target_formants = _normalize_formants(target_formants)
    measured_formants = _normalize_formants(measured_formants)

    tf1, tf2 = target_formants.get("f1"), target_formants.get("f2")
    mf1, mf2 = measured_formants.get("f1"), measured_formants.get("f2")

    measured_valid, target_valid = _validate_formants(mf1, mf2, tf2)

    # ------------------------------------------------------------
    # Gating: invalid → title only
    # ------------------------------------------------------------
    if (not measured_valid) or (not stable) or (conf < 0.25):
        _set_title(window, ax, vowel, vowel_score, resonance_score, overall)
        window.canvas.draw_idle()
        return

    # ------------------------------------------------------------
    # Draw measured point
    # ------------------------------------------------------------
    point = _draw_measured_point(ax, mf1, mf2)

    if not had_previous:
        window.vowel_measured_artist = point

    # ------------------------------------------------------------
    # Draw line to target
    # ------------------------------------------------------------
    line = _draw_target_line(ax, tf1, tf2, mf1, mf2) if target_valid else None

    if not had_previous:
        window.vowel_line_artist = line

    # ------------------------------------------------------------
    # Title + redraw
    # ------------------------------------------------------------
    _set_title(window, ax, vowel, vowel_score, resonance_score, overall)
    window.canvas.draw_idle()


# ======================================================================
# Helper functions (pure, complexity-free)
# ======================================================================

def _remove_old_artists(window):
    for attr in ("vowel_measured_artist", "vowel_line_artist"):
        artist = getattr(window, attr, None)
        if artist is not None:
            try:
                artist.remove()
            except Exception:
                pass
        setattr(window, attr, None)


def _extract_confidence_and_stability(window):
    analyzer = getattr(window, "analyzer", None)
    latest_raw = getattr(window, "latest_raw", None)

    if latest_raw is None and analyzer is not None:
        try:
            latest_raw = analyzer.get_latest_raw()
        except Exception:
            latest_raw = None

    latest_raw = latest_raw or {}
    conf = float(latest_raw.get("confidence", 1.0))

    smoother = getattr(analyzer, "formant_smoother", None) if analyzer else None
    stable = getattr(smoother, "formants_stable", True) if smoother else True

    return conf, stable


def _valid(x):
    return x is not None and np.isfinite(x)


def _validate_formants(mf1, mf2, tf2):

    # Accept F2-only frames
    measured_valid = _valid(mf2) if not _valid(mf1) else (_valid(mf1) and _valid(mf2))
    target_valid = _valid(tf2)

    return measured_valid, target_valid


def _draw_measured_point(ax, mf1, mf2):
    try:
        y = mf1 if _valid(mf1) else 300.0
        return ax.scatter([mf2], [y], c="red", s=40)
    except Exception:
        return None


def _draw_target_line(ax, tf1, tf2, mf1, mf2):
    try:
        y1 = tf1 if _valid(tf1) else (mf1 if _valid(mf1) else 300.0)
        y2 = mf1 if _valid(mf1) else 300.0
        return ax.plot([tf2, mf2], [y1, y2], ...)
    except Exception:
        return None


def _set_title(window, ax, vowel, vowel_score, resonance_score, overall):
    v = vowel or "None"

    overall = 0.0 if overall is None else float(overall)
    vowel_score = 0.0 if vowel_score is None else float(vowel_score)
    resonance_score = 0.0 if resonance_score is None else float(resonance_score)

    if hasattr(window, "vowel_status_text"):
        window.vowel_status_text.set_text(
            f"/{v}/  Overall={overall:.2f}  "
            f"(Vowel={vowel_score:.2f}, Resonance={resonance_score:.2f})"
        )
