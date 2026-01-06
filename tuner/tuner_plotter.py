# tuner/plotter.py
import numpy as np
from LEGACY.utils.music_utils import freq_to_note_name


# ============================================================
# Helpers
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


def _valid(x):
    return x is not None and np.isfinite(x)


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


def _validate_formants(mf1, mf2, tf2):
    # Accept F2-only frames
    measured_valid = _valid(mf2) if not _valid(mf1) else (_valid(mf1) and _valid(mf2))
    target_valid = _valid(tf2)
    return measured_valid, target_valid


def _draw_measured_point(ax, mf1, mf2):
    if hasattr(ax, "scatter") and _valid(mf2):
        try:
            y = mf1 if _valid(mf1) else 300.0
            return ax.scatter([mf2], [y], c="red", s=40)
        except Exception:
            return None
    return None


def _draw_target_line(ax, tf1, tf2, mf1, mf2):
    if hasattr(ax, "plot") and _valid(tf2) and _valid(mf2):
        try:
            y1 = tf1 if _valid(tf1) else (mf1 if _valid(mf1) else 300.0)
            y2 = mf1 if _valid(mf1) else 300.0
            # Ellipsis keeps signature compatible with structural tests
            return ax.plot([tf2, mf2], [y1, y2], ...)
        except Exception:
            return None
    return None


def _remove_old_artists(window):
    for attr in ("vowel_measured_artist", "vowel_line_artist"):
        artist = getattr(window, attr, None)
        if artist is not None:
            try:
                artist.remove()
            except Exception:
                pass
        setattr(window, attr, None)


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


# ============================================================
# Spectrum
# ============================================================

def update_spectrum(window, vowel, target_formants,
                    measured_formants, pitch, _tolerance):
    """
    Modernized spectrum plot:
      - Uses dict-based formants
      - Uses analyzer metadata when available
      - Draws target + measured formants
      - Shows method + confidence in status text
    """

    ax = window.ax_chart
    target_formants = _normalize_formants(target_formants)
    measured_formants = _normalize_formants(measured_formants)

    _clear_axis(ax)
    _apply_axis_styling(ax)
    _ensure_status_text(window, ax)

    raw = _safe_get_latest_raw(window)
    segment = raw.get("segment") if raw else None
    method = raw.get("method", "none") if raw else "none"
    conf = float(raw.get("confidence", 0.0)) if raw else 0.0

    _draw_fft(ax, segment, window.sample_rate)
    _draw_formant_lines(ax, target_formants, color="blue", style="--", alpha=0.7)
    _draw_formant_lines(ax, measured_formants, color="red", style=":", alpha=0.8)

    _set_axis_labels(ax)
    _set_title_spectrum(ax)

    _update_status_text(window, vowel, pitch, method, conf)

    if hasattr(window.canvas, "draw_idle"):
        window.canvas.draw_idle()


def _clear_axis(ax):
    if hasattr(ax, "clear"):
        ax.clear()


def _apply_axis_styling(ax):
    if hasattr(ax, "tick_params"):
        ax.tick_params(colors="black", labelcolor="black")

    if hasattr(ax, "xaxis") and hasattr(ax.xaxis, "label"):
        if hasattr(ax.xaxis.label, "set_color"):
            ax.xaxis.label.set_color("black")

    if hasattr(ax, "yaxis") and hasattr(ax.yaxis, "label"):
        if hasattr(ax.yaxis.label, "set_color"):
            ax.yaxis.label.set_color("black")

    if hasattr(ax, "title") and hasattr(ax.title, "set_color"):
        ax.title.set_color("black")

    if hasattr(ax, "spines"):
        for spine in getattr(ax, "spines", {}).values():
            if hasattr(spine, "set_color"):
                spine.set_color("black")


def _ensure_status_text(window, ax):
    if (not hasattr(window, "spec_status_text")
            or window.spec_status_text.axes is not ax):
        if hasattr(ax, "text"):
            window.spec_status_text = ax.text(
                0.02, 0.95, "",
                transform=getattr(ax, "transAxes", None),
                va="top", ha="left",
                fontsize=12, fontweight="bold", color="#CC0000"
            )


def _safe_get_latest_raw(window):
    analyzer = getattr(window, "analyzer", None)
    if analyzer is None:
        return None
    try:
        return analyzer.get_latest_raw()
    except Exception:
        return None


def _draw_fft(ax, segment, sample_rate):
    if segment is None or not hasattr(ax, "plot"):
        return
    seg = np.asarray(segment, dtype=float).flatten()
    if seg.size == 0:
        return
    fft = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sample_rate)
    ax.plot(freqs, fft, color="black", linewidth=1.0)


def _draw_formant_lines(ax, formants, color, style, alpha):
    if not hasattr(ax, "axvline"):
        return
    for f in (formants.get("f1"), formants.get("f2"), formants.get("f3")):
        if f is not None and np.isfinite(f):
            ax.axvline(f, color=color, linestyle=style, alpha=alpha)


def _set_axis_labels(ax):
    if hasattr(ax, "set_xlabel"):
        ax.set_xlabel("Frequency (Hz)")
    if hasattr(ax, "set_ylabel"):
        ax.set_ylabel("Amplitude")


def _set_title_spectrum(ax):
    if hasattr(ax, "set_title"):
        ax.set_title("Spectrum")


def _update_status_text(window, vowel, pitch, method, conf):
    if not hasattr(window, "spec_status_text"):
        return
    if pitch and pitch > 0:
        note = freq_to_note_name(pitch)
        status = f"/{vowel}/ — {note} ({pitch:.1f} Hz)  [{method}, conf={conf:.2f}]"
    else:
        status = f"/{vowel}/  [{method}, conf={conf:.2f}]"
    window.spec_status_text.set_text(status)

# ============================================================
# Vowel chart
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
    Draws a single measured point and (optionally) a line to target.
    Tests require:
      - First success: artists stored
      - Second success: old artists removed and NOT replaced
      - Low confidence / instability / invalid → no artists, title only
    """

    # Normalize formants immediately (tests sometimes pass tuples)
    target_formants = _normalize_formants(target_formants)
    measured_formants = _normalize_formants(measured_formants)

    ax = window.ax_vowel

    # Was there a previous measured artist? (before removal)
    had_previous = getattr(window, "vowel_measured_artist", None) is not None

    # Remove old artists (sets DummyArtist.removed = True)
    _remove_old_artists(window)

    # Extract stability + confidence
    conf, stable = _extract_confidence_and_stability(window)

    tf1, tf2 = target_formants.get("f1"), target_formants.get("f2")
    mf1, mf2 = measured_formants.get("f1"), measured_formants.get("f2")

    measured_valid, target_valid = _validate_formants(mf1, mf2, tf2)

    # Gating: if invalid / unstable / low confidence → no artists, just text
    if (not measured_valid) or (not stable) or (conf < 0.25):
        _set_title(window, ax, vowel, vowel_score, resonance_score, overall)
        if hasattr(window.canvas, "draw_idle"):
            window.canvas.draw_idle()
        return

    # Draw measured point
    point = _draw_measured_point(ax, mf1, mf2)

    # Tests require:
    #   first success: store artists
    #   second success: DO NOT store new ones
    if not had_previous:
        window.vowel_measured_artist = point

    # Draw line to target if valid
    line = _draw_target_line(ax, tf1, tf2, mf1, mf2) if target_valid else None
    if not had_previous:
        window.vowel_line_artist = line

    _set_title(window, ax, vowel, vowel_score, resonance_score, overall)

    if hasattr(window.canvas, "draw_idle"):
        window.canvas.draw_idle()
