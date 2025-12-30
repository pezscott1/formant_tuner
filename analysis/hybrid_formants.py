from dataclasses import dataclass
from typing import Optional, Dict, Any
from analysis.lpc import estimate_formants
from analysis.true_envelope import estimate_formants_te


@dataclass
class BaseFormantLike:
    f1: Optional[float]
    f2: Optional[float]
    f3: Optional[float]
    confidence: float
    method: str


@dataclass
class HybridFormantResult:
    f1: Optional[float]
    f2: Optional[float]
    f3: Optional[float]
    confidence: float
    method: str          # "lpc" or "te"
    primary: str         # "lpc" or "te"
    lpc: BaseFormantLike
    te: BaseFormantLike
    debug: Dict[str, Any]


# ------------------------- sanity helpers -------------------------
def _valid_scalar(x: Optional[float]) -> bool:
    return x is not None and x == x  # not None and not NaN


def _plausible_pair(f1: Optional[float], f2: Optional[float]) -> bool:
    """Very basic human-range plausibility for F1/F2."""
    if not _valid_scalar(f1) or not _valid_scalar(f2):
        return False

    # Generic human ranges; these are deliberately loose
    if not (150 <= f1 <= 1200):
        return False
    if not (400 <= f2 <= 3500):
        return False
    if f2 <= f1:
        return False

    return True


def _is_back_vowel(v: Optional[str]) -> bool:
    return v in {"u", "ɔ", "o", "ʊ", "ɒ"}


def _is_front_vowel(v: Optional[str]) -> bool:
    return v in {"i", "e", "ɛ", "ɪ", "y"}

# ------------------------- decision logic -------------------------


def choose_formants_hybrid(
    lpc_res,
    te_res,
    vowel_hint: Optional[str] = None,
) -> HybridFormantResult:
    """
    Decide between LPC and TE given both results and an optional vowel hint.

    lpc_res: FormantResult from analysis.lpc.estimate_formants
    te_res:  TEFormantResult from analysis.true_envelope.estimate_formants_te
    """

    # Normalize to a common representation
    lpc = BaseFormantLike(
        f1=lpc_res.f1,
        f2=lpc_res.f2,
        f3=lpc_res.f3,
        confidence=float(getattr(lpc_res, "confidence", 0.0) or 0.0),
        method=str(getattr(lpc_res, "method", "lpc")),
    )
    te = BaseFormantLike(
        f1=te_res.f1,
        f2=te_res.f2,
        f3=te_res.f3,
        confidence=1.0,  # you can refine this later with a TE-specific score
        method="te",
    )

    back = _is_back_vowel(vowel_hint)
    front = _is_front_vowel(vowel_hint)

    dbg: Dict[str, Any] = {
        "vowel_hint": vowel_hint,
        "back_vowel": back,
        "front_vowel": front,
        "te_vetoes": [],
        "lpc_vetoes": [],
    }

    # -------------------- 1. Base plausibility --------------------

    lpc_ok = _plausible_pair(lpc.f1, lpc.f2)
    te_ok = _plausible_pair(te.f1, te.f2)

    dbg["lpc_base_ok"] = lpc_ok
    dbg["te_base_ok"] = te_ok

    # -------------------- 2. TE vetoes (strict) --------------------

    # These are derived from your actual behavior: TE often collapses F2 downward.

    if _valid_scalar(te.f1) and te.f1 < 200:
        te_ok = False
        dbg["te_vetoes"].append("f1_too_low")

    # TE F2 too low for any real vowel (except back vowels)
    if _valid_scalar(te.f2) and te.f2 < 800 and not back:
        te_ok = False
        dbg["te_vetoes"].append("f2_too_low")

    if _valid_scalar(te.f1) and _valid_scalar(te.f2) and te.f2 <= te.f1:
        te_ok = False
        dbg["te_vetoes"].append("f2_leq_f1")

    # For clearly front vowels, TE F2 < ~1200 is almost always harmonic confusion.
    if front and _valid_scalar(te.f2) and te.f2 < 1200:
        te_ok = False
        dbg["te_vetoes"].append("front_low_f2")

    # -------------------- 3. LPC vetoes (mild) --------------------

    # For back vowels (/u/, /ɔ/), LPC F2 >> 1800 is typically F3 masquerading as F2.
    if back and _valid_scalar(lpc.f2) and lpc.f2 > 1800:
        lpc_ok = False
        dbg["lpc_vetoes"].append("back_high_f2")

    # If LPC returned None for either, it's not usable.
    if not _valid_scalar(lpc.f1) or not _valid_scalar(lpc.f2):
        lpc_ok = False
        dbg["lpc_vetoes"].append("missing_f1_or_f2")

    # -------------------- 4. Primary method by vowel class --------------------

    if back:
        primary: str = "te"
    elif front:
        primary = "lpc"
    else:
        primary = "lpc"

    dbg["primary"] = primary

    # -------------------- 5. Main selection --------------------

    chosen: str = "unknown"
    f1 = f2 = f3 = None
    confidence = 0.0

    def use_lpc():
        nonlocal chosen, f1, f2, f3, confidence
        chosen = "lpc"
        f1, f2, f3 = lpc.f1, lpc.f2, lpc.f3
        # If LPC didn't provide a confidence, assume reasonably strong
        confidence = lpc.confidence or 0.8

    def use_te():
        nonlocal chosen, f1, f2, f3, confidence
        chosen = "te"
        f1, f2, f3 = te.f1, te.f2, te.f3
        # TE baseline; you can later modulate this with a real score
        confidence = 0.7

    if lpc_ok and te_ok:
        # Both plausible: choose by primary method
        if primary == "te":
            use_te()
        else:
            use_lpc()
        dbg["selection_case"] = "both_ok"
    elif lpc_ok and not te_ok:
        use_lpc()
        dbg["selection_case"] = "only_lpc_ok"
    elif te_ok and not lpc_ok:
        use_te()
        dbg["selection_case"] = "only_te_ok"
    else:
        # Both implausible: fall back to LPC, but very low confidence
        use_lpc()
        confidence = min(confidence, 0.2)
        dbg["selection_case"] = "both_bad"

    # Down-weight if we had to go against primary preference
    if chosen != primary:
        confidence *= 0.9
        dbg["primary_mismatch"] = True
    else:
        dbg["primary_mismatch"] = False

    return HybridFormantResult(
        f1=f1,
        f2=f2,
        f3=f3,
        confidence=confidence,
        method=chosen,
        primary=primary,
        lpc=lpc,
        te=te,
        debug=dbg,
    )

# ------------------------- top-level API -------------------------


def estimate_formants_hybrid(
    signal,
    sr: int,
    vowel_hint: Optional[str] = None,
    debug: bool = False,
) -> HybridFormantResult:
    """
    Run LPC + TE on the same frame and choose a hybrid estimate.

    This is designed to be a drop-in replacement for the existing
    estimate_formants(signal, sr, debug=...) call in the engine.
    """

    lpc_res = estimate_formants(signal, sr, debug=debug)
    te_res = estimate_formants_te(signal, sr)

    return choose_formants_hybrid(
        lpc_res=lpc_res,
        te_res=te_res,
        vowel_hint=vowel_hint,
    )
