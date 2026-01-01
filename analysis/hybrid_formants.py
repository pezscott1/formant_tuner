from dataclasses import dataclass
from typing import Optional, Dict, Any
from analysis.true_envelope import estimate_formants_te
from analysis.lpc import estimate_formants as lpc_formants


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
def estimate_formants(*args, **kwargs):
    return lpc_formants(*args, **kwargs)


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
        confidence=1.0,
        method="te",
    )

    back = _is_back_vowel(vowel_hint)
    front = _is_front_vowel(vowel_hint)

    dbg = _init_debug(vowel_hint, back, front)
    _print_input_debug(lpc, te, vowel_hint)

    # 1. Base plausibility
    lpc_ok = _plausible_pair(lpc.f1, lpc.f2)
    te_ok = _plausible_pair(te.f1, te.f2)
    dbg["lpc_base_ok"] = lpc_ok
    dbg["te_base_ok"] = te_ok

    # 2. TE vetoes
    te_ok = _apply_te_vetoes(te, te_ok, dbg, back, front)

    # 3. LPC vetoes
    lpc_ok = _apply_lpc_vetoes(lpc, lpc_ok, dbg, back)

    # 4. Primary method
    primary = _choose_primary(front, back)
    dbg["primary"] = primary

    # 5. Main selection
    chosen, f1, f2, f3, confidence = _select_formants(
        lpc, te, lpc_ok, te_ok, primary, front, vowel_hint, dbg
    )

    # 6. Primary mismatch penalty
    if primary != "hybrid_front" and chosen != primary:
        confidence *= 0.9
        dbg["primary_mismatch"] = True
    else:
        dbg["primary_mismatch"] = False

    _print_output_debug(chosen, primary, f1, f2, f3, confidence, dbg)

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


def _init_debug(vowel_hint, back, front):
    return {
        "vowel_hint": vowel_hint,
        "back_vowel": back,
        "front_vowel": front,
        "te_vetoes": [],
        "lpc_vetoes": [],
    }


def _print_input_debug(lpc, te, vowel_hint):
    print("\n--- HYBRID INPUT ---")
    print(f"vowel_hint={vowel_hint}")
    print(f"LPC: f1={lpc.f1}, f2={lpc.f2}, f3={lpc.f3}, conf={lpc.confidence}")
    print(f"TE:  f1={te.f1},  f2={te.f2},  f3={te.f3},  conf={te.confidence}")


def _print_output_debug(chosen, primary, f1, f2, f3, confidence, dbg):
    print("--- HYBRID OUTPUT ---")
    print(f"chosen={chosen}, primary={primary}")
    print(f"final f1={f1}, f2={f2}, f3={f3}, conf={confidence}")
    print(f"selection_case={dbg.get('selection_case')}")
    print(f"debug={dbg}")
    print("----------------------\n")


def _apply_te_vetoes(te, te_ok, dbg, back, front):
    if _valid_scalar(te.f1) and te.f1 < 200:
        te_ok = False
        dbg["te_vetoes"].append("f1_too_low")
    if _valid_scalar(te.f1) and te.f1 > 800:
        te_ok = False
        dbg["te_vetoes"].append("f1_too_high")

    if _valid_scalar(te.f2) and te.f2 < 800 and not back:
        te_ok = False
        dbg["te_vetoes"].append("f2_too_low")

    if _valid_scalar(te.f1) and _valid_scalar(te.f2) and te.f2 <= te.f1:
        te_ok = False
        dbg["te_vetoes"].append("f2_leq_f1")

    if front and _valid_scalar(te.f2) and te.f2 < 1200:
        te_ok = False
        dbg["te_vetoes"].append("front_low_f2")

    return te_ok


def _apply_lpc_vetoes(lpc, lpc_ok, dbg, back):
    if back and _valid_scalar(lpc.f2) and lpc.f2 > 1800:
        lpc_ok = False
        dbg["lpc_vetoes"].append("back_high_f2")

    if not _valid_scalar(lpc.f1) or not _valid_scalar(lpc.f2):
        lpc_ok = False
        dbg["lpc_vetoes"].append("missing_f1_or_f2")

    return lpc_ok


def _choose_primary(front, back):
    if back:
        return "te"
    if front:
        return "hybrid_front"
    return "lpc"


def _select_formants(lpc, te, lpc_ok, te_ok, primary, front, vowel_hint, dbg):
    chosen = "unknown"
    f1 = f2 = f3 = None
    confidence = 0.0

    def use_lpc():
        return "lpc", lpc.f1, lpc.f2, lpc.f3, (lpc.confidence or 0.8)

    def use_te():
        return "te", te.f1, te.f2, te.f3, 0.7

    if front:
        return _select_front_hybrid(lpc, te, lpc_ok, te_ok, vowel_hint, dbg)

    if lpc_ok and te_ok:
        return use_te() if primary == "te" else use_lpc(), None
    if lpc_ok:
        return use_lpc()
    if te_ok:
        return use_te()

    chosen, f1, f2, f3, confidence = use_lpc()
    confidence = min(confidence, 0.2)
    dbg["selection_case"] = "both_bad"
    return chosen, f1, f2, f3, confidence


def _select_front_hybrid(lpc, te, lpc_ok, te_ok, vowel_hint, dbg):
    dbg["selection_case"] = "front_hybrid"
    chosen = "hybrid_front"

    # F1
    if te_ok and _valid_scalar(te.f1):
        f1 = te.f1
    elif lpc_ok and _valid_scalar(lpc.f1):
        f1 = lpc.f1
        dbg["te_vetoes"].append("te_f1_missing_or_bad_used_lpc")
    else:
        f1 = 350.0 if vowel_hint == "ɛ" else None
        dbg["te_vetoes"].append("no_valid_f1")

    # F2
    if lpc_ok and _valid_scalar(lpc.f2):
        f2 = lpc.f2
    elif te_ok and _valid_scalar(te.f2):
        f2 = te.f2
        dbg["lpc_vetoes"].append("lpc_f2_missing_or_bad_used_te")
    else:
        f2 = None
        dbg["lpc_vetoes"].append("no_valid_f2")

    # F3
    f3 = None
    if f2 is not None:
        if lpc_ok and f2 == lpc.f2:
            f3 = lpc.f3
        elif te_ok and f2 == te.f2:
            f3 = te.f3

    # Confidence
    confidence = 0.0
    if _valid_scalar(f1):
        confidence += 0.4
    if _valid_scalar(f2):
        confidence += 0.4
    confidence += 0.2 * max(lpc.confidence, te.confidence)
    confidence = min(1.0, confidence)

    return chosen, f1, f2, f3, confidence


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
