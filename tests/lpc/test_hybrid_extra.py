from analysis.hybrid_formants import (
    apply_lpc_vetoes,
    apply_te_vetoes,
    choose_primary,
    select_formants
)
from types import SimpleNamespace as SimpleNS


def test_te_veto_f1_too_low():
    te = SimpleNS(f1=150, f2=1200)
    dbg = {"te_vetoes": []}
    out = apply_te_vetoes(te, True, dbg, back=False, front=False)
    assert not out
    assert "f1_too_low" in dbg["te_vetoes"]


def test_te_veto_f1_too_high():
    te = SimpleNS(f1=900, f2=1200)
    dbg = {"te_vetoes": []}
    out = apply_te_vetoes(te, True, dbg, back=False, front=False)
    assert not out
    assert "f1_too_high" in dbg["te_vetoes"]


def test_te_veto_f2_too_low_non_back():
    te = SimpleNS(f1=300, f2=600)
    dbg = {"te_vetoes": []}
    out = apply_te_vetoes(te, True, dbg, back=False, front=False)
    assert not out
    assert "f2_too_low" in dbg["te_vetoes"]


def test_te_veto_f2_leq_f1():
    te = SimpleNS(f1=500, f2=400)
    dbg = {"te_vetoes": []}
    out = apply_te_vetoes(te, True, dbg, back=False, front=False)
    assert not out
    assert "f2_leq_f1" in dbg["te_vetoes"]


def test_te_veto_front_low_f2():
    te = SimpleNS(f1=300, f2=1000)
    dbg = {"te_vetoes": []}
    out = apply_te_vetoes(te, True, dbg, back=False, front=True)
    assert not out
    assert "front_low_f2" in dbg["te_vetoes"]


def test_lpc_veto_back_high_f2():
    lpc = SimpleNS(f1=300, f2=2000)
    dbg = {"lpc_vetoes": []}
    out = apply_lpc_vetoes(lpc, True, dbg, back=True)
    assert not out
    assert "back_high_f2" in dbg["lpc_vetoes"]


def test_lpc_veto_missing_f1_or_f2():
    lpc = SimpleNS(f1=None, f2=1200)
    dbg = {"lpc_vetoes": []}
    out = apply_lpc_vetoes(lpc, True, dbg, back=False)
    assert not out
    assert "missing_f1_or_f2" in dbg["lpc_vetoes"]


def test_primary_back():
    assert choose_primary(front=False, back=True) == "te"


def test_primary_front():
    assert choose_primary(front=True, back=False) == "hybrid_front"


def test_primary_neutral():
    assert choose_primary(front=False, back=False) == "lpc"


def test_select_both_ok_primary_te():
    dbg = {}
    chosen, f1, f2, f3, conf = select_formants(
        lpc=SimpleNS(f1=300, f2=1500, f3=None, confidence=0.8),
        te=SimpleNS(f1=290, f2=1400, f3=None, confidence=1.0),
        lpc_ok=True, te_ok=True,
        primary="te", front=False, vowel_hint=None, dbg=dbg
    )
    assert chosen == "te"
    assert dbg["selection_case"] == "both_ok"


def test_select_only_lpc_ok():
    dbg = {}
    chosen, *_ = select_formants(
        lpc=SimpleNS(f1=300, f2=1500, f3=None, confidence=0.8),
        te=SimpleNS(f1=None, f2=None, f3=None, confidence=1.0),
        lpc_ok=True, te_ok=False,
        primary="lpc", front=False, vowel_hint=None, dbg=dbg
    )
    assert chosen == "lpc"
    assert dbg["selection_case"] == "only_lpc_ok"


def test_select_only_te_ok():
    dbg = {}
    chosen, *_ = select_formants(
        lpc=SimpleNS(f1=None, f2=None, f3=None, confidence=0.8),
        te=SimpleNS(f1=300, f2=1500, f3=None, confidence=1.0),
        lpc_ok=False, te_ok=True,
        primary="te", front=False, vowel_hint=None, dbg=dbg
    )
    assert chosen == "te"
    assert dbg["selection_case"] == "only_te_ok"


def test_select_both_bad():
    dbg = {}
    chosen, *_ = select_formants(
        lpc=SimpleNS(f1=None, f2=None, f3=None, confidence=0.8),
        te=SimpleNS(f1=None, f2=None, f3=None, confidence=1.0),
        lpc_ok=False, te_ok=False,
        primary="lpc", front=False, vowel_hint=None, dbg=dbg
    )
    assert chosen == "lpc"
    assert dbg["selection_case"] == "both_bad"
