# tests/conftest.py
import os, sys
import pytest
import numpy as np
from scipy.signal import iirpeak, lfilter

# Ensure project root is importable for pytest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from voice_analysis import Analyzer

@pytest.fixture
def analyzer():
    return Analyzer(voice_type="tenor", smoothing=True, smooth_size=3)

@pytest.fixture
def sr():
    return 16000


def synth_vowel(formants, *args, sr=None, dur=0.4, f0=130.0, **kwargs):
    """
    Robust synth helper:
      - accepts sr as positional or keyword
      - always initializes src before use
      - tuning knobs: bw_base, harmonic_exp
    """
    # Resolve sr from args or keyword
    if sr is None:
        if args:
            try:
                sr = int(args[0])
            except Exception:
                sr = 16000
        else:
            sr = 16000
    sr = int(sr)

    # tuning knobs (override via kwargs if desired)
    bw_base = float(kwargs.get("bw_base", kwargs.get("bw", 40.0)))
    harmonic_exp = float(kwargs.get("harmonic_exp", 0.35))
    max_harm = min(120, max(12, int((sr / 2) / max(50.0, f0))))

    t = np.linspace(0, dur, int(sr * dur), endpoint=False)

    # ALWAYS initialize src before using it
    src = np.zeros_like(t)

    # richer harmonic source with controlled tilt
    for n in range(1, max_harm):
        src += (1.0 / (n ** harmonic_exp)) * np.sin(2 * np.pi * f0 * n * t)

    y = src

    # apply resonant bandpass for each formant; narrower for higher formants
    for i, fc in enumerate(formants):
        # scale bandwidth by index so F2/F3 get narrower and stronger
        if i == 0:
            bw = bw_base * 1.0
        elif i == 1:
            bw = bw_base * 0.35
        else:
            bw = bw_base * 0.3
        # ensure Q is reasonable
        Q = max(1.0, fc / max(1.0, bw))
        b, a = iirpeak(fc, Q=Q, fs=sr)
        y = lfilter(b, a, y)

    # small gain and normalize
    y = y * 0.95
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)


def unpack_formants(res):
    """
    Normalize the various return shapes from estimate_formants_lpc into a 3-tuple (f1,f2,f3).
    Accepts None, (f1,f2), (f1,f2,f3), or (f1,f2,f3,candidates).
    """
    if res is None:
        return None, None, None
    if isinstance(res, (tuple, list)):
        if len(res) >= 3:
            return res[0], res[1], res[2]
        if len(res) == 2:
            return res[0], res[1], None
    return None, None, None