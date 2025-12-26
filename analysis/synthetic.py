import numpy as np


def synthetic_vowel(f1, f2, sr=44100, dur=0.05):
    """
    Generate a simple synthetic vowel-like signal with two formants.

    Parameters
    ----------
    f1 : float
        First formant frequency in Hz.
    f2 : float
        Second formant frequency in Hz.
    sr : int
        Sample rate.
    dur : float
        Duration in seconds.

    Returns
    -------
    np.ndarray
        The synthetic audio signal.
    """
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sig = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    sig *= np.hamming(len(sig))
    return sig
