import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

from formant_utils import estimate_formants_lpc

# --- Simple vowel-like synthetic signal ---
def synth_vowel(f0=120, f1=500, f2=1500, sr=16000, dur=1.0):
    """
    Generate a synthetic vowel-like signal using source-filter model.
    f0 = pitch, f1/f2 = formant center frequencies
    """
    t = np.arange(int(sr*dur)) / sr
    # Source: impulse train at f0
    source = np.sin(2*np.pi*f0*t)
    # Formant filters: simple resonators
    def resonator(fc, bw=100):
        R = np.exp(-np.pi*bw/sr)
        theta = 2*np.pi*fc/sr
        a = [1, -2*R*np.cos(theta), R**2]
        b = [1-R]
        return b, a

    b1, a1 = resonator(f1)
    b2, a2 = resonator(f2)

    y = lfilter(b1, a1, source)
    y = lfilter(b2, a2, y)
    return y

# --- Generate synthetic /a/ vowel ---
sr = 16000
y = synth_vowel(f0=120, f1=700, f2=1200, sr=sr, dur=1.0)

# --- Estimate formants ---
from tests.conftest import unpack_formants
res = estimate_formants_lpc(y, sr)
f1, f2, _ = unpack_formants(res)
print("Estimated F1:", f1, "Estimated F2:", f2)

# --- Plot spectrum for sanity check ---
plt.specgram(y, NFFT=1024, Fs=sr, noverlap=512)
plt.title("Synthetic vowel spectrum")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.show()