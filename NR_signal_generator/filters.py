import numpy as np
import scipy.signal as sig

from . import numerology


def nr_filter(x, fs, bw, fr):
    """
    Return x, one period of a cyclic signal, circularly low-pass filtered to
    channel bandwidth bw.

    Kaiser-window FIR with 80 dB stopband attenuation, cut-off at bw / 2 and
    its transition band in the guard band of the channel.

    Parameters
    ----------
    x : array
        Signal to be filtered
    fs : float
        Sampling frequency
    bw : float
        Channel bandwidth
    fr : string ("FR1", "FR2-1")
        Frequency range

    Returns
    -------
    Filtered signal of len(x) samples, delay compensated by a circular
    shift.
    """
    if len(x) == 0:
        return np.zeros(0, complex)
    guard = bw - numerology.max_bw_config(fr, bw)
    numtaps, beta = sig.kaiserord(80, guard / (fs / 2))
    if numtaps % 2 == 0:
        numtaps += 1
    taps = sig.firwin(numtaps, bw / 2, window=("kaiser", beta), fs=fs)
    # Fold the taps onto one period so that filters longer than x are exact
    wrapped = np.arange(numtaps) % len(x)
    h = np.bincount(wrapped, weights=taps, minlength=len(x))
    y = np.fft.ifft(np.fft.fft(x) * np.fft.fft(h))
    return np.roll(y, -(numtaps // 2))
