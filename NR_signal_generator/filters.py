import numpy as np
import scipy.signal as sig

from . import numerology


def nr_filter(x, fs, bw, fr):
    """
    Return x low-pass filtered to channel bandwidth bw, and the filter order.

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
    Filtered signal of len(x) + order samples, delay compensated by a
    circular shift, and the filter order.
    """
    guard = bw - numerology.max_bw_config(fr, bw)
    numtaps, beta = sig.kaiserord(80, guard / (fs / 2))
    if numtaps % 2 == 0:
        numtaps += 1
    taps = sig.firwin(numtaps, bw / 2, window=("kaiser", beta), fs=fs)
    order = numtaps - 1
    y = sig.fftconvolve(x, taps)
    return np.roll(y, -order // 2), order
