import numpy as np
from scipy import signal

from . import sequences


def pss_align(x, nfft, pss_start, n_id2):
    """
    Return x circularly shifted so that the PSS symbol starts at pss_start.

    The PSS delay is the peak of the cross-correlation of x with the
    time-domain PSS, d_PSS(n) on subcarriers n - 64 around DC as mapped by
    the generator.

    Parameters
    ----------
    x : array
        Complex time signal of one BWP
    nfft : integer
        FFT size in samples
    pss_start : integer
        Sample at which the useful part of the PSS symbol starts in the
        aligned signal, T_l + ncp_l of the PSS symbol l
    n_id2 : integer
        Second cell ID defined for 5G NR, N_ID^(2) in {0, 1, 2}

    Returns
    -------
    Complex time signal of len(x) samples.

    Example
    -------
    pss_align(x, nfft=2048, pss_start=4400, n_id2=0)

    """
    replica = np.fft.ifft(np.roll(np.pad(sequences.pss(n_id2), (0, int(nfft) - 127)), -64))
    delay = np.argmax(np.abs(signal.correlate(x, replica, "full"))) - (len(replica) - 1)
    return np.roll(x, int(pss_start - delay))
