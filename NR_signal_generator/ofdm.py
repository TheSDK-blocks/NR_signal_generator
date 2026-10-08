import numpy as np


def symbol_starts(ncp, nfft):
    """
    Return the start sample of each OFDM symbol, including its cyclic prefix.

        T_0 = 0,  T_(i+1) = T_i + nfft + ncp_i

    Parameters
    ----------
    ncp : array
        Cyclic prefix length of each OFDM symbol in samples
    nfft : integer
        FFT size in samples

    Returns
    -------
    Array of len(ncp) + 1 sample indices. The last element is the total
    number of samples.

    Example
    -------
    symbol_starts(ncp=[160, 144, 144], nfft=2048)

    """
    return np.concatenate(([0], np.cumsum(nfft + np.asarray(ncp))))
