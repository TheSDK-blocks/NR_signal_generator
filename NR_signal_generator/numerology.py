import numpy as np


def nr_cp_lengths(n_symb, mu, num):
    """
    Return the cyclic prefix length of each OFDM symbol.

        N_cp(l) = Ncp1  if l mod (7 * 2^mu) = 0
                  Ncp2  otherwise

    The first symbol of every half-subframe (0.5 ms) has the longer cyclic
    prefix (TS 38.211, Section 5.3.1).

    Parameters
    ----------
    n_symb : integer
        Number of OFDM symbols
    mu : integer (0,1,2,3,4)
        5G NR numerology
    num : dict
        NR parameters as returned by NRparameters, with keys "Ncp1" and "Ncp2"

    Returns
    -------
    Array of n_symb cyclic prefix lengths in samples.

    Example
    -------
    nr_cp_lengths(n_symb=14, mu=1, num=self.NRparameters(mu=1, BW=20e6))

    """
    long_cp = np.arange(n_symb) % (7 * 2**mu) == 0
    return np.where(long_cp, num["Ncp1"], num["Ncp2"])
