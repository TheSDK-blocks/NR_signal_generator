import numpy as np

# Maximum transmission bandwidth configuration N_RB per numerology mu and
# channel bandwidth in MHz (TS 38.104, Tables 5.3.2-1 and 5.3.2-2)
N_RB = {
    "FR1": {
        0: {3: 15, 5: 25, 7: 35, 10: 52, 15: 79, 20: 106, 25: 133, 30: 160,
            35: 188, 40: 216, 45: 242, 50: 270},
        1: {5: 11, 10: 24, 15: 38, 20: 51, 25: 65, 30: 78, 35: 92, 40: 106,
            45: 119, 50: 133, 60: 162, 70: 189, 80: 217, 90: 245, 100: 273},
        2: {10: 11, 15: 18, 20: 24, 25: 31, 30: 38, 35: 44, 40: 51, 45: 58,
            50: 65, 60: 79, 70: 93, 80: 107, 90: 121, 100: 135},
    },
    "FR2-1": {
        2: {50: 66, 100: 132, 200: 264},
        3: {50: 32, 100: 66, 200: 132, 400: 264},
    },
}


def n_rb(fr, mu, bw):
    """
    Return the number of resource blocks of a carrier, N_RB, from
    TS 38.104, Table 5.3.2-1 (FR1) or Table 5.3.2-2 (FR2-1).

    Parameters
    ----------
    fr : string ("FR1", "FR2-1")
        Frequency range
    mu : integer
        5G NR numerology, 0, 1 or 2 in FR1 and 2 or 3 in FR2-1
    bw : float
        Channel bandwidth in Hz

    Raises
    ------
    ValueError
        If the table has no entry for fr, mu and bw.

    Example
    -------
    n_rb(fr="FR1", mu=1, bw=100e6)

    """
    try:
        return N_RB[fr][mu][bw / 1e6]
    except KeyError:
        raise ValueError(
            f"{bw / 1e6:g} MHz at mu = {mu} is not a {fr} channel bandwidth (TS 38.104, Section 5.3.2)"
        ) from None


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
        NR parameters as returned by nr_parameters, with keys "Ncp1" and "Ncp2"

    Returns
    -------
    Array of n_symb cyclic prefix lengths in samples.

    Example
    -------
    nr_cp_lengths(n_symb=14, mu=1, num=nr_parameters("FR1", 1, 20e6))

    """
    long_cp = np.arange(n_symb) % (7 * 2**mu) == 0
    return np.where(long_cp, num["Ncp1"], num["Ncp2"])


def nr_parameters(fr, mu, bw, osr=1):
    """
    Return the carrier parameters of numerology mu and channel bandwidth bw.

    Parameters
    ----------
    fr : string ("FR1", "FR2-1")
        Frequency range
    mu : integer (0,1,2,3)
        5G NR numerology
    bw : float
        Channel bandwidth in Hz
    osr : integer
        Oversampling factor

    Returns
    -------
    Dictionary with keys Fs, NFFT, Ncp1, Ncp2 (scaled by osr), RB, Nsc and
    Lroll.
    """
    RB = n_rb(fr, mu, bw)
    if RB < 20:
        raise ValueError(
            f"carrier of {RB} RBs at mu = {mu} cannot hold the 20 RB SS/PBCH block (TS 38.211, Section 7.4.3.1)"
        )
    # FFT sizes, the smallest one with at most 85 % occupancy of the
    # occupied subcarriers is used
    fft_sizes = (128, 256, 512, 1024, 2048, 4096)
    # Raised-cosine roll-off length per FFT size, chosen to keep EVM < 1 %
    l_roll = {128: 4, 256: 6, 512: 4, 1024: 6, 2048: 8, 4096: 10}

    n_sc = 12 * RB
    scs = 15e3 * 2**mu
    nfft = next(n for n in fft_sizes if 0.85 * n >= n_sc)
    fs = nfft * scs

    # Cyclic prefix lengths in samples at fs (TS 38.211, Section 5.3.1)
    ncp = 144 * nfft / 2048
    ncp_long = ncp + 16 * 2**mu * nfft / 2048

    return {
        "Fs": fs * osr,
        "NFFT": nfft * osr,
        "Ncp1": ncp_long * osr,
        "Ncp2": ncp * osr,
        "RB": RB,
        "Nsc": n_sc,
        "Lroll": l_roll[nfft],
    }


def max_bw_config(fr, bw):
    """
    Return the largest transmission bandwidth configuration, N_RB * 12 * SCS,
    of channel bandwidth bw over the SCS of fr (TS 38.104, Table 6.6.3.2-1,
    Note 2).

    Parameters
    ----------
    fr : string ("FR1", "FR2-1")
        Frequency range
    bw : float
        Channel bandwidth in Hz

    Returns
    -------
    Transmission bandwidth in Hz.
    """
    bw_config = []
    for mu, n_rb_of_bw in N_RB[fr].items():
        if bw / 1e6 in n_rb_of_bw:
            bw_config.append(n_rb_of_bw[bw / 1e6] * 12 * 15e3 * 2**mu)
    return max(bw_config)
