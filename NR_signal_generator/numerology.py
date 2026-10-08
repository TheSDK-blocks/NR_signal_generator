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
