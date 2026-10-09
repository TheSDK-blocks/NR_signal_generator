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


# Minimum guard band in kHz per numerology mu and channel bandwidth in MHz
# (TS 38.104, Tables 5.3.3-1 and 5.3.3-2)
GUARD_BAND = {
    "FR1": {
        0: {3: 142.5, 5: 242.5, 7: 342.5, 10: 312.5, 15: 382.5, 20: 452.5,
            25: 522.5, 30: 592.5, 35: 572.5, 40: 552.5, 45: 712.5, 50: 692.5},
        1: {5: 505, 10: 665, 15: 645, 20: 805, 25: 785, 30: 945, 35: 925,
            40: 905, 45: 1065, 50: 1045, 60: 825, 70: 965, 80: 925, 90: 885,
            100: 845},
        2: {10: 1010, 15: 990, 20: 1330, 25: 1310, 30: 1290, 35: 1630,
            40: 1610, 45: 1590, 50: 1570, 60: 1530, 70: 1490, 80: 1450,
            90: 1410, 100: 1370},
    },
    "FR2-1": {
        2: {50: 1210, 100: 2450, 200: 4930},
        3: {50: 1900, 100: 2420, 200: 4900, 400: 9860},
    },
}

# Channel rasters in Hz for which TS 38.104, Section 5.4.1.2 defines the
# nominal channel spacing
CHANNEL_RASTERS = {"FR1": (100e3, 15e3), "FR2-1": (60e3,)}


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


def guard_band(fr, mu, bw):
    """
    Return the minimum guard band of a carrier from TS 38.104,
    Table 5.3.3-1 (FR1) or Table 5.3.3-2 (FR2-1).

    Parameters
    ----------
    fr : string ("FR1", "FR2-1")
        Frequency range
    mu : integer
        5G NR numerology
    bw : float
        Channel bandwidth in Hz

    Returns
    -------
    Guard band in Hz.
    """
    try:
        return GUARD_BAND[fr][mu][bw / 1e6] * 1e3
    except KeyError:
        raise ValueError(
            f"{bw / 1e6:g} MHz at mu = {mu} is not a {fr} channel bandwidth (TS 38.104, Section 5.3.3)"
        ) from None


def nominal_spacing(fr, bw1, bw2, raster):
    """
    Return the nominal channel spacing of two adjacent carriers in intra-band
    contiguous carrier aggregation (TS 38.104, Section 5.4.1.2).

        spacing = floor((bw1 + bw2 - 2 |GB1 - GB2|) / (2 step)) * step

    step = 300 kHz for a 100 kHz raster, 15 kHz * 2^mu0 for a 15 kHz raster
    and 60 kHz * 2^(mu0 - 2) for a 60 kHz raster. mu0 is the largest
    numerology with both channel bandwidths; the guard bands GB are taken at
    mu0. The spec takes mu0 from the numerologies the operating band
    supports; this assumes the band supports every numerology of fr. With a
    15 kHz raster and no common numerology, mu0 = 1.

    Parameters
    ----------
    fr : string ("FR1", "FR2-1")
        Frequency range
    bw1, bw2 : float
        Channel bandwidths in Hz
    raster : float
        Channel raster of the operating band in Hz, 100e3 or 15e3 in FR1 and
        60e3 in FR2-1

    Returns
    -------
    Nominal channel spacing in Hz.
    """
    if raster not in CHANNEL_RASTERS[fr]:
        raise ValueError(f"{raster / 1e3:g} kHz is not a {fr} channel raster (TS 38.104, Section 5.4.2)")

    common = [mu for mu, gb in GUARD_BAND[fr].items() if bw1 / 1e6 in gb and bw2 / 1e6 in gb]
    if common:
        mu0 = max(common)
    elif raster == 15e3:
        mu0 = 1
    else:
        raise ValueError(f"{bw1 / 1e6:g} and {bw2 / 1e6:g} MHz have no common numerology in {fr}")

    if raster == 100e3:
        step = 300e3
    elif raster == 15e3:
        step = 15e3 * 2**mu0
    else:
        step = 60e3 * 2 ** (mu0 - 2)

    gb1 = guard_band(fr, mu0, bw1)
    gb2 = guard_band(fr, mu0, bw2)
    total = bw1 + bw2 - 2 * abs(gb1 - gb2)
    return np.floor(total / (2 * step)) * step
