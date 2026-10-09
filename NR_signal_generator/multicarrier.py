import numpy as np

from . import numerology
from . import ofdm


def plan(fr, bw, bwp, osr):
    """
    Return the sample rate and frequency plan of a multi-carrier signal.

    Carriers are placed side by side in list order, centred on 0 Hz. A carrier
    with negative bandwidth is a gap of that width. The common sample rate is
    the smallest multiple of the LCM of the carrier sample rates that is at
    least their sum, times osr.

    Parameters
    ----------
    fr : string ("FR1", "FR2-1")
        Frequency range
    bw : array
        Channel bandwidth of each carrier in Hz, negative for a gap
    bwp : array
        Bandwidth part [mu, Nsymb, start, size] of each carrier
    osr : integer
        Oversampling ratio of the common sample rate

    Returns
    -------
    Dictionary with the common sample rate "Fs", the oversampling ratio "osr"
    of each carrier relative to its own sample rate, the centre frequency
    "f_off" of each carrier, and the signal period "length" in samples.
    """
    n = len(bw)
    fs_carrier = np.zeros(n)
    length_carrier = np.zeros(n)
    lcm = 1
    for i in range(n):
        if bw[i] != 0:
            dl = numerology.nr_parameters(fr, bwp[i][0], abs(bw[i]))
            fs_carrier[i] = dl["Fs"]
            lcm = np.lcm(lcm, int(dl["Fs"]))
            if bw[i] > 0:
                ncp = numerology.nr_cp_lengths(bwp[i][1], bwp[i][0], dl)
                length_carrier[i] = ofdm.symbol_starts(ncp, dl["NFFT"])[-1]

    fs = osr * lcm * np.ceil(fs_carrier.sum() / lcm)
    osr_carrier = np.zeros(n)
    f_off = np.zeros(n)
    bw_abs = np.abs(bw)
    for i in range(n):
        if bw[i] != 0:
            osr_carrier[i] = fs / fs_carrier[i]
        if bw[i] > 0:
            f_off[i] = bw_abs[:i].sum() + bw[i] / 2 - bw_abs.sum() / 2

    length = max(osr_carrier[i] * length_carrier[i] for i in range(n) if bw[i] > 0)
    return {"Fs": fs, "osr": osr_carrier, "f_off": f_off, "length": int(length)}
