import numpy as np


def equalize(RE, RE_id, start, stop):
    """
    Return the zero-forcing equalized resource-element grid (TS 38.104, Annex B.6).

    The equalizer coefficients are estimated from the complex ratios
    Z(k, l) = RE(k, l) / RE_id(k, l) at the DMRS, on the even subcarriers k of
    the BWP in the symbols l with l mod 14 = 2:

        A(k)   = mean_l |Z(k, l)|
        Phi(k) = mean_l unwrap_l(arg Z(k, l)),  unwrapping jumps of at least pi
        A, Phi smoothed across the DMRS subcarriers k_j by a moving average of
               2 h_j + 1 samples, h_j = min(j, K - 1 - j, 9)
        H(k)   = interp(A)(k) exp(j interp(Phi)(k)) for every subcarrier k

    and the grid is equalized as RE(k, l) / H(k). The averaging spans the whole
    signal instead of the 10 ms of the specification.

    Parameters
    ----------
    RE : array
        Received resource-element grid of shape (Nsc, Nsymb)
    RE_id : array
        Ideal grid of the same shape: the DMRS and PSS values at their
        positions and 1 elsewhere
    start : integer
        First subcarrier of the BWP
    stop : integer
        Last subcarrier of the BWP plus one

    Returns
    -------
    Equalized complex grid of shape (Nsc, Nsymb).

    Example
    -------
    equalize(RE, RE_id, start=0, stop=612)

    """
    Nsc, n_symb = RE.shape
    if n_symb < 3:
        raise ValueError(
            f"no DMRS symbol: equalization needs at least 3 OFDM symbols, got {n_symb}"
        )

    k = np.arange(start, stop, 2)
    l = np.flatnonzero(np.arange(n_symb) % 14 == 2)
    z = RE[np.ix_(k, l)] / RE_id[np.ix_(k, l)]
    amplitude = moving_average(np.abs(z).mean(axis=1))
    phase = moving_average(np.unwrap(np.angle(z), axis=1).mean(axis=1))

    subcarriers = np.arange(Nsc)
    h = np.interp(subcarriers, k, amplitude) * np.exp(1j * np.interp(subcarriers, k, phase))
    return RE / h[:, None]


def moving_average(x, half_width=9):
    """
    Return the centred moving average of x, with the window shrunk
    symmetrically near the edges.

        y(j) = mean(x(j - h_j .. j + h_j)),  h_j = min(j, len(x) - 1 - j, half_width)

    Parameters
    ----------
    x : array
        Real samples
    half_width : integer
        Largest half-width of the window; the full window is 2 half_width + 1

    Returns
    -------
    Array of len(x) averages.

    Example
    -------
    moving_average(np.arange(10.0), half_width=2)

    """
    j = np.arange(len(x))
    h = np.minimum(np.minimum(j, len(x) - 1 - j), half_width)
    c = np.concatenate(([0.0], np.cumsum(x)))
    return (c[j + h + 1] - c[j - h]) / (2 * h + 1)
