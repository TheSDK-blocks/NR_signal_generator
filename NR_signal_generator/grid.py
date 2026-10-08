import numpy as np

from . import sequences


def data_re_mask(n_rb, n_symb, lo, hi):
    """
    Return the data resource elements of a BWP grid.

    mask: boolean array of shape (12 * n_rb, n_symb), True for data REs.

    Excluded REs:
        - subcarriers outside [floor(n_rb * lo) * 12, ceil(n_rb * hi) * 12)
        - DMRS: even subcarriers of the BWP in symbols l with l mod 14 = 2
        - PSS block: 240 subcarriers centred in the carrier, symbol min(n_symb, 4) - 1

    This is a simplified NR-like layout, not TS 38.211 compliant:
    the SS/PBCH block occupies a single symbol (no SSS or PBCH), and the
    DMRS comb is referenced to the BWP start instead of point A.
    """
    n_sc = 12 * n_rb
    start = int(np.floor(n_rb * lo) * 12)
    stop = int(np.ceil(n_rb * hi) * 12)

    mask = np.zeros((n_sc, n_symb), dtype=bool)
    mask[start:stop] = True

    dmrs_symbols = np.arange(n_symb) % 14 == 2
    mask[start:stop:2, dmrs_symbols] = False

    pss_start = n_sc // 2 - 120
    mask[pss_start : pss_start + 240, min(n_symb, 4) - 1] = False

    return mask


def reference_grid(n_rb, n_symb, mu, lo, hi, n_id_cell, n_id2):
    """
    Return the DMRS and PSS resource elements of a BWP grid.

    values: complex array of shape (12 * n_rb, n_symb), reference values at is_ref.
    is_ref: boolean array of shape (12 * n_rb, n_symb), True for reference REs.

    DMRS: even subcarriers of the BWP in symbols l with l mod 14 = 2; the
    sequence restarts every frame of 140 * 2^mu symbols.
    PSS: 240 subcarriers centred in the carrier, [0]*56 + d_PSS + [0]*57,
    in symbol min(N, 4) - 1 of the last frame, where N is its symbol count.
    The PSS overwrites DMRS where they overlap.
    """
    n_sc = 12 * n_rb
    start = int(np.floor(n_rb * lo) * 12)
    stop = int(np.ceil(n_rb * hi) * 12)

    values = np.zeros((n_sc, n_symb), complex)
    is_ref = np.zeros((n_sc, n_symb), dtype=bool)

    symb_in_frame = 2**mu * 10 * 14
    for frame_start in range(0, n_symb, symb_in_frame):
        n = min(symb_in_frame, n_symb - frame_start)
        r_dmrs = sequences.dmrs(n_id_cell, n, (stop - start) / 12)
        dmrs_symbols = np.flatnonzero(np.arange(n) % 14 == 2)
        values[start:stop:2, frame_start + dmrs_symbols] = r_dmrs[:, dmrs_symbols]
        is_ref[start:stop:2, frame_start + dmrs_symbols] = True

    pss_symbol = frame_start + min(n, 4) - 1
    pss_start = n_sc // 2 - 120
    values[pss_start : pss_start + 240, pss_symbol] = np.concatenate(
        (np.zeros(56), sequences.pss(n_id2), np.zeros(57))
    )
    is_ref[pss_start : pss_start + 240, pss_symbol] = True

    return values, is_ref


def data_symbols(re_grid, mask):
    """
    Return the data symbols of a resource-element grid.

    Symbols are taken by position, in column-major order: symbol by symbol,
    ascending subcarrier within each symbol. This is the order in which the
    generator maps data to the grid.

    Parameters
    ----------
    re_grid : array
        Resource-element grid of shape (Nsc, Nsymb)
    mask : array of bool
        Data resource elements of the grid, as returned by data_re_mask

    Returns
    -------
    One-dimensional array of mask.sum() data symbols.

    Example
    -------
    data_symbols(re_grid, data_re_mask(n_rb=51, n_symb=14, lo=0, hi=1))

    """
    return re_grid.T[mask.T]
