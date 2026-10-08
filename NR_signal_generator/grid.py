import numpy as np

from . import sequences

DMRS_SYMBOL = 2  # symbol l mod 14 carrying DMRS


def pss_symbol(n_symb):
    """
    Return the OFDM symbol carrying the PSS.

    The PSS is in symbol 3 of the first frame. Signals shorter than 4
    symbols carry it in their last symbol instead, moved one symbol earlier
    if that symbol carries DMRS (l mod 14 = DMRS_SYMBOL), so the two never collide.

    This placement is a simplification, not TS 38.213 compliant: the SS/PBCH
    block candidate positions of TS 38.213, Section 4.1, its periodicity and
    its SSS and PBCH are not modelled.

    Parameters
    ----------
    n_symb : integer
        Number of OFDM symbols

    Example
    -------
    pss_symbol(n_symb=3)

    """
    l = min(n_symb, 4) - 1
    if l % 14 == DMRS_SYMBOL:
        l -= 1
    return l


def dmrs_mask(n_rb, n_symb, lo, hi):
    """
    Return the DMRS resource elements of a BWP grid.

    DMRS occupy the even subcarriers of the BWP, counted from its first
    subcarrier floor(n_rb * lo) * 12, in symbols l with l mod 14 = DMRS_SYMBOL.

    Parameters
    ----------
    n_rb : integer
        Number of resource blocks of the carrier
    n_symb : integer
        Number of OFDM symbols
    lo : float
        Lower edge of the BWP as a fraction of the carrier
    hi : float
        Upper edge of the BWP as a fraction of the carrier

    Returns
    -------
    Boolean array of shape (12 * n_rb, n_symb), True for DMRS REs.

    Example
    -------
    dmrs_mask(n_rb=51, n_symb=14, lo=0, hi=1)

    """
    start = int(np.floor(n_rb * lo) * 12)
    stop = int(np.ceil(n_rb * hi) * 12)
    mask = np.zeros((12 * n_rb, n_symb), dtype=bool)
    mask[start:stop:2, np.arange(n_symb) % 14 == DMRS_SYMBOL] = True
    return mask


def data_re_mask(n_rb, n_symb, lo, hi):
    """
    Return the data resource elements of a BWP grid.

    mask: boolean array of shape (12 * n_rb, n_symb), True for data REs.

    Excluded REs:
        - subcarriers outside [floor(n_rb * lo) * 12, ceil(n_rb * hi) * 12)
        - DMRS: even subcarriers of the BWP in symbols l with l mod 14 = 2
        - PSS block: 240 subcarriers centred in the carrier, symbol pss_symbol(n_symb)

    This is a simplified NR-like layout, not TS 38.211 compliant:
    the SS/PBCH block occupies a single symbol (no SSS or PBCH), and the
    DMRS comb is referenced to the BWP start instead of point A.
    """
    n_sc = 12 * n_rb
    start = int(np.floor(n_rb * lo) * 12)
    stop = int(np.ceil(n_rb * hi) * 12)

    mask = np.zeros((n_sc, n_symb), dtype=bool)
    mask[start:stop] = True

    mask[dmrs_mask(n_rb, n_symb, lo, hi)] = False

    pss_start = n_sc // 2 - 120
    mask[pss_start : pss_start + 240, pss_symbol(n_symb)] = False

    return mask


def reference_grid(n_rb, n_symb, mu, lo, hi, n_id_cell, n_id2):
    """
    Return the DMRS and PSS resource elements of a BWP grid.

    values: complex array of shape (12 * n_rb, n_symb), reference values at is_ref.
    is_ref: boolean array of shape (12 * n_rb, n_symb), True for reference REs.

    DMRS: even subcarriers of the BWP in symbols l with l mod 14 = 2; the
    sequence restarts every frame of 140 * 2^mu symbols.
    PSS: 240 subcarriers centred in the carrier, [0]*56 + d_PSS + [0]*57,
    in symbol pss_symbol(n_symb).

    This is a simplified NR-like layout, not TS 38.211 compliant: the PSS is
    sent without the rest of the SS/PBCH block (SSS, PBCH and its DMRS), and
    is centred in the carrier instead of placed on the synchronization raster.
    """
    n_sc = 12 * n_rb
    is_ref = dmrs_mask(n_rb, n_symb, lo, hi)
    k = np.flatnonzero(is_ref.any(axis=1))
    l = np.flatnonzero(is_ref.any(axis=0))

    values = np.zeros((n_sc, n_symb), complex)
    symb_in_frame = 2**mu * 10 * 14
    for frame_start in range(0, n_symb, symb_in_frame):
        n = min(symb_in_frame, n_symb - frame_start)
        l_frame = l[(l >= frame_start) & (l < frame_start + n)]
        if l_frame.size == 0:
            continue
        r_dmrs = sequences.dmrs(n_id_cell, n, len(k) / 6)
        values[np.ix_(k, l_frame)] = r_dmrs[:, l_frame - frame_start]

    l_pss = pss_symbol(n_symb)
    pss_start = n_sc // 2 - 120
    values[pss_start : pss_start + 240, l_pss] = np.concatenate(
        (np.zeros(56), sequences.pss(n_id2), np.zeros(57))
    )
    is_ref[pss_start : pss_start + 240, l_pss] = True

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
