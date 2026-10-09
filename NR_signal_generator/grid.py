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


def dmrs_mask(n_rb, n_symb, bwp_start, bwp_size):
    """
    Return the DMRS resource elements of a BWP grid.

    DMRS occupy the even subcarriers of the BWP, counted from point A, the
    lowest subcarrier of the carrier, in symbols l with l mod 14 = DMRS_SYMBOL.

    Parameters
    ----------
    n_rb : integer
        Number of resource blocks of the carrier
    n_symb : integer
        Number of OFDM symbols
    bwp_start : integer
        First resource block of the BWP, N_BWP^start
    bwp_size : integer
        Number of resource blocks of the BWP, N_BWP^size

    Returns
    -------
    Boolean array of shape (12 * n_rb, n_symb), True for DMRS REs.

    Raises
    ------
    ValueError
        If the BWP is not whole resource blocks within the carrier.

    Example
    -------
    dmrs_mask(n_rb=51, n_symb=14, bwp_start=0, bwp_size=51)

    """
    if bwp_start != int(bwp_start) or bwp_size != int(bwp_size):
        raise ValueError(f"BWP start and size must be whole RBs, got {bwp_start}, {bwp_size}")
    if bwp_start < 0 or bwp_size < 1 or bwp_start + bwp_size > n_rb:
        raise ValueError(
            f"BWP of {bwp_size} RBs from RB {bwp_start} does not fit the carrier of {n_rb} RBs"
        )

    start = 12 * int(bwp_start)
    stop = 12 * int(bwp_start + bwp_size)
    mask = np.zeros((12 * n_rb, n_symb), dtype=bool)
    mask[start:stop:2, np.arange(n_symb) % 14 == DMRS_SYMBOL] = True
    return mask


def data_re_mask(n_rb, n_symb, bwp_start, bwp_size):
    """
    Return the data resource elements of a BWP grid.

    mask: boolean array of shape (12 * n_rb, n_symb), True for data REs.

    Excluded REs:
        - subcarriers outside [12 * bwp_start, 12 * (bwp_start + bwp_size))
        - DMRS: even subcarriers of the BWP in symbols l with l mod 14 = 2
        - PSS block: 240 subcarriers centred in the carrier, symbol pss_symbol(n_symb)

    This is a simplified NR-like layout, not TS 38.211 compliant: the SS/PBCH
    block occupies a single symbol (no SSS or PBCH).
    """
    n_sc = 12 * n_rb
    mask = np.zeros((n_sc, n_symb), dtype=bool)
    dmrs = dmrs_mask(n_rb, n_symb, bwp_start, bwp_size)
    mask[12 * int(bwp_start) : 12 * int(bwp_start + bwp_size)] = True
    mask[dmrs] = False

    pss_start = n_sc // 2 - 120
    mask[pss_start : pss_start + 240, pss_symbol(n_symb)] = False

    return mask


def reference_grid(n_rb, n_symb, mu, bwp_start, bwp_size, n_id_cell, n_id2):
    """
    Return the DMRS and PSS resource elements of a BWP grid.

    values: complex array of shape (12 * n_rb, n_symb), reference values at is_ref.
    is_ref: boolean array of shape (12 * n_rb, n_symb), True for reference REs.

    DMRS: dmrs_mask, with the sequence r(m) counted from point A, the lowest
    subcarrier of the carrier, so that a BWP starting at RB bwp_start carries
    r(6 bwp_start) onwards; the sequence restarts every frame of 140 * 2^mu
    symbols.
    PSS: 240 subcarriers centred in the carrier, [0]*56 + d_PSS + [0]*57,
    in symbol pss_symbol(n_symb).

    This is a simplified NR-like layout, not TS 38.211 compliant: the PSS is
    sent without the rest of the SS/PBCH block (SSS, PBCH and its DMRS), and
    is centred in the carrier instead of placed on the synchronization raster.
    """
    n_sc = 12 * n_rb
    is_ref = dmrs_mask(n_rb, n_symb, bwp_start, bwp_size)
    k = np.flatnonzero(is_ref.any(axis=1))
    l = np.flatnonzero(is_ref.any(axis=0))

    values = np.zeros((n_sc, n_symb), complex)
    symb_in_frame = 2**mu * 10 * 14
    for frame_start in range(0, n_symb, symb_in_frame):
        n = min(symb_in_frame, n_symb - frame_start)
        l_frame = l[(l >= frame_start) & (l < frame_start + n)]
        if l_frame.size == 0:
            continue
        r_dmrs = sequences.dmrs(n_id_cell, n, int(bwp_start + bwp_size))
        values[np.ix_(k, l_frame)] = r_dmrs[6 * int(bwp_start) :, l_frame - frame_start]

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
    data_symbols(re_grid, data_re_mask(n_rb=51, n_symb=14, bwp_start=0, bwp_size=51))

    """
    if np.shape(re_grid) != mask.shape:
        raise ValueError(
            f"grid of shape {np.shape(re_grid)} does not match the data mask of shape {mask.shape}"
        )
    return re_grid.T[mask.T]


def map_data(re_grid, mask, symbols):
    """
    Return a copy of re_grid with symbols mapped in data_symbols order.

    Parameters
    ----------
    re_grid : array
        Resource-element grid of shape (Nsc, Nsymb)
    mask : array of bool
        Data resource elements of the grid
    symbols : array
        Data symbols, at most mask.sum()

    Returns
    -------
    Complex array of shape (Nsc, Nsymb).
    """
    if np.shape(re_grid) != mask.shape:
        raise ValueError(
            f"grid of shape {np.shape(re_grid)} does not match the data mask of shape {mask.shape}"
        )
    l, k = np.nonzero(mask.T)
    if len(symbols) > len(k):
        raise ValueError(f"{len(symbols)} data symbols do not fit {len(k)} data resource elements")
    out = np.array(re_grid, complex)
    out[k[: len(symbols)], l[: len(symbols)]] = symbols
    return out
