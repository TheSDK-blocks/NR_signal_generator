import numpy as np


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
