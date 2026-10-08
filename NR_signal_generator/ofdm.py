import numpy as np


def symbol_starts(ncp, nfft):
    """
    Return the start sample of each OFDM symbol, including its cyclic prefix.

        T_0 = 0,  T_(i+1) = T_i + nfft + ncp_i

    Parameters
    ----------
    ncp : array
        Cyclic prefix length of each OFDM symbol in samples
    nfft : integer
        FFT size in samples

    Returns
    -------
    Array of len(ncp) + 1 sample indices. The last element is the total
    number of samples.

    Example
    -------
    symbol_starts(ncp=[160, 144, 144], nfft=2048)

    """
    return np.concatenate(([0], np.cumsum(nfft + np.asarray(ncp))))


def modulate(grid, nfft, ncp, n_roll):
    """
    Return the cyclic, windowed OFDM time signal of a resource-element grid.

    Each symbol is the IFFT of its subcarriers, extended by its cyclic
    prefix and a cyclic suffix of n_roll samples, and tapered with
    complementary raised-cosine edges of n_roll samples (weighted
    overlap-add). The fall of each symbol overlaps the rise of the next
    over the first n_roll samples of its cyclic prefix, where the two
    weights sum to one. The signal is cyclic: the suffix of the last symbol
    wraps to the start. n_roll = 0 gives plain CP-OFDM.

        x(T_i + r) += w_i(r) X_i((r - ncp_i) mod nfft),  0 <= r < ncp_i + nfft + n_roll

    where X_i is the IFFT of symbol i and T_i its start from symbol_starts.

    Parameters
    ----------
    grid : array
        Resource-element grid of shape (Nsc, Nsymb). Row Nsc / 2 is the DC
        subcarrier, lower rows are negative frequencies.
    nfft : integer
        FFT size in samples, nfft >= Nsc
    ncp : array
        Cyclic prefix length of each OFDM symbol in samples
    n_roll : integer
        Raised-cosine overlap length in samples, at most min(ncp)

    Returns
    -------
    Complex time signal of symbol_starts(ncp, nfft)[-1] samples.

    Example
    -------
    modulate(grid, nfft=2048, ncp=nr_cp_lengths(14, 1, num), n_roll=8)

    """
    nfft = int(nfft)
    ncp = np.asarray(ncp).astype(int)
    n_roll = int(n_roll)
    n_sc, n_symb = grid.shape

    pad_lo = nfft // 2 - n_sc // 2
    padded = np.pad(grid, ((pad_lo, nfft - n_sc - pad_lo), (0, 0)))
    symbols = np.fft.ifft(np.fft.ifftshift(padded, axes=0), axis=0)

    rise = 0.5 * (1 - np.cos(np.pi * (np.arange(n_roll) + 0.5) / n_roll))

    starts = symbol_starts(ncp, nfft)
    x = np.zeros(starts[-1], complex)
    for i in range(n_symb):
        window = np.concatenate((rise, np.ones(ncp[i] + nfft - n_roll), rise[::-1]))
        r = np.arange(ncp[i] + nfft + n_roll)
        segment = symbols[(r - ncp[i]) % nfft, i] * window
        np.add.at(x, (starts[i] + r) % starts[-1], segment)
    return x


def demodulate(x, nfft, ncp, n_sc):
    """
    Return the resource-element grid of an OFDM time signal.

    The FFT window of each symbol starts in the middle of its cyclic prefix,
    ceil(ncp_i / 2) samples before the nominal position, which keeps it clear
    of the windowed start of the cyclic prefix and of timing errors in either
    direction. The resulting linear phase is removed after the FFT.

        s_i = -ceil(ncp_i / 2)
        Y_i(k) = FFT{x(T_i + ncp_i + s_i + n)}(k) exp(-j 2 pi k s_i / nfft),  0 <= n < nfft

    where T_i is the symbol start from symbol_starts. The grid is the n_sc
    subcarriers centred on DC, in the same layout that modulate takes.

    Parameters
    ----------
    x : array
        Complex time signal, time-aligned so that the first symbol starts at
        sample 0
    nfft : integer
        FFT size in samples
    ncp : array
        Cyclic prefix length of each OFDM symbol in samples
    n_sc : integer
        Number of occupied subcarriers, n_sc <= nfft

    Returns
    -------
    Complex resource-element grid of shape (n_sc, len(ncp)).

    Example
    -------
    demodulate(x, nfft=2048, ncp=nr_cp_lengths(14, 1, num), n_sc=1272)

    """
    nfft = int(nfft)
    ncp = np.asarray(ncp).astype(int)
    n_sc = int(n_sc)

    shift = -np.ceil(ncp / 2)
    window_start = (symbol_starts(ncp, nfft)[:-1] + ncp + shift).astype(int)
    symbols = x[window_start[:, None] + np.arange(nfft)].T

    k = np.arange(nfft)
    bins = np.fft.fft(symbols, axis=0) * np.exp(-1j * 2 * np.pi * k[:, None] * shift / nfft)

    pad_lo = nfft // 2 - n_sc // 2
    return np.fft.fftshift(bins, axes=0)[pad_lo : pad_lo + n_sc]
