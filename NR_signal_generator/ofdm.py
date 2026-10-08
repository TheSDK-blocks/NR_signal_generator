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
    prefix and tapered with a raised-cosine window of n_roll samples on
    each side. The tapers of neighbouring symbols overlap and add
    (weighted overlap-add). The signal is cyclic: the taper of the last
    symbol wraps to the start and that of the first symbol to the end.

        x(T_i + r) += w_i(r) X_i((r - ncp_i) mod nfft),  -n_roll <= r < ncp_i + nfft + n_roll - 1

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
        Raised-cosine taper length in samples

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

    rise = 0.5 * (1 + np.cos(np.pi * np.arange(n_roll + 1, 2 * n_roll + 1) / n_roll))
    fade = 0.5 * (1 + np.cos(np.pi * np.arange(0, n_roll) / n_roll))

    starts = symbol_starts(ncp, nfft)
    x = np.zeros(starts[-1], complex)
    for i in range(n_symb):
        window = np.concatenate((rise, np.ones(ncp[i] + nfft - 1), fade))
        r = np.arange(-n_roll, ncp[i] + nfft + n_roll - 1)
        segment = symbols[(r - ncp[i]) % nfft, i] * window
        np.add.at(x, (starts[i] + r) % starts[-1], segment)
    return x
