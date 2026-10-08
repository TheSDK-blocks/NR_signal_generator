import numpy as np


def equalize(RE, RE_id, start, stop):
    """
    Return the zero-forcing equalized resource-element grid (TS 38.104, Annex B.6).

    The equalizer coefficients are estimated from the complex ratios of the
    received and ideal grids at the DMRS subcarriers: phase unwrapped across
    DMRS symbols, amplitude and phase averaged over time, smoothed across
    frequency with a moving average of up to 19 DMRS subcarriers, and linearly
    interpolated to every subcarrier.

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
    Nsc, N_symb_TOT = RE.shape
    if N_symb_TOT < 3:
        raise ValueError(
            f"no DMRS symbol: equalization needs at least 3 OFDM symbols, got {N_symb_TOT}"
        )

    # calculate the complex ratios of the post-FFT acquired signal "RE" and the
    # post-FFT ideal signal "RE_id", for each reference symbol
    complex_ratios = np.divide(RE, RE_id, out=np.zeros_like(RE), where=RE_id != 0)
    a = np.absolute(complex_ratios)
    phi = np.angle(complex_ratios)
    #  unwrap phase of complex ratios at symbol #0 of each slot
    l = np.arange(0, N_symb_TOT, dtype=int)
    l = l[l % 14 == 2]
    k1 = np.arange(start, stop, 2)
    for k in k1:
        for i in np.arange(1, l.size):
            delta_phi = phi[k, int(l[i])] - phi[k, int(l[i - 1])]
            if np.absolute(delta_phi) >= np.pi:
                phi[k, l[i:]] = phi[k, l[i:]] - 2 * np.pi * np.sign(delta_phi)

    # perform time averaging at each reference signal subcarrier of the complex
    # ratios (in TS 36.104 the time-averaging length is 10 subframes, here for
    # simplicity the time-averaging length is that of the signal)
    a_avg = np.zeros((int(Nsc)))
    phi_avg = np.zeros((int(Nsc)))

    l = np.arange(0, N_symb_TOT, dtype=int)
    l = l[l % 14 == 2]
    k1 = np.arange(start, stop, 2)

    for k in k1:
        test = a[int(k), l]
        a_avg[int(k)] = np.mean(a[int(k), l])
        phi_avg[int(k)] = np.mean(phi[int(k), l])

    # the equalizer coefficients for amplitude and phase "a_coeff" and
    # "phi_coeff" at the reference signal subcarriers are obtained by computing
    # the moving average in the frequency domain of the time-averaged reference
    # signal subcarriers, i.e. every third subcarrier (or sixth, if less than 5
    # OFDM symbols)
    a_coeff = np.zeros((int(Nsc)))
    phi_coeff = np.zeros((int(Nsc)))
    k = np.arange(start, stop, 2)

    for i in np.arange(1, k.size + 1):
        m_avg_w_length = min(2 * i - 1, 2 * (k.size - i) + 1, 19)
        m_avg_imp_resp = np.ones(m_avg_w_length) / m_avg_w_length
        test = np.dot(
            m_avg_imp_resp,
            a_avg[
                k[
                    int(i - np.floor(m_avg_w_length / 2) - 1) : int(
                        i + np.floor(m_avg_w_length / 2)
                    )
                ]
            ],
        )
        a_coeff[k[int(i - 1)]] = np.dot(
            m_avg_imp_resp,
            a_avg[
                k[
                    int(i - np.floor(m_avg_w_length / 2) - 1) : int(
                        i + np.floor(m_avg_w_length / 2)
                    )
                ]
            ],
        )
        phi_coeff[k[int(i - 1)]] = np.dot(
            m_avg_imp_resp,
            phi_avg[
                k[
                    int(i - np.floor(m_avg_w_length / 2) - 1) : int(
                        i + np.floor(m_avg_w_length / 2)
                    )
                ]
            ],
        )

    # perform linear interpolation to compute coefficients for each subcarrier
    a_coeff = np.interp(np.arange(1, Nsc + 1), k + 1, a_coeff[k])
    phi_coeff = np.interp(np.arange(1, Nsc + 1), k + 1, phi_coeff[k])

    cnstl = np.zeros((RE.shape), complex)
    # equalize resource elements and return
    for i in np.arange(0, Nsc):
        cnstl[i, :] = RE[i, :] / (a_coeff[i] * np.exp(1j * phi_coeff[i]))

    return cnstl
