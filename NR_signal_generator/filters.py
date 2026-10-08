import warnings

import numpy as np
import scipy.signal as sig

from . import numerology


def nr_filter(raw_vector, Fs, BW, fr):
    """
    Return raw_vector low-pass filtered to the channel bandwidth BW, and the
    filter order.

    Parameters
    ----------
    raw_vector : array
        Signal to be filtered
    Fs : integer
        Sampling frequency
    BW : float
        Channel bandwidth
    fr : string ("FR1", "FR2-1")
        Frequency range

    Returns
    -------
    Filtered signal of len(raw_vector) + order samples, and the filter order.
    """
    # get sampling frequency

    Fs = int(Fs)
    # select filter parameters
    # fac=1
    # fac=BW/150e6
    # order=self.fil_len*np.ceil(osr*fac) # filter order

    dF = 0
    for mu in range(0, 5):
        try:
            up = numerology.nr_parameters(fr, 4 - mu, BW)
            if ((up["RB"]) * 12 * 15e3 * 2 ** (4 - mu)) > dF:
                dF = (up["RB"]) * 12 * 15e3 * 2 ** (4 - mu)
        except:
            continue
    dF = (BW - dF) * 0.5

    Fc = BW / 2  # center of transition bandwidth
    t_vect = np.arange(0, len(raw_vector)) / Fs
    # filter NR signal with circular convolution
    # order=200

    ripple_dB = -1
    att_dB = -80
    # ripple=ripple_dB
    # att=att_dB
    ripple = 10 ** (ripple_dB / 20)
    att = 10 ** (att_dB / 20)
    D = (
        0.005309 * (np.log10(ripple)) ** 2 + 0.07114 * (np.log10(ripple)) - 0.4761
    ) * (np.log10(att)) - (
        0.00266 * (np.log10(ripple)) ** 2 + 0.5941 * (np.log10(ripple)) + 0.4278
    )
    f = 11.012 + 0.51244 * (np.log10(ripple) - np.log10(att))
    ord_approx = (D - f * (dF / Fs) ** 2) / (dF / Fs) + 1  # Herrmann
    ord_approx2 = (-20 * np.log10(np.sqrt(ripple * att)) - 13) / (
        14.6 * dF / Fs
    ) + 1

    mid = (ord_approx + ord_approx2) / 2
    N_min = max(20, int(mid - 100))
    N_max = int(mid + 100)
    # print(ord_approx)
    # print(ord_approx2)
    sb_max_dB = -80
    # if int(ord_approx)%2!=0:
    #    ord_approx+=1
    # if int(ord_approx2)%2!=0:
    #    ord_approx2+=1
    test = N_min
    sb_best = 1
    att_freq = Fc + dF
    if att_freq > Fs / 2:
        att_freq = Fc

    sb_max_dB_list = [i for i in range(0, abs(sb_max_dB - 10), 20)]
    sb_max_dB_list.reverse()
    sb_max_list = [
        10 ** (-sb_max_dB_list[i] / 20) for i in range(len(sb_max_dB_list))
    ]
    done = False

    # python doesn't allow you to change the for loop iterator mid-loop so let's make a 1D-loop
    allindices = []
    for sb_max in sb_max_list:
        for ord in range(N_min, N_max):
            allindices.append([sb_max, ord])

    skipcount = 32  # skip n iterations during the broad-phase
    skipindex = 0

    for currentphase in range(2):  # broad-phase first and then narrow-phase
        for index in allindices:
            sb_max, ord = index

            # broad phase iteration (when skipcount != 0)
            if skipindex > 0:
                skipindex -= 1
                continue
            skipindex = skipcount

            try:
                b = sig.remez(
                    int(ord + 1),
                    [0, Fc - dF, att_freq, 0.5 * Fs],
                    [1, 0],
                    fs=int(Fs),
                )
                # order=ord
                # plt.figure()
                # plt.plot(b)
                # plt.show(block=False)
                w, h = sig.freqz(b, [1], worN=2000, fs=Fs)
                idx = np.argwhere(w > ((att_freq)))[0][0]
                sb = max(np.abs(h[idx:]))
                if sb < sb_best:
                    sb_best = sb
                if sb <= sb_max:
                    order = ord
                    break
            except Exception as e:
                warnings.warn(f"{e}")
        else:
            continue
        break

        # go back n iterations and try again with the narrow-phase
        if currentphase == 0:
            endindex = allindices.index(index)
            startindex = endindex - skipcount
            if startindex < 0:
                startindex = 0
            allindices = allindices[startindex : endindex + 1]
            skipcount = 0
            skipindex = 0

    bb = b
    # bb=np.pad(b,(0,len(raw_vector)-len(b)),constant_values=0)
    # s_fil=np.fft.ifft(np.fft.fft(raw_vector,len(raw_vector))*np.fft.fft(bb,len(raw_vector)))
    s_fil = np.convolve(raw_vector.reshape((-1, 1))[:, 0], bb, mode="full").reshape(
        (-1, 1)
    )
    # compensate group delay (order/2)
    s_out = np.concatenate((s_fil[int(order / 2) :], s_fil[0 : int(order / 2)]))
    # s_out=s_fil
    return s_out[:, 0], order
