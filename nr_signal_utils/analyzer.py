import numpy as np

from . import constellation
from . import equalizer
from . import filters
from . import grid
from . import measurements
from . import multicarrier
from . import ofdm
from . import sync


def demodulate_carrier(c, x, osr):
    """
    Return the equalized data symbols of one carrier and the bits they carry.

    Parameters
    ----------
    c : carrier.Carrier
        Carrier
    x : array
        One period of the carrier signal at baseband
    osr : integer
        Oversampling ratio relative to the carrier sample rate

    Returns
    -------
    Equalized data symbols in mapping order, and the demapped bits.
    """
    dl = c.parameters(osr)
    ncp = c.cp_lengths(osr)
    starts = ofdm.symbol_starts(ncp, dl["NFFT"])
    l_pss = grid.pss_symbol(c.n_symb)
    aligned = sync.pss_align(x, dl["NFFT"], starts[l_pss] + ncp[l_pss], c.n_id_2)
    re_grid = ofdm.demodulate(aligned, dl["NFFT"], ncp, dl["Nsc"])

    values, is_ref = c.reference_grid
    re_ref = np.ones(re_grid.shape, complex)
    re_ref[is_ref] = values[is_ref]
    equalized = equalizer.equalize(re_grid, re_ref, c.pilot_mask)

    symbols = grid.data_symbols(equalized, c.data_mask)
    return symbols, constellation.demodulate(symbols, c.qam)


def analyze(y, ref_symbols, carriers, osr=1, rx_filter=True):
    """
    Return the equalized data symbols, bits and EVM of each carrier of a
    multi-carrier NR downlink signal.

    y must be exactly one period of a cyclic signal, as produced by
    generator.generate: the receive filter, the PSS alignment and the FFT
    windows all treat it as periodic. Any period of a steady-state cyclic
    signal works, wherever it starts. Non-cyclic signals, such as arbitrary
    captures, are not supported.

    Parameters
    ----------
    y : array
        Complex signal at the sample rate multicarrier.plan(carriers, osr)["Fs"]
    ref_symbols : list
        Transmitted data symbols of each carrier, as returned by generator.generate
    carriers : list of carrier.Carrier
        Carriers of the signal
    osr : integer
        Oversampling ratio of the common sample rate
    rx_filter : bool
        Apply the receive channel-select filter to each carrier

    Returns
    -------
    Lists of the equalized data symbols, the demapped bits and the EVM of each
    carrier.
    """
    plan = multicarrier.plan(carriers, osr)
    if len(y) != plan["length"]:
        raise ValueError(f"signal of {len(y)} samples is not one period of {plan['length']} samples")
    if len(ref_symbols) != len(carriers):
        raise ValueError(f"{len(ref_symbols)} reference symbol arrays for {len(carriers)} carriers")

    fs = plan["Fs"]
    symbols = []
    bits = []
    evm = []
    for c, osr_c, ref in zip(carriers, plan["osr"], ref_symbols):
        x = multicarrier.mix(y, fs, -c.offset)
        if rx_filter:
            x = filters.nr_filter(x, fs, c.bw, c.fr)
        s, b = demodulate_carrier(c, x, osr_c)
        symbols.append(s)
        bits.append(b)
        evm.append(measurements.evm(ref, s))
    return symbols, bits, evm
