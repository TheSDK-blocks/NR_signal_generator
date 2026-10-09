import numpy as np

from . import constellation
from . import filters
from . import grid
from . import multicarrier
from . import ofdm


def data_symbols(carriers, bits=None, seed=0):
    """
    Return the data symbols of each carrier.

    Parameters
    ----------
    carriers : list of carrier.Carrier
        Carriers of the signal
    bits : list or None
        Bits of each carrier, exactly filling its data resource elements
        (Carrier.n_bits), or None for pseudorandom bits. Pseudorandom bits stand
        in for MAC padding and scrambling and are seeded from seed and the
        carrier index, so they are reproducible. User data shorter than the
        allocation must be padded by the caller, e.g. with pseudorandom bits.
        A None entry gives pseudorandom bits for that carrier only.
    seed : integer
        Seed of the pseudorandom bits

    Returns
    -------
    List of the data symbols of each carrier, in mapping order.
    """
    if bits is None:
        bits = [None] * len(carriers)
    if len(bits) != len(carriers):
        raise ValueError(f"{len(bits)} bit arrays for {len(carriers)} carriers")

    symbols = []
    for i, (c, b) in enumerate(zip(carriers, bits)):
        if b is None:
            rng = np.random.RandomState((i + seed * 13 + 1) * 123)
            b = rng.randint(2, size=c.n_bits)
        elif len(b) != c.n_bits:
            raise ValueError(f"carrier {i}: {len(b)} bits do not fill the {c.n_bits}-bit allocation")
        symbols.append(constellation.modulate(b, c.qam))
    return symbols


def modulate_carrier(c, symbols, osr):
    """
    Return the OFDM signal of one carrier at baseband.

    Parameters
    ----------
    c : carrier.Carrier
        Carrier
    symbols : array
        Data symbols of the carrier, in mapping order
    osr : integer
        Oversampling ratio relative to the carrier sample rate

    Returns
    -------
    Complex baseband signal of one period.
    """
    dl = c.parameters(osr)
    values, is_ref = c.reference_grid
    re_grid = grid.map_data(values, c.data_mask, symbols)
    return ofdm.modulate(re_grid, dl["NFFT"], c.cp_lengths(osr), dl["Lroll"] * osr)


def generate(carriers, osr=1, tx_filter=True, bits=None, seed=0):
    """
    Return one period of a cyclic multi-carrier NR downlink signal and its
    data symbols.

    The samples are the peak envelope x of the RF signal Re{x exp(j 2 pi f t)}
    in volts across 50 ohm; each carrier has the mean power Carrier.power_dbm.
    The sample rate is multicarrier.plan(carriers, osr)["Fs"].

    Parameters
    ----------
    carriers : list of carrier.Carrier
        Carriers of the signal
    osr : integer
        Oversampling ratio of the common sample rate
    tx_filter : bool
        Apply the transmit channel filter to each carrier
    bits : list or None
        Bits of each carrier, see data_symbols
    seed : integer
        Seed of the pseudorandom bits, see data_symbols

    Returns
    -------
    Complex signal, and the list of the data symbols of each carrier.
    """
    symbols = data_symbols(carriers, bits, seed)
    plan = multicarrier.plan(carriers, osr)
    fs = plan["Fs"]

    x = np.zeros(plan["length"], complex)
    for c, osr_c, s in zip(carriers, plan["osr"], symbols):
        y = modulate_carrier(c, s, osr_c)
        if tx_filter:
            y = filters.nr_filter(y, fs, c.bw, c.fr)
        y = y * np.sqrt(c.mean_square / np.mean(np.abs(y) ** 2))
        x = x + multicarrier.mix(y, fs, c.offset)
    return x, symbols
