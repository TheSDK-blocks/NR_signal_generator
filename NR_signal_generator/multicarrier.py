import dataclasses

import numpy as np

from . import numerology


def contiguous(carriers, raster, center=0.0):
    """
    Return the carriers placed side by side at the nominal channel spacing of
    intra-band contiguous carrier aggregation (TS 38.104, Section 5.4.1.2),
    with the aggregated channel bandwidth centred at center.

    Parameters
    ----------
    carriers : list of carrier.Carrier
        Carriers in ascending frequency order
    raster : float
        Channel raster of the operating band in Hz, see numerology.nominal_spacing
    center : float
        Centre of the aggregated channel bandwidth in Hz

    Returns
    -------
    List of carriers with their offsets set.
    """
    offsets = [0.0]
    for low, high in zip(carriers, carriers[1:]):
        spacing = numerology.nominal_spacing(low.fr, low.bw, high.bw, raster)
        offsets.append(offsets[-1] + spacing)

    edge_low = offsets[0] - carriers[0].bw / 2
    edge_high = offsets[-1] + carriers[-1].bw / 2
    shift = center - (edge_low + edge_high) / 2
    return [dataclasses.replace(c, offset=o + shift) for c, o in zip(carriers, offsets)]


def plan(carriers, osr):
    """
    Return the sample rate plan of a multi-carrier signal.

    The common sample rate is osr times the smallest multiple of the LCM of
    the carrier sample rates at which the occupied bandwidth, from the lowest
    to the highest transmission bandwidth edge around 0 Hz, is at most 85 %
    of the sample rate, as for a single carrier in numerology.nr_parameters.

    Parameters
    ----------
    carriers : list of carrier.Carrier
        Carriers of the signal
    osr : integer
        Oversampling ratio of the common sample rate

    Returns
    -------
    Dictionary with the common sample rate "Fs", the oversampling ratio "osr"
    of each carrier relative to its own sample rate, and the signal period
    "length" in samples.

    Raises
    ------
    ValueError
        If the carriers differ in duration, if the transmission bandwidths of
        two carriers overlap, or if a carrier offset does not keep the signal
        period cyclic.
    """
    half_subframes = {int(np.round(c.duration / 0.5e-3)) for c in carriers}
    if len(half_subframes) > 1:
        listed = ", ".join(f"{n * 0.5:g}" for n in sorted(half_subframes))
        raise ValueError(f"carriers must have the same duration, got {listed} ms")
    period = half_subframes.pop() * 0.5e-3

    edges = sorted((c.offset - c.bw_config / 2, c.offset + c.bw_config / 2) for c in carriers)
    for (_, high), (low, _) in zip(edges, edges[1:]):
        if low < high:
            raise ValueError(f"carriers overlap between {low / 1e6:g} and {high / 1e6:g} MHz")

    lcm = 1
    half_span = 0.0
    for c in carriers:
        lcm = np.lcm(lcm, int(c.parameters()["Fs"]))
        half_span = max(half_span, abs(c.offset) + c.bw_config / 2)
    fs = osr * lcm * np.ceil(2 * half_span / (0.85 * lcm))

    osr_carrier = np.array([fs / c.parameters()["Fs"] for c in carriers])
    length = int(np.round(period * fs))

    for c in carriers:
        cycles = c.offset * period
        if not np.isclose(cycles, np.round(cycles)):
            raise ValueError(
                f"offset {c.offset / 1e6:g} MHz is not a whole number of cycles over the {period * 1e3:g} ms period"
            )

    return {"Fs": fs, "osr": osr_carrier, "length": length}
