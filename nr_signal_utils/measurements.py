import numpy as np

from . import spectrum


def evm(reference, received):
    """
    Return the EVM of equalised received symbols against reference symbols.

        EVM = sqrt(sum(|r - s|^2) / sum(|s|^2))

    Parameters
    ----------
    reference : array
        Reference data symbols s
    received : array
        Equalised received data symbols r, same shape as reference

    Returns
    -------
    EVM as a fraction.
    """
    if np.shape(reference) != np.shape(received):
        raise ValueError(f"reference shape {np.shape(reference)} does not match received shape {np.shape(received)}")
    error = np.sum(np.abs(received - reference) ** 2)
    power = np.sum(np.abs(reference) ** 2)
    return np.sqrt(error / power)


def aclr_channels(f, carriers, aclr_offset=0):
    """
    Return the channels of the ACLR measurement of each carrier.

    Each carrier gets up to two adjacent channels on each side, spaced by its
    channel bandwidth (or aclr_offset). An adjacent channel is kept if a
    channel bandwidth around its centre lies within the spectrum f and clear
    of the channel bandwidths of the other carriers.

    Parameters
    ----------
    f : array
        Frequencies in Hz in ascending order, as returned by spectrum.psd
    carriers : list of carrier.Carrier
        Carriers of the signal
    aclr_offset : float
        Adjacent channel spacing in Hz, 0 for the carrier channel bandwidth

    Returns
    -------
    List of (carrier index, k, spectrum.Channel), with k = 0 for the
    carrier's own channel and k < 0 or k > 0 for the adjacent channels below
    or above it.
    """
    channels = []
    for i, c in enumerate(carriers):
        if aclr_offset == 0:
            spacing = c.bw
        else:
            spacing = aclr_offset
        edges_below = [n.offset + n.bw / 2 for n in carriers if n.offset < c.offset]
        edges_above = [n.offset - n.bw / 2 for n in carriers if n.offset > c.offset]
        low = max([f[0]] + edges_below)
        high = min([f[-1]] + edges_above)
        for k in range(-2, 3):
            center = c.offset + k * spacing
            if k == 0:
                channels.append((i, k, spectrum.Channel(c.offset, c.bw_config)))
            elif low <= center - c.bw / 2 and center + c.bw / 2 <= high:
                channels.append((i, k, spectrum.Channel(center, c.aclr_bw)))
    return channels


def aclr(f, p, carriers, aclr_offset=0):
    """
    Return the adjacent channel leakage of each carrier, see aclr_channels.

    Parameters
    ----------
    f : array
        Frequencies in Hz in ascending order of the two-sided spectrum, as
        returned by spectrum.psd
    p : array
        Power spectral density, as returned by spectrum.psd
    carriers : list of carrier.Carrier
        Carriers of the signal
    aclr_offset : float
        See aclr_channels

    Returns
    -------
    List of dictionaries, one per channel, with the carrier index "carrier",
    "k" and "channel" as in aclr_channels, the level of the channel relative
    to the carrier's own channel "aclr" in dB, and relative to the highest
    spectral density "level" in dB.

    Raises
    ------
    ValueError
        If a channel extends past the Nyquist frequency.
    """
    nyquist = abs(f[0])
    peak = max(p)
    result = []
    for i, k, channel in aclr_channels(f, carriers, aclr_offset):
        low = channel.center - channel.width / 2
        high = channel.center + channel.width / 2
        if low < -nyquist or high > nyquist:
            raise ValueError(
                f"channel {k} of carrier {i} from {low / 1e6:g} to {high / 1e6:g} MHz "
                f"extends past the Nyquist frequency {nyquist / 1e6:g} MHz"
            )
        own = spectrum.Channel(carriers[i].offset, carriers[i].bw_config)
        reference = spectrum.band_level(f, p, own)
        level = spectrum.band_level(f, p, channel)
        entry = {
            "carrier": i,
            "k": k,
            "channel": channel,
            "aclr": 10 * np.log10(level / reference),
            "level": 10 * np.log10(level / peak),
        }
        result.append(entry)
    return result
