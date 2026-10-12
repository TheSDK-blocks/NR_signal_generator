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

    One carrier gets up to two adjacent channels on each side, spaced by its
    channel bandwidth (or aclr_offset), as far as the spectrum f covers them.
    Several carriers get up to two adjacent channels each, limited by the
    neighbouring carriers.

    Parameters
    ----------
    f : array
        Frequencies in Hz in ascending order, as returned by spectrum.psd
    carriers : list of carrier.Carrier
        Carriers of the signal
    aclr_offset : float
        Adjacent channel spacing (one carrier) or centre (several carriers)
        in Hz, 0 for the carrier channel bandwidth and offset

    Returns
    -------
    List of (carrier index, k, spectrum.Channel), with k = 0 for the
    carrier's own channel and k < 0 or k > 0 for the adjacent channels below
    or above it.
    """
    channels = []
    if len(carriers) == 1:
        c = carriers[0]
        n = (max(f) - min(f)) / c.bw
        if n >= 5 and max(f) > c.offset + 2.5 * c.bw and min(f) < c.offset - 2.5 * c.bw:
            n = 5
        elif n >= 3 and max(f) >= c.offset + 1.5 * c.bw and min(f) <= c.offset - 1.5 * c.bw:
            n = 3
        else:
            n = 0
        half = int(np.floor(n / 2))
        for k in range(-half, half + 1):
            if k == 0:
                channels.append((0, k, spectrum.Channel(c.offset, c.bw_config)))
            elif aclr_offset == 0:
                channels.append((0, k, spectrum.Channel(c.offset + k * c.bw, c.aclr_bw)))
            else:
                channels.append((0, k, spectrum.Channel(c.offset + k * aclr_offset, c.aclr_bw)))
        return channels

    for i, c in enumerate(carriers):
        lf = np.abs(f - (c.offset - 0.5 * c.bw_config)).argmin()
        hf = np.abs(f - (c.offset + 0.5 * c.bw_config)).argmin()
        if i == 0:
            min_f = min(f)
        else:
            min_f = hf + carriers[i - 1].bw
        lf = f[lf]
        hf = f[hf]
        left = (lf - min_f) / c.bw
        if left >= 2 and min_f <= lf - 2.5 * c.bw:
            left = 2
        elif left >= 1 and min_f <= lf - 1.5 * c.bw:
            left = 1
        else:
            left = 0
        if i < len(carriers) - 1:
            max_f = hf
        else:
            max_f = max(f)
        right = (max_f - hf) / c.bw
        if right >= 2 and max_f >= hf - 2.5 * c.bw:
            right = 2
        elif right >= 1 and max_f >= hf - 1.5 * c.bw:
            right = 1
        else:
            right = 0
        for k in range(-int(left), int(right + 1)):
            if k == 0:
                channels.append((i, k, spectrum.Channel(c.offset, c.bw_config)))
            elif aclr_offset == 0:
                channels.append((i, k, spectrum.Channel(c.offset + k * c.bw, c.aclr_bw)))
            else:
                channels.append((i, k, spectrum.Channel(aclr_offset + k * c.bw, c.aclr_bw)))
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
