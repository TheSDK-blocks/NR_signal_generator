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


def aclr_channels(f, carriers):
    """
    Return the channels of the ACLR measurement of each carrier.

    Every carrier gets its own channel, and the lowest and highest carriers
    the first adjacent channel below and above them, spaced by their channel
    bandwidth (TS 38.104, Section 6.6.3.2). The second adjacent channel is
    added where the spectrum f covers it.

    Parameters
    ----------
    f : array
        Frequencies in Hz in ascending order of the two-sided spectrum, as
        returned by spectrum.psd
    carriers : list of carrier.Carrier
        Carriers of the signal

    Returns
    -------
    List of (carrier index, k, spectrum.Channel), with k = 0 for the
    carrier's own channel and k = -2, -1 or k = 1, 2 for the adjacent
    channels below or above it.
    """
    nyquist = abs(f[0])
    offsets = [c.offset for c in carriers]
    lowest = int(np.argmin(offsets))
    highest = int(np.argmax(offsets))
    channels = []
    for i, c in enumerate(carriers):
        if i == lowest:
            second = spectrum.Channel(c.offset - 2 * c.bw, c.aclr_bw)
            if second.center - second.width / 2 >= -nyquist:
                channels.append((i, -2, second))
            channels.append((i, -1, spectrum.Channel(c.offset - c.bw, c.aclr_bw)))
        channels.append((i, 0, spectrum.Channel(c.offset, c.bw_config)))
        if i == highest:
            channels.append((i, 1, spectrum.Channel(c.offset + c.bw, c.aclr_bw)))
            second = spectrum.Channel(c.offset + 2 * c.bw, c.aclr_bw)
            if second.center + second.width / 2 <= nyquist:
                channels.append((i, 2, second))
    return channels


def aclr(f, p, carriers):
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

    Returns
    -------
    List of dictionaries, one per channel, with the carrier index "carrier",
    "k" and "channel" as in aclr_channels, the power in the carrier's own
    channel relative to the power in the channel "aclr" in dB, positive as in
    TS 38.104, Section 6.6.3.2, and the mean spectral density in the channel
    relative to the highest spectral density "level" in dB.

    Raises
    ------
    ValueError
        If a first adjacent channel extends past the Nyquist frequency.
    """
    nyquist = abs(f[0])
    peak = max(p)
    result = []
    for i, k, channel in aclr_channels(f, carriers):
        low = channel.center - channel.width / 2
        high = channel.center + channel.width / 2
        if low < -nyquist or high > nyquist:
            raise ValueError(
                f"channel {k} of carrier {i} from {low / 1e6:g} to {high / 1e6:g} MHz "
                f"extends past the Nyquist frequency {nyquist / 1e6:g} MHz"
            )
        own = spectrum.Channel(carriers[i].offset, carriers[i].bw_config)
        reference = spectrum.band_power(f, p, own)
        power = spectrum.band_power(f, p, channel)
        density = power / channel.width
        entry = {
            "carrier": i,
            "k": k,
            "channel": channel,
            "aclr": 10 * np.log10(reference / power),
            "level": 10 * np.log10(density / peak),
        }
        result.append(entry)
    return result
