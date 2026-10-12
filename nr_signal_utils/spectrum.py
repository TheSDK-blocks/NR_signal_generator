from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

# Frequency axis scale per unit
UNITS = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}


@dataclass(frozen=True)
class Channel:
    """
    Frequency band of a channel.

    Parameters
    ----------
    center : float
        Centre frequency in Hz
    width : float
        Bandwidth in Hz
    """

    center: float
    width: float


def band_level(f, p, channel):
    """
    Return the mean power spectral density between the frequency bins nearest
    to the band edges of a channel.

    Parameters
    ----------
    f : array
        Frequencies in Hz in ascending order, as returned by psd
    p : array
        Power spectral density, as returned by psd
    channel : Channel
        Channel

    Returns
    -------
    Mean power spectral density.
    """
    low = np.abs(f - (channel.center - channel.width / 2)).argmin()
    high = np.abs(f - (channel.center + channel.width / 2)).argmin()
    return np.mean(p[low:high])


def psd(x, fs):
    """
    Return the two-sided power spectral density of a signal, Welch averaged
    over segments of a tenth of the signal with a Hann window and 50 % overlap.

    Parameters
    ----------
    x : array
        Signal
    fs : float
        Sample rate in Hz

    Returns
    -------
    Frequencies in Hz in ascending order, and the power spectral density in
    |x|^2 / Hz.
    """
    nperseg = round(len(x) * 10 / 100)
    noverlap = round(nperseg * 50 / 100)
    window = sig.windows.tukey(nperseg, 1)
    f, p = sig.welch(x, fs, window, nperseg, noverlap=noverlap, detrend=False, return_onesided=False)
    order = f.argsort()
    return f[order], p[order]


def plot_psd(f, p, ax=None, unit="GHz", label=""):
    """
    Plot a power spectral density in dB relative to its highest value.

    Parameters
    ----------
    f : array
        Frequencies in Hz, as returned by psd
    p : array
        Power spectral density, as returned by psd
    ax : matplotlib Axes or None
        Axes to plot into, None for a new figure
    unit : string ("Hz", "kHz", "MHz", "GHz")
        Frequency axis unit
    label : string
        Legend label

    Returns
    -------
    The matplotlib Axes.
    """
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {', '.join(UNITS)}, got {unit}")
    if ax is None:
        _, ax = plt.subplots()
    level = 10 * np.log10(p / max(p))
    ax.plot(f / UNITS[unit], level, label=str(label))
    ax.grid(visible=True)
    ax.legend()
    ax.set_title("Signal spectrum")
    ax.set_xlabel(f"Frequency [{unit}]")
    ax.set_ylabel("PSD [dB]")
    return ax
