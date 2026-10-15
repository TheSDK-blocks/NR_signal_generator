import matplotlib.pyplot as plt
import numpy as np

from . import spectrum


def plot_constellation(received, reference, evm, ax=None, title=""):
    """
    Plot equalized received data symbols over the reference symbols,
    annotated with the EVM.

    Parameters
    ----------
    received : array
        Equalized received data symbols, as returned by analyzer.analyze
    reference : array
        Reference data symbols, as returned by generator.generate
    evm : float
        EVM as a fraction, as returned by analyzer.analyze
    ax : matplotlib Axes or None
        Axes to plot into, None for a new figure
    title : string
        Axes title

    Returns
    -------
    The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(received.real, received.imag, ".", color="cornflowerblue", markersize=2)
    ax.plot(reference.real, reference.imag, "+", color="darkred", markersize=8, markeredgewidth=1.5)
    # Symmetric limits with a quarter of the largest component as padding
    components = np.concatenate((received.real, received.imag, reference.real, reference.imag))
    limit = 1.25 * np.abs(components).max()
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left")
    ax.set_title(f"EVM {100 * evm:.4f} %", loc="right")
    return ax


def plot_nr_spectrum(f, p, aclr, ax=None, unit="GHz", label=""):
    """
    Plot a power spectral density with spectrum.plot_psd and draw the
    adjacent channels of an ACLR result as bars at their level, labelled with
    their ACLR.

    Parameters
    ----------
    f : array
        Frequencies in Hz, as returned by spectrum.psd
    p : array
        Power spectral density, as returned by spectrum.psd
    aclr : list
        ACLR result, as returned by measurements.aclr
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
    ax = spectrum.plot_psd(f, p, ax, unit, label)
    scale = spectrum.UNITS[unit]
    for entry in aclr:
        if entry["k"] != 0:
            channel = entry["channel"]
            low = (channel.center - channel.width / 2) / scale
            high = (channel.center + channel.width / 2) / scale
            ax.hlines(entry["level"], low, high, colors="r", zorder=10)
            label = str(round(entry["aclr"], 2))
            ax.text(x=low, y=entry["level"] + 3, s=label, fontsize="small", color="r")
    return ax
