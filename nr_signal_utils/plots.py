from . import spectrum


def annotate_aclr(ax, result, unit="GHz"):
    """
    Draw the adjacent channels of an ACLR result as bars at their level,
    labelled with their ACLR.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes of the spectrum, as returned by spectrum.plot_psd
    result : list
        ACLR result, as returned by measurements.aclr
    unit : string ("Hz", "kHz", "MHz", "GHz")
        Frequency axis unit of ax

    Returns
    -------
    The matplotlib Axes.
    """
    scale = spectrum.UNITS[unit]
    for entry in result:
        if entry["k"] != 0:
            channel = entry["channel"]
            low = (channel.center - channel.width / 2) / scale
            high = (channel.center + channel.width / 2) / scale
            ax.hlines(entry["level"], low, high, colors="r", zorder=10)
            label = str(round(entry["aclr"], 2))
            ax.text(x=low, y=entry["level"] + 3, s=label, fontsize="small", color="r")
    return ax
