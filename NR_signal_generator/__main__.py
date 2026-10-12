import matplotlib.pyplot as plt

from . import NR_signal_generator
from nr_signal_utils.carrier import Carrier
from nr_signal_utils.multicarrier import contiguous
from nr_signal_utils.plots import plot_constellation
from nr_signal_utils.plots import plot_nr_spectrum


def single_carrier():
    """Return one 100 MHz 64QAM carrier."""
    return [Carrier("FR1", 100e6, 1, 0.5e-3, "64QAM")]


def multi_carrier():
    """Return three contiguous carriers at different powers."""
    carriers = [
        Carrier("FR1", 20e6, 1, 0.5e-3, "16QAM", power_dbm=-20.0),
        Carrier("FR1", 100e6, 1, 0.5e-3, "256QAM", power_dbm=0.0),
        Carrier("FR1", 40e6, 1, 0.5e-3, "64QAM", power_dbm=-10.0),
    ]
    return contiguous(carriers, raster=100e3)


def run(carriers, osr, aclr=False):
    """Generate and analyze the carriers, print the EVM and plot the results, with the ACLR if aclr."""
    test = NR_signal_generator()
    test.carriers = carriers
    test.osr = osr
    test.in_bits = ["max"] * len(carriers)
    test.tx_filter = False
    test.run_gen()

    test.IOS.Members["in_dem"].Data = test.IOS.Members["out"].Data
    test.run_dem()

    for i, c in enumerate(carriers):
        print(f"Carrier {i}: {c.bw / 1e6:g} MHz {c.qam} at {c.offset / 1e6:g} MHz, {c.power_dbm:g} dBm, EVM {100 * test.EVM[i]:.4f} %")

    plt.figure()
    plt.plot(test.s_struct["s"][:, 0], test.s_struct["s"][:, 1])
    plt.plot(test.s_struct["s"][:, 0], test.s_struct["s"][:, 2])
    plt.show(block=False)
    for i in range(0, len(carriers)):
        plot_constellation(test.dem_cnstl_vec[i], test.cnstl[i], test.EVM[i], title="Carrier " + str(i + 1))
        plt.show(block=False)

    aclr_result = []
    if aclr:
        aclr_result = test.ACLR
    ax = plot_nr_spectrum(*test.psd, aclr_result)
    ax.set_ylim(-120, 10)
    plt.show()


def main():
    demo = "single"  # "single" or "multi"
    if demo == "single":
        run(single_carrier(), osr=5, aclr=True)
    elif demo == "multi":
        run(multi_carrier(), osr=1)
    else:
        raise ValueError(f'demo must be "single" or "multi", got {demo}')


if __name__ == "__main__":
    main()
