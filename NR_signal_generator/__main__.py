import pdb
import numpy as np
import matplotlib.pyplot as plt

from plot_PSD import plot_PSD

from . import NR_signal_generator
from .carrier import Carrier

carriers = [Carrier("FR1", 100e6, 1, 0.5e-3, "64QAM")]
in_bits = ["max"] * len(carriers)
osr = 1

test = NR_signal_generator()
test.carriers = carriers
test.osr = osr
test.in_bits = in_bits
test.run_gen()
rand = np.random.rand(1000, 2) / 10000

test.IOS.Members["in_dem"].Data = np.vstack([rand, test.IOS.Members["out"].Data])
test.run_dem()
test.run_EVM()

plt.figure()
plt.plot(test.s_struct["s"][:, 0], test.s_struct["s"][:, 1])
plt.plot(test.s_struct["s"][:, 0], test.s_struct["s"][:, 2])
plt.show(block=False)
for i in range(0, len(carriers)):
    print(f"Carrier {i}: EVM {100 * test.EVM[i]:.4f} %")
for i in range(0, len(carriers)):
    plt.figure()
    plt.plot(test.dem_cnstl_vec[i].real, test.dem_cnstl_vec[i].imag, "o")
    plt.plot(test.cnstl[i].real, test.cnstl[i].imag, "o")
    plt.title("Carrier " + str(i + 1))
    plt.show(block=False)

a = plot_PSD(signal=test.s_struct["s"][:, 1], Fs=test.s_struct["Fs"])

pdb.set_trace()
