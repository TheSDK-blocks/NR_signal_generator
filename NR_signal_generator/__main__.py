import pdb
import numpy as np
import matplotlib.pyplot as plt

from plot_PSD import plot_PSD

from . import NR_signal_generator

# length=1024
# method="multi"
# method="single"
# BWP=np.array([[1,7,0,273],[1,7,0,273]])  # [mu, symbols, N_BWP^start, N_BWP^size] per carrier
BWP = np.array([[1, 7, 0, 273]])
# BW=np.array([400e6,400e6])
BW = np.array([100e6])
# mu=[0,0]
# QAM="16QAM"
QAM = "64QAM"
# QAM="256QAM"
# QAM="4QAM"
# QAM="BPSK" # Not for downlink
# bits=np.array([0,0,0,1]*3816)

# bits=np.array(list(bits)*(int(240/4))) # same as ones in matlab nsymb 2 16qam
# bits=np.array(list(bits)*(int(6000)))
# bits=np.random.randint(2,size=1028*2)
# bits=np.random.randint(2,size=1710*6)
# bits2=np.random.randint(2,size=4752*6)
# bits3=np.random.randint(2,size=3738*6)
# bits=np.random.randint(2,size=1710*4)
# in_bits=[bits]*len(BW)
# in_bits =np.array([bits2])
in_bits = np.array(["max"])
# in_bits=np.array(["max", "max"])
osr = 1
Fc = 0  # 1e9
if not hasattr(BW, "__len__"):
    a = 5
elif hasattr(BW, "__len__"):
    test = NR_signal_generator()
    # test.IOS.Members['in_dem']=test.IOS.Members['out']
    test.BW = BW
    test.BWP = BWP
    test.QAM = QAM
    test.osr = osr
    test.Fc_gen = Fc
    # test.include_time_vector=1
    test.in_bits = in_bits
    test.run_gen()
    rand = np.random.rand(1000, 2) / 10000

    test.IOS.Members["in_dem"].Data = np.vstack(
        [rand, test.IOS.Members["out"].Data]
    )
    test.run_dem()
    test.run_EVM()
    a = np.transpose(
        np.vstack((test.s_struct["s"][:, 0], test.s_struct["s"][:, 1]))
    )
    # np.savetxt('signal.csv',test.s_struct["s"],delimiter=',')

    plt.figure()
    plt.plot(test.s_struct["s"][:, 0], test.s_struct["s"][:, 1])
    plt.plot(test.s_struct["s"][:, 0], test.s_struct["s"][:, 2])
    plt.show(block=False)
    print(test.EVM)
    for i in range(0, BW.size):
        if np.size(test.EVM[i]) and test.EVM[i] != 0:
            plt.figure()
            plt.plot(test.rxDataSymbols[i].real, test.rxDataSymbols[i].imag, "o")
            plt.plot(test.cnstl[i].real, test.cnstl[i].imag, "o")
            plt.title("Carrier " + str(i + 1))
            plt.show(block=False)

# meas(test,test.s_struct["Fs"])
a = plot_PSD(
    signal=test.s_struct["s"][:, 1],
    # 100,
    Fs=test.s_struct["Fs"]
)

pdb.set_trace()