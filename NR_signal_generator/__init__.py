# -------------------------------------------------------------------------------
# --Copyright (c) 2017  Aalto University
# --
# --Permission is hereby granted, free of charge, to any person obtaining a copy
# --of this software and associated documentation files (the "Software"), to deal
# --in the Software without restriction, including without limitation the rights
# --to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# --copies of the Software, and to permit persons to whom the Software is
# --furnished to do so, subject to the following conditions:
# --
# --The above copyright notice and this permission notice shall be included in all
# --copies or substantial portions of the Software.
# --
# --THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# --IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# --FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# --AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# --LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# --OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# --SOFTWARE.
# -------------------------------------------------------------------------------

# Structure, functions and almost all comments are copied from Matlab code done by Enrico Roverato
import os
import sys

if not (os.path.abspath("../../thesdk") in sys.path):
    sys.path.append(os.path.abspath("../../thesdk"))

from thesdk import *

import pdb
import numpy as np
import scipy.signal as sig
from scipy import interpolate as inter
import matplotlib.pyplot as plt
from plot_PSD import plot_PSD

from . import constellation
from . import grid
from . import numerology
from . import ofdm
from . import sequences


class NR_signal_generator(thesdk):  # rtl,eldo,thesdk
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/" + __name__

    def __init__(self, *arg):  # ,BW,osr,Nsymb,qam_type,bits

        self.print_log(type="I", msg="Inititalizing %s" % (__name__))

        # Properties that can be propagated from parent
        self.proplist = [
            "signal_id",
            "BW",
            "BWP",
            "osr",
            "QAM",
            "in_bits",
            "include_time_vector",
            "Fc_gen",
        ]

        self.IOS = Bundle()
        self.IOS.Members["in_dem"] = IO()  # Pointer for input data

        self.IOS.Members["out"] = IO()  # Pointer for output data

        self.BWP = np.array([[[4, 7, 0, 1]]])
        self.QAM = "64QAM"
        self.osr = 1
        self.BW = np.array([200e6])
        self.in_bits = np.array([["max"]])

        self.seed = 0
        self.include_time_vector = 0
        self.Fc_gen = 0

        self.BW_conf = []
        self.ACLR_BW = []
        self.model = "py"
        # Can be set externally, but is not propagated
        self.par = False  # By default, no parallel processingi
        self.queue = []  # By default, no parallel processing
        self.IOS.Members["control_write"] = IO()
        self.fil = "on"
        self.equalizer = "on"

        self.fil_len = 0  # 100
        self.norm = (
            "max"  # max = normalize I & Q separately, amp = normalize amplitude to one
        )

        self.signal_id = 0  # used for generating different seeds for different signals

        if len(arg) >= 1:
            parent = arg[0]
            self.copy_propval(parent, self.proplist)
            self.parent = parent

    def main_gen(self):
        if not hasattr(self.BW, "__len__"):
            self.BW = np.array([self.BW])
        if not hasattr(self.BW, "size"):
            self.BW = np.array([self.BW])

        self.cnstl, self.gen_bits = self.genMultiQAM()
        if self.Fc_gen != 0:
            self.osr_based_on_Fc()
        self.s_struct, self.f_off = self.genMultiNRdownlink()
        if self.include_time_vector == 1:
            self.IOS.Members["out"].Data = self.s_struct[
                "s"
            ]  # output signal, matrix with  columns time, I signal, Q signal
        else:
            self.IOS.Members["out"].Data = np.transpose(
                np.vstack((self.s_struct["s"][:, 1], self.s_struct["s"][:, 2]))
            )
        if len(self.BW_conf) == 1:
            self.BW_conf = self.BW_conf[0]

        if len(self.ACLR_BW) == 1:
            self.ACLR_BW = self.ACLR_BW[0]
        elif len(self.ACLR_BW) == 0:
            self.ACLR_BW = 0

    def run_gen(self, *arg):
        if self.model == "py":
            self.main_gen()

    def main_dem(self, **kwargs):
        car_return = kwargs.get("car_return", False)
        sig_offset = kwargs.get("sig_offset", 0)
        NR_car_id = kwargs.get("NR_car_id", -1)
        self.rec_sig = self.IOS.Members[
            "in_dem"
        ].Data  # Input signal as matrix descibed in main_gen()
        if not hasattr(self.BW, "__len__"):
            self.BW = [self.BW]
        self.dem = self.demMultiNRdownlink(
            car_return=car_return, sig_offset=sig_offset, NR_car_id=NR_car_id
        )
        self.dem_bits, self.dem_cnstl_vec = self.MultiQAMtoBit(NR_car_id=NR_car_id)

    def run_EVM(self, **kwargs):
        NR_car_id = kwargs.get("NR_car_id", -1)
        if self.model == "py":
            self.main_EVM(NR_car_id=NR_car_id)

    def main_EVM(self, **kwargs):
        NR_car_id = kwargs.get("NR_car_id", -1)
        self.EVM, self.rxDataSymbols = self.measMultiEVMdownlink(NR_car_id=NR_car_id)

    def run_dem(self, *arg, **kwargs):
        car_return = kwargs.get(
            "car_return", False
        )  # return the carriers back to their original spots
        sig_offset = kwargs.get("sig_offset", 0)
        NR_car_id = kwargs.get("NR_car_id", -1)
        if self.model == "py":
            self.main_dem(
                car_return=car_return, sig_offset=sig_offset, NR_car_id=NR_car_id
            )

    def osr_based_on_Fc(self):
        BW_vect_abs = np.abs(self.BW)
        BW_tot = np.sum(BW_vect_abs)  # get total bandwidth (in Hz)
        Fs_estimate = np.ceil(BW_tot / 2)
        req_fs = self.Fc_gen * 2 + Fs_estimate
        osr = np.ceil(req_fs / Fs_estimate)
        if osr == 0:
            pass
        else:
            self.osr = osr

    def MultiQAMtoBit(self, **kwargs):
        """Method for calculate binary data based on recieved constellation points for multiple carriers.


        Example
        -------
        self.MultiQAMtoBit()

        """
        NR_car_id = kwargs.get("NR_car_id", -1)

        qam_type = self.QAM
        dem = self.dem
        BW = self.BW
        BWP = self.BWP
        bits = []
        dem_cnstl_vec = []
        for i in range(0, len(dem)):
            if NR_car_id == -1 or i == NR_car_id:
                temp1, temp2 = self.QAMtoBit(cnstl=dem[i], BW=BW[i], BWP=BWP[i])
                bits.append(temp1)
                dem_cnstl_vec.append(temp2)
            else:
                bits.append(np.array([]))
                dem_cnstl_vec.append(np.array([]))

        return bits, dem_cnstl_vec

    def measMultiEVMdownlink(self, **kwargs):
        """Method for calculating EVM based on generated constellation points and recieved constellation points for multiple carriers.

        Example
        -------
        self.measMultiEVMdownlink()

        """

        NR_car_id = kwargs.get("NR_car_id", -1)

        BW = self.BW
        BWP = self.BWP
        dem = self.dem
        cnstl = self.s_struct["cnstl"]

        N_BW = len(cnstl)
        EVM = []
        rxDataSymbols = []
        # measure EVM separately for each constellation
        for i in range(0, N_BW):
            if NR_car_id == -1 or i == NR_car_id:
                N_BWP = len(cnstl[i])
                EVM_BWP = np.zeros(N_BWP)
                rxDataSymbols_BWP = []
                for j in range(0, N_BWP):
                    if not cnstl[i][j].size == 0:
                        EVM1, rxDataSymbols1 = self.measEVMdownlink(
                            BW=BW[i], BWP=BWP[i][j], cnstl=cnstl[i][j], dem=dem[i][j]
                        )
                        EVM_BWP[j] = EVM1
                        rxDataSymbols_BWP.append(rxDataSymbols1)

                    else:
                        rxDataSymbols_BWP.append([])
                rxDataSymbols.append(rxDataSymbols_BWP)
                EVM.append(EVM_BWP)
            else:
                rxDataSymbols.append(np.array([]))
                EVM.append(np.array([]))
        return EVM, rxDataSymbols

    def demMultiNRdownlink(self, **kwargs):
        """Method for demodulate constellation points from recieved signal for multiple carriers.

        Example
        -------
        self.demMultiNRdownlink()

        """
        # return the carriers back to their original spots. Use this to find which carrier is at DC
        car_return = kwargs.get("car_return", False)
        sig_offset = kwargs.get("sig_offset", 0)
        NR_car_id = kwargs.get("NR_car_id", -1)

        sign = self.rec_sig
        BW = self.BW
        BWP = self.BWP
        Fs = self.s_struct["Fs"]
        f_off = self.f_off
        FFRwpos = 0
        N_BW = BW.size
        cnstl = []
        # t=np.arange(0,sign.size)/Fs # initialize time vector (for mixing)
        if sign.shape[1] == 2:
            s = sign[:, 0] + 1j * sign[:, 1]
            t = t = np.arange(0, len(s)) / self.s_struct["Fs"]
        else:
            t = sign[:, 0]
            s = sign[:, 1] + 1j * sign[:, 2]
        # demodulate carriers
        for i in range(0, N_BW):
            BWi = BW[i]
            if BWi > 0 and (NR_car_id == -1 or i == NR_car_id):

                # mix current carrier so that it is centered at 0 Hz
                if car_return == False:
                    v_mixed = s * np.exp(-1j * 2 * np.pi * (self.Fc_gen + f_off[i]) * t)
                else:
                    carrier_offset = self.Fc_gen + f_off[i] + sig_offset
                    v_mixed = s * np.exp(-1j * 2 * np.pi * carrier_offset * t)
                osr = []
                # calculate OSR of current carrier
                for j in range(0, len(BWP[i])):
                    dl_osrl = self.NRparameters(mu=BWP[i][j][0], BW=BW[i])
                    osr.append(dl_osrl["Fs"])
                osr = np.around(Fs / np.array(osr))

                v_filt = v_mixed
                if self.fil == "on":
                    v_filt = self.NRfilter(
                        Fs=Fs, raw_vector=v_mixed, BW=BWi, osr=max(osr)
                    )

                if car_return == True:
                    t2 = (
                        np.arange(0, len(v_filt)) / self.s_struct["Fs"]
                    )  # NRfilter changes the array size
                    v_filt = v_filt * np.exp(+1j * 2 * np.pi * carrier_offset * t2)

                a = self.demNRdownlink(s=v_filt, BW=BWi, BWP=BWP[i], osr=osr)

                """
                osr=[]
                # calculate OSR of current carrier
                for j in range(0,len(BWP[i])):
                    dl_osrl=self.NRparameters(mu=BWP[i][j][0],BW=BW[i])
                    osr.append(dl_osrl["Fs"])
                osr=np.around(Fs/np.array(osr))
                
                a=self.demNRdownlink(s=s,BW=BWi,BWP=BWP[i],osr=osr)
                """
                cnstl.append(a)
            else:
                cnstl.append(np.array([]))
        return cnstl

    def genMultiNRdownlink(self):
        """Method for generating signal from constellation points for multiple carriers.


        Example
        -------
        self.genMultiNRdownlink()

        """

        # NOFDMsym=cnstl[0][0].size
        BW = self.BW
        BWP = self.BWP
        tot_osr = self.osr
        cnstl = self.cnstl
        N_BW = BW.size
        NFFT = []
        Fss = []
        LCM = int(1)

        # get integers proportional to base sampling rates
        # get also their LCM (least common multiple)
        BW_vect_abs = np.abs(BW)
        maxsf = 0
        slength = []
        for i in range(0, N_BW):
            sub_NFFT = []
            slen = []
            sub_fs = []
            if BW[i] != 0:

                if hasattr(BWP[i], "__len__"):
                    for j in range(0, len(BWP[i])):
                        up = self.NRparameters(mu=BWP[i][j][0], BW=BW_vect_abs[i])
                        start = up["RB"] * BWP[i][j][2]
                        stop = up["RB"] * BWP[i][j][3]
                        if stop < start:
                            raise Exception("Wrong BW of BWP")

                        sub_fs.append(up["Fs"])
                        LCM = np.lcm(LCM, int(up["Fs"]))
                        if BW[i] > 0:
                            sub_NFFT.append(up["NFFT"])
                            ncp = numerology.nr_cp_lengths(BWP[i][j][1], BWP[i][j][0], up)
                            slen.append(ofdm.symbol_starts(ncp, up["NFFT"])[-1])
                        if up["Fs"] > maxsf:
                            maxsf = up["Fs"]

                else:

                    BW_of_BWP = BW[i]
                    up2 = self.NRparameters(mu=mu[i], BW=BW_of_BWP)
                    NFFT.append(up2["NFFT"])
                    LCM = np.lcm(LCM, int(up2["NFFT"]))
                    if up2["Fs"] > maxsf:
                        maxsf = up2["Fs"]
            else:
                sub_fs.append(0)
            NFFT.append(sub_NFFT)
            Fss.append(sub_fs)
            slength.append(slen)

        # get integer proportional to overall sampling rate
        self.NFFT_debug = max(NFFT)
        NFFT_tot_min = 0

        a = np.array_split(Fss, len(Fss))
        a = np.concatenate(a, axis=0)
        for i in range(0, len(Fss)):
            NFFT_tot_min += max(a[i])

        NFFT_tot = tot_osr * LCM * np.ceil(NFFT_tot_min / LCM)
        # calculate individual oversampling ratios

        Fss = np.concatenate(a, axis=0)
        Fss = Fss[Fss != 0]
        osr = NFFT_tot * np.ones_like(Fss) / Fss
        max_len = []

        ind = 0
        for i in range(0, N_BW):
            if BW[i] > 0:
                sub_len = []
                for j in range(0, len(BWP[i])):
                    sub_len.append(osr[ind] * slength[i][j])
                    ind += 1
                max_len.append(sum(sub_len))
            else:
                ind += 1

        max_len = max(max_len)
        slength = max_len
        Fs = maxsf * min(osr)
        smatrix = np.zeros((int(slength + self.fil_len * max(osr)), int(N_BW)), complex)
        cnstlmatrix = []
        # generate carriers
        osr_ind = 0
        for i in range(0, N_BW):
            BWi = BW[i]
            if BWi > 0:
                sub_cnstlmatrix = []
                sub_smatrix = np.zeros((int(slength), 1), complex)
                out = self.genNRdownlink(
                    BW=BWi,
                    BWP=BWP[i],
                    osr=osr[osr_ind : int(osr_ind + len(BWP[i]))],
                    cnstl=cnstl[i],
                )
                cnstlmatrix.append(out["cnstl"])

                s = np.concatenate(out["s"])
                self.testvar = s
                if self.fil == "on":
                    s = np.pad(s, (0, (int(slength) - len(s))), constant_values=0)
                    s = self.NRfilter(
                        Fs=Fs,
                        raw_vector=s,
                        BW=BWi,
                        osr=max(osr[osr_ind : int(osr_ind + len(BWP[i]))]),
                    )
                if len(s) > len(smatrix):
                    zeros_matrix = np.zeros(
                        (len(s) - len(smatrix), np.shape(smatrix)[1])
                    )
                    smatrix = np.vstack([smatrix, zeros_matrix])
                smatrix[0 : len(s), i] = self.normalize(x=s, opt="max", k=1)
                osr_ind = int(osr_ind + len(BWP[i]))
            else:
                cnstlmatrix.append([])
                osr_ind = int(osr_ind + len(BWP[i]))

        s_raw = np.zeros(int(slength + self.fil_len), complex)
        BWtot = np.sum(BW_vect_abs)  # get total bandwidth (in Hz)

        # mix carriers to proper frequency offset
        f_off = np.zeros(N_BW)
        t_vect = np.arange(0, slength + self.fil_len) / Fs
        for i in range(0, N_BW):
            BWi = BW[i]

            if BWi > 0:
                f_off[i] = np.sum(BW_vect_abs[0:i]) + BWi / 2 - BWtot / 2
                s_raw = s_raw + smatrix[:, i] * np.exp(
                    1j * 2 * np.pi * (self.Fc_gen + f_off[i]) * t_vect
                )
        if self.norm == "max":
            s = self.normalize(x=s_raw, opt="max", k=1)  # normalize final signal to 1
        elif self.norm == "amp":
            s = self.normalize(x=s_raw, opt="amp", k=1)  # normalize final signal to 1

        # s=s_raw
        output_format = np.transpose(np.vstack((t_vect, np.real(s), np.imag(s))))
        out = {
            "s": output_format,
            "cnstl": cnstlmatrix,
            "Fs": Fs,
        }

        return out, f_off

    def genMultiQAM(self):
        """Method for generating QAM constellation points based on input bits for multiple carriers.

        Example
        -------
        self.genMultiQAM()

        """
        BW = self.BW
        BWP = self.BWP
        qam_type = self.QAM
        bits = self.in_bits

        N_BW = BW.size
        cnstl = []
        gen_bits = []
        for i in range(0, N_BW):
            BWi = BW[i]
            if BWi > 0:
                sub_cnstl = []
                sub_bits = []
                b = np.size(bits[i], 0)
                for j in range(0, b):
                    a, bit = self.genQAM(
                        carrier_id=i, bits=bits[i][j], BW=BW[i], BWP=BWP[i][j]
                    )
                    sub_cnstl.append(a)
                    sub_bits.append(bit)
                cnstl.append(sub_cnstl)
                gen_bits.append(sub_bits)
            else:
                cnstl.append([])
                gen_bits.append([])
        return cnstl, gen_bits

    def spline_inter(self, **kwargs):
        """Method for calculating spline interpolation for resource block sizeing.

        Parameters
        ----------
        mu : integer (0,1,2,3,4)
           5G NR numerology
        BW : integer
            Bandwidth of carrier.

        Example
        -------
        self.spline_inter(mu=1,BW=10e6)

        """

        mu = kwargs.get("mu")
        BW = kwargs.get("BW")

        if mu == 0:  # Numbers of RB from standard 38.101
            x = [0, 5e6, 10e6, 15e6, 20e6, 25e6, 40e6, 50e6]
            y = [0, 25, 52, 79, 106, 133, 216, 270]
        elif mu == 1:
            x = [0, 5e6, 10e6, 15e6, 20e6, 25e6, 40e6, 50e6, 60e6, 80e6, 100e6]
            y = [0, 11, 24, 38, 51, 65, 106, 133, 162, 217, 273]
        elif mu == 2:
            x = [0, 10e6, 15e6, 20e6, 25e6, 40e6, 50e6, 60e6, 80e6, 100e6, 200e6]
            y = [0, 11, 18, 24, 31, 51, 65, 79, 107, 135, 264]
        elif mu == 3:
            x = [0, 50e6, 100e6, 200e6, 400e6]
            y = [0, 32, 66, 132, 264]
        elif mu == 4:
            x = [0, 100e6, 200e6, 400e6]
            y = [0, 32, 64, 128]

        if mu == 4:
            p = inter.interp1d(x, y)
        else:
            p = inter.interp1d(x, y, kind="cubic")

        try:
            RB = int(np.floor(p(BW)))
        except:
            raise Exception("Width of selected BWP is too big for selected mu")

        return RB

    def NRparameters(self, **kwargs):
        """Method for calculating necessary parameters

        Parameters
        ----------
        mu : integer (0,1,2,3,4)
           5G NR numerology
        BW : integer
            Bandwidth of carrier
        osr : integer
            Oversampling factor

        Example
        -------
        self.NRparameters(mu=1, BW=10e6, osr= 1)

        """

        mu = kwargs.get("mu")
        BW = kwargs.get("BW")
        gen = kwargs.get("gen", 0)
        N_slot_in_subframe = 2**mu
        SCS = 2**mu * 15e3

        min_BW = 20 * SCS * 12

        RB = self.spline_inter(mu=mu, BW=BW)
        if gen == 1:
            self.BW_conf.append(RB * 12 * SCS)
        temp1 = [
            0,
            5e6,
            10e6,
            15e6,
            20e6,
            25e6,
            40e6,
            50e6,
            60e6,
            80e6,
            100e6,
            200e6,
            400e6,
        ]
        temp2 = [
            0,
            25 * 12 * 15e3,
            52 * 12 * 15e3,
            79 * 12 * 15e3,
            106 * 12 * 15e3,
            133 * 12 * 15e3,
            216 * 12 * 15e3,
            270 * 12 * 15e3,
            162 * 12 * 30e3,
            217 * 12 * 30e3,
            273 * 12 * 30e3,
            264 * 12 * 60e3,
            264 * 12 * 120e3,
        ]
        if BW in temp1 and gen == 1:
            self.ACLR_BW.append(temp2[np.where(temp1 == BW)[0][0]])

        if BW < min_BW:
            raise Exception("Width of selected BWP is too small for selected mu")

        NFFT = 2 ** np.ceil(np.log2(RB * 12 / 0.9))  # FFT size
        NFFT = max(128, NFFT)
        # NFFT=4096
        Tc = 1 / (15e3 * 2**mu * NFFT)
        Ts = 1 / (15e3 * 2048)
        k = Ts / Tc
        Fs = NFFT * SCS  # sampling frequency
        Ncp1 = 144 * k * (1 / (2**mu)) + 16 * k  # length of cyclic prefix 0 and 7*2**mu
        Ncp2 = 144 * k * (1 / (2**mu))  # length of cyclic prefixes else
        Nofdm1 = NFFT + Ncp1  # length of OFDM symbol 0
        Nofdm2 = NFFT + Ncp2  # length of OFDM symbols 1-6

        if "osr" in kwargs:
            osr = kwargs.get("osr")
        else:
            osr = 1

        # set parameters that are not proportional to BW:
        # - W     = EVM window length
        # - Lroll = optimum symbol rolloff length to keep EVM < 1%
        # - RB    = number of Resource Blocks
        Lroll = 0
        if NFFT == 128:
            Lroll = 4
        elif NFFT == 256:
            Lroll = 6
        elif NFFT == 512:
            Lroll = 4
        elif NFFT == 1024:
            Lroll = 6
        elif NFFT == 2048:
            Lroll = 8
        if Lroll == 0:
            Lroll = max(0, 8 - 2 * (11 - (np.log2(NFFT))))
        up = {
            "Fs": Fs * osr,
            "NFFT": NFFT * osr,
            "Ncp1": Ncp1 * osr,
            "Ncp2": Ncp2 * osr,
            "Nofdm1": Nofdm1 * osr,
            "Nofdm2": Nofdm2 * osr,
            # "Nslot":Nslot*osr,
            # "W":W*osr,
            "RB": RB,
            "Nsc": RB * 12,  # number of occupied subcarriers
            "Lroll": Lroll,
        }

        return up

    def genQAM(self, **kwargs):
        """Method for generating constellation point based on input bits.

        Parameters
        ----------
        bits : array of binary values
           Binary input data
        BW : integer
            Bandwidth of carrier
        BWP : array of certain structure
            Bandwidth part of corresponding carrier. [a,b,c,d]
            where a=numerology,b=number of OFDM symbols, c=lowest used frequency of bandwidth in %
            d=highest used frequency of bandwidth in %, 0=<c<d=<1.
        Example
        -------
        self.genQAM(bits=[1,0...1,2],BW=10e6, BWP=[0,14,0,1])

        """

        carrier_id = kwargs.get("carrier_id")
        bits = kwargs.get("bits")
        BW = kwargs.get("BW")
        BWP = kwargs.get("BWP")

        qam_type = self.QAM
        dl = self.NRparameters(mu=BWP[0], BW=BW)
        Nsymb = int(BWP[1])
        Ncnstl = int(grid.data_re_mask(dl["RB"], Nsymb, BWP[2], BWP[3]).sum())

        if bits == "max":
            seed = (carrier_id + self.signal_id * 13 + 1) * 123
            rng = np.random.RandomState(seed)
            bits = rng.randint(2, size=constellation.BITS_PER_SYMBOL[self.QAM] * Ncnstl)

        m = constellation.BITS_PER_SYMBOL[qam_type]
        return constellation.modulate(bits, qam_type), np.reshape(bits, (-1, m))

    def QAMtoBit(self, **kwargs):
        """Method for generating binary array based on constellation points.

        Parameters
        ----------
        cnstl : array
           Array of constellation points
        BW : integer
            Bandwidth of carrier
        BWP : array of certain structure
            Bandwidth part of corresponding carrier. [a,b,c,d]
            where a=numerology,b=number of OFDM symbols, c=lowest used frequency of bandwidth in %
            d=highest used frequency of bandwidth in %, 0=<c<d=<1.
        Example
        -------
        self.QAMtoBit(cnstl=[0.2+i*0.5,...,1-i*0,7], BW=10e6, BWP=[1,7,0,1])

        """

        cnstl = kwargs.get("cnstl")
        BW = kwargs.get("BW")
        BWP = kwargs.get("BWP")

        qam_type = self.QAM
        cnstl_of_carrier = []
        vector = []

        for n in range(0, len(cnstl)):
            if cnstl == []:
                cnstl_of_carrier.append([])
                vector.append([])

            mu = BWP[n][0]
            dl = self.NRparameters(mu=mu, BW=BW)  # get NR parameters
            N_ID_1 = 0  # physical-layer cell-identity group
            N_ID_2 = 0  # physical-layer identity within the group
            N_ID_cell = 3 * N_ID_1 + N_ID_2  # cell identity
            Nsymb = int(BWP[n][1])
            mask = grid.data_re_mask(dl["RB"], Nsymb, BWP[n][2], BWP[n][3])
            DataSymbols_nzero = grid.data_symbols(cnstl[n], mask).reshape(-1, 1)
            cnstl_of_carrier.append(DataSymbols_nzero)
            vector.append(constellation.demodulate(DataSymbols_nzero.flatten(), qam_type))
        return vector, cnstl_of_carrier

    def genNRdownlink(self, **kwargs):
        """Method for generating 5G NR signal based on constellation points.

        Parameters
        ----------
        BW : integer
            Bandwidth of carrier
        BWP : array of certain structure
            Bandwidth part of corresponding carrier. [a,b,c,d]
            where a=numerology,b=number of OFDM symbols, c=lowest used frequency of bandwidth in %
            d=highest used frequency of bandwidth in %, 0=<c<d=<1.
        osr : integer
            Oversampling factor
        cnstl : array
            Array of constellation points
        Example
        -------
        self.genNRdownlink(BW=10e6, BWP=[2,7,0,1],osr=1, cnstl=[1-i*0.6,...,-0,6+i*0.3])

        """

        BW = kwargs.get("BW")
        BWP = kwargs.get("BWP")
        osr = kwargs.get("osr")
        cnstl = kwargs.get("cnstl")

        signal = []
        REs = []
        for n in range(0, len(BWP)):
            mu = BWP[n][0]
            dl = self.NRparameters(mu=mu, BW=BW, osr=osr[n], gen=1)
            N_ID_1 = 0  # physical-layer cell-identity group
            N_ID_2 = 0  # physical-layer identity within the group
            N_ID_cell = 3 * N_ID_1 + N_ID_2  # cell identity
            Nsymb = int(BWP[n][1])
            ncp = numerology.nr_cp_lengths(Nsymb, mu, dl)

            # RE=np.copy(cnstl)
            RE = np.full((dl["Nsc"], Nsymb), None)
            start = int(np.floor(dl["RB"] * BWP[n][2]) * 12)
            stop = int(np.ceil(dl["RB"] * BWP[n][3]) * 12)
            BW_of_BWP = (stop - start) * (15e3 * 2**mu)
            dl2 = self.NRparameters(
                mu=mu, BW=BW_of_BWP, osr=osr[n]
            )  # To check if BWP is more than 20 RB
            RE[:start, :] = 0
            RE[stop:, :] = 0
            # RE=np.zeros((dl["Nsc"],Nsymb),complex)
            values, is_ref = grid.reference_grid(
                dl["RB"], Nsymb, mu, BWP[n][2], BWP[n][3], N_ID_cell, N_ID_2
            )
            RE[is_ref] = values[is_ref]

            mask = grid.data_re_mask(dl["RB"], Nsymb, BWP[n][2], BWP[n][3])
            unusedOFDM = np.argwhere(mask.T)[:, ::-1]

            if len(cnstl[n]) > len(unusedOFDM):
                raise Exception("Not enough OFDM symbols")

            for i in range(0, len(cnstl[n])):
                l = unusedOFDM[i][1]
                k = unusedOFDM[i][0]
                RE[k, l] = cnstl[n][i]

            RE[RE == None] = 0 + 0 * 1j
            RE = RE.astype(complex)

            s = ofdm.modulate(RE, dl["NFFT"], ncp, dl["Lroll"] * osr[n])

            signal.append(s)
            REs.append(RE)
        out = {"s": signal, "cnstl": REs, "Fs": dl["Fs"]}

        return out

    def normalize(self, **kwargs):
        """Method for normalizing signal.

        Parameters
        ----------
        x : array
           Signal
        opt : sting
            Normalization option (max, pow)
        k : float
            Input is normalized to this value
            - if option = "max", then max(|Re(yi)|, |Im(yi)|) = k
            - if option = "pow" or "totpow", then the statistical power of y  is k^2

        Example
        -------
        self.normalize(x=[0.3,...,0.7], opt="max",k=1)

        """

        x = kwargs.get("x")
        opt = kwargs.get("opt")
        k = kwargs.get("k")

        # INPUTS:
        # x = input vector or matrix
        # opt = can be either "max", "pow", or "totpow"
        # k = input is normalized to this factor
        #     - if option = "max", then max(|Re(yi)|, |Im(yi)|) = k
        #     - if option = "pow" or "totpow", then the statistical power of y (or
        #       each of its columns) is k^2
        if opt == "max":
            test_matrix = np.concatenate(
                (np.absolute(np.real(x)), np.absolute(np.imag(x)))
            )
            max_value = test_matrix.max()
            y = k * x / max_value

        elif opt == "amp":
            amp = np.sqrt((np.real(x) ** 2) + (np.imag(x) ** 2))
            max_amp = max(amp)
            y = k * x / max_amp

        elif opt == "pow":
            pow = (np.absolute(x) ** 2).sum() / len(x)
            y = k * x / np.sqrt(pow)

        elif opt == "totpow":
            """TBD if needed"""
        return y

    def NRfilter(self, **kwargs):
        """Method for filtering signal.

        Parameters
        ----------
        Fs : integer
           Sampling frequency
        raw_vector : array
            Signal to be filtered
        BW : integer
            Bandwidth
        osr : integer
            Oversampling factor
        Example
        -------
        self.NRfilter(Fs=5e9,raw_vector=[0.3,...,0.7], opt="max",k=1)

        """

        Fs = kwargs.get("Fs")
        raw_vector = kwargs.get("raw_vector")
        BW = kwargs.get("BW")

        # get sampling frequency

        Fs = int(Fs)
        if "osr" in kwargs:
            osr = kwargs.get("osr")
        else:
            osr = 1
        # select filter parameters
        # fac=1
        # fac=BW/150e6
        # order=self.fil_len*np.ceil(osr*fac) # filter order

        dF = 0
        for mu in range(0, 5):
            try:
                up = self.NRparameters(mu=(4 - mu), BW=BW)
                if ((up["RB"]) * 12 * 15e3 * 2 ** (4 - mu)) > dF:
                    dF = (up["RB"]) * 12 * 15e3 * 2 ** (4 - mu)
            except:
                continue
        dF = (BW - dF) * 0.5

        Fc = BW / 2  # center of transition bandwidth
        t_vect = np.arange(0, len(raw_vector)) / Fs
        # filter NR signal with circular convolution
        # order=200

        ripple_dB = -1
        att_dB = -80
        # ripple=ripple_dB
        # att=att_dB
        ripple = 10 ** (ripple_dB / 20)
        att = 10 ** (att_dB / 20)
        D = (
            0.005309 * (np.log10(ripple)) ** 2 + 0.07114 * (np.log10(ripple)) - 0.4761
        ) * (np.log10(att)) - (
            0.00266 * (np.log10(ripple)) ** 2 + 0.5941 * (np.log10(ripple)) + 0.4278
        )
        f = 11.012 + 0.51244 * (np.log10(ripple) - np.log10(att))
        ord_approx = (D - f * (dF / Fs) ** 2) / (dF / Fs) + 1  # Herrmann
        ord_approx2 = (-20 * np.log10(np.sqrt(ripple * att)) - 13) / (
            14.6 * dF / Fs
        ) + 1

        mid = (ord_approx + ord_approx2) / 2
        N_min = max(20, int(mid - 100))
        N_max = int(mid + 100)
        # print(ord_approx)
        # print(ord_approx2)
        sb_max_dB = -80
        # if int(ord_approx)%2!=0:
        #    ord_approx+=1
        # if int(ord_approx2)%2!=0:
        #    ord_approx2+=1
        test = N_min
        sb_best = 1
        att_freq = Fc + dF
        if att_freq > Fs / 2:
            att_freq = Fc

        sb_max_dB_list = [i for i in range(0, abs(sb_max_dB - 10), 20)]
        sb_max_dB_list.reverse()
        sb_max_list = [
            10 ** (-sb_max_dB_list[i] / 20) for i in range(len(sb_max_dB_list))
        ]
        done = False

        # python doesn't allow you to change the for loop iterator mid-loop so let's make a 1D-loop
        allindices = []
        for sb_max in sb_max_list:
            for ord in range(N_min, N_max):
                allindices.append([sb_max, ord])

        skipcount = 32  # skip n iterations during the broad-phase
        skipindex = 0

        for currentphase in range(2):  # broad-phase first and then narrow-phase
            for index in allindices:
                sb_max, ord = index

                # broad phase iteration (when skipcount != 0)
                if skipindex > 0:
                    skipindex -= 1
                    continue
                skipindex = skipcount

                try:
                    b = sig.remez(
                        int(ord + 1),
                        [0, Fc - dF, att_freq, 0.5 * Fs],
                        [1, 0],
                        fs=int(Fs),
                    )
                    # order=ord
                    # plt.figure()
                    # plt.plot(b)
                    # plt.show(block=False)
                    w, h = sig.freqz(b, [1], worN=2000, fs=Fs)
                    idx = np.argwhere(w > ((att_freq)))[0][0]
                    sb = max(np.abs(h[idx:]))
                    if sb < sb_best:
                        sb_best = sb
                    if sb <= sb_max:
                        order = ord
                        self.fil_len = ord
                        break
                except Exception as e:
                    self.print_log(type="W", msg=f"{e}")
            else:
                continue
            break

            # go back n iterations and try again with the narrow-phase
            if currentphase == 0:
                endindex = allindices.index(index)
                startindex = endindex - skipcount
                if startindex < 0:
                    startindex = 0
                allindices = allindices[startindex : endindex + 1]
                skipcount = 0
                skipindex = 0

        bb = b
        # bb=np.pad(b,(0,len(raw_vector)-len(b)),constant_values=0)
        # s_fil=np.fft.ifft(np.fft.fft(raw_vector,len(raw_vector))*np.fft.fft(bb,len(raw_vector)))
        s_fil = np.convolve(raw_vector.reshape((-1, 1))[:, 0], bb, mode="full").reshape(
            (-1, 1)
        )
        # compensate group delay (order/2)
        s_out = np.concatenate((s_fil[int(order / 2) :], s_fil[0 : int(order / 2)]))
        # s_out=s_fil
        return s_out[:, 0]

    def demNRdownlink(self, **kwargs):
        """Method for demodulation signal.

        Parameters
        ----------
        s : array
           Signal
        BW : integer
            Bandwidth
        BWP : array of certain structure
            Bandwidth part of corresponding carrier. [a,b,c,d]
            where a=numerology,b=number of OFDM symbols, c=lowest used frequency of bandwidth in %
            d=highest used frequency of bandwidth in %, 0=<c<d=<1.
        osr : integer
            Oversampling factor

        Example
        -------
        self.demNRdownlink(s=[0.3,...,0.7], BW=10e6,BWP=[1,14,0,1], osr=1)

        """

        s = kwargs.get("s")
        BW = kwargs.get("BW")
        BWP = kwargs.get("BWP")
        osr = kwargs.get("osr")

        FFTwpos = 0
        end_of_prev_sig = 0
        equalize = self.equalizer
        cnstl_of_carrier = []
        # if self.fil=="on":

        #    #sign=self.NRfilter(sign,BW,osr)
        for n in range(0, len(BWP)):
            mu = BWP[n][0]
            N_symb_TOT = int(BWP[n][1])

            # get various parameters related to input signal
            dl = self.NRparameters(mu=mu, BW=BW, osr=osr[n])  # general NR parameters
            ncp = numerology.nr_cp_lengths(N_symb_TOT, mu, dl)
            starts = ofdm.symbol_starts(ncp, dl["NFFT"])
            N_sampl = int(starts[-1])
            if len(BWP) > 1:
                sign = s[end_of_prev_sig : end_of_prev_sig + N_sampl]
                end_of_prev_sig = end_of_prev_sig + N_sampl
            else:
                sign = s
                N_sampl = len(s)
            # if self.fil=="on":
            #    sign=self.NRfilter(mu,sign,BW,osr[n])
            N_slots = np.floor(N_symb_TOT / 14)  # number of slots (integer)
            N_ID_1 = 0  # physical-layer cell-identity group
            N_ID_2 = 0  # physical-layer identity within the group
            N_ID_cell = 3 * N_ID_1 + N_ID_2  # cell identity
            # generate PSS in the time-domain
            d_PSS = sequences.pss(N_ID_2)  # generate PSS sequence
            # map PSS sequence to subcarriers
            PSSf = np.zeros((int(dl["NFFT"])), complex)
            PSSf[-64:] = d_PSS[0:64]
            PSSf[1:64] = d_PSS[64:127]
            PSSt = np.fft.ifft(PSSf)  # convert to time-domain
            # find match of PSS within the input vector, by computing cross-correlation
            # and looking for its maximum
            pad = np.pad(PSSt, (0, len(sign) - len(PSSt)), constant_values=0)
            xcmax = np.argmax(np.absolute(np.correlate(sign, pad, "full")))
            # calculate ideal result of the cross-correlation maximum (see above)
            if N_symb_TOT == 0:
                print("ERROR")
                return 0
            l_pss = min(N_symb_TOT, 4) - 1
            xcmax_id = N_sampl + starts[l_pss] + ncp[l_pss]

            # circularly shift input vector so that PSS location becomes "as expected"
            invect_aligned = np.roll(sign, int(xcmax_id - xcmax) - 1)

            # calculate shift of FFT window from "nominal" position (i.e. when CP is
            # completely discarded)
            # invect_aligned[:]=1
            FFTw_shift_1 = -dl["Ncp1"] / 2 + FFTwpos
            FFTw_shift_2 = -1 * np.ceil(dl["Ncp2"] / 2) + FFTwpos
            long_cp = np.arange(N_symb_TOT) % (7 * 2**mu) == 0
            FFTw_shift = np.where(long_cp, FFTw_shift_1, FFTw_shift_2)
            FFTw_start = (starts[:-1] + ncp + FFTw_shift).astype(int)
            OFDMmatrix = invect_aligned[FFTw_start[:, None] + np.arange(int(dl["NFFT"]))].T

            # calculate compensation factor, due to shift of FFT window from "nominal"
            # position (i.e. when CP is completely discarded)
            k = np.arange(0, dl["NFFT"])
            e = np.outer(
                np.exp(-1j * 2 * np.pi * k * FFTw_shift_2 / dl["NFFT"]),
                np.ones((int(N_symb_TOT))),
            )
            for index in np.arange(0, int(N_symb_TOT), (7 * 2**mu)):
                e[:, int(index)] = np.exp(
                    -1j * 2 * np.pi * k * FFTw_shift_1 / dl["NFFT"]
                )

            # calculate FFT and apply compensation factor
            subcarriers = np.multiply(
                np.transpose(np.fft.fft(np.transpose(OFDMmatrix))), e
            )
            # map subcarriers to resource elements
            RE = np.zeros((int(dl["Nsc"]), int(N_symb_TOT)), complex)
            RE[0 : int(dl["Nsc"] / 2), :] = subcarriers[int(-dl["Nsc"] / 2) :, :]
            RE[int(dl["Nsc"] / 2) :, :] = subcarriers[0 : int(dl["Nsc"] / 2), :]
            # if equalization is deactivated, return at this point already
            if equalize == "off":
                cnstl = RE
                cnstl_of_carrier.append(cnstl)
                continue
            # re-create DMRS/PSS grid (i.e. post-FFT ideal reference signal)
            RE_id = np.ones((RE.shape), complex)
            start = int(np.floor(dl["RB"] * BWP[n][2]) * 12)
            stop = int(np.ceil(dl["RB"] * BWP[n][3]) * 12)
            values, is_ref = grid.reference_grid(
                dl["RB"], N_symb_TOT, mu, BWP[n][2], BWP[n][3], N_ID_cell, N_ID_2
            )
            RE_id[is_ref] = values[is_ref]

            # calculate the complex ratios of the post-FFT acquired signal "RE" and the
            # post-FFT ideal signal "RE_id", for each reference symbol
            complex_ratios = np.divide(RE, RE_id)
            a = np.absolute(complex_ratios)
            phi = np.angle(complex_ratios)
            #  unwrap phase of complex ratios at symbol #0 of each slot
            l = np.arange(0, N_symb_TOT, dtype=int)
            l = l[l % 14 == 2]
            k1 = np.arange(start, stop, 2)
            for k in k1:
                for i in np.arange(1, l.size):
                    delta_phi = phi[k, int(l[i])] - phi[k, int(l[i - 1])]
                    if np.absolute(delta_phi) >= np.pi:
                        phi[k, l[i:]] = phi[k, l[i:]] - 2 * np.pi * np.sign(delta_phi)

            # perform time averaging at each reference signal subcarrier of the complex
            # ratios (in TS 36.104 the time-averaging length is 10 subframes, here for
            # simplicity the time-averaging length is that of the signal)
            a_avg = np.zeros((int(dl["Nsc"])))
            phi_avg = np.zeros((int(dl["Nsc"])))

            l = np.arange(0, N_symb_TOT, dtype=int)
            l = l[l % 14 == 2]
            k1 = np.arange(start, stop, 2)

            for k in k1:
                test = a[int(k), l]
                a_avg[int(k)] = np.mean(a[int(k), l])
                phi_avg[int(k)] = np.mean(phi[int(k), l])

            # the equalizer coefficients for amplitude and phase "a_coeff" and
            # "phi_coeff" at the reference signal subcarriers are obtained by computing
            # the moving average in the frequency domain of the time-averaged reference
            # signal subcarriers, i.e. every third subcarrier (or sixth, if less than 5
            # OFDM symbols)
            a_coeff = np.zeros((int(dl["Nsc"])))
            phi_coeff = np.zeros((int(dl["Nsc"])))
            k_PSS = np.arange(0, 240) - np.ceil(240 / 2) + dl["Nsc"] / 2
            k = np.arange(start, stop, 2)

            if N_symb_TOT == 3:  # exclude 5+5 null reference subcarriers around the PSS
                for i in np.concatenate((range(0, 56), range(182, 239))):
                    if np.any(k + 1 == k_PSS[i]):
                        ind_k_to_remove = np.argwhere(k + 1 == k_PSS[i])
                        k = k[
                            np.concatenate(
                                (
                                    np.arange(0, ind_k_to_remove - 1),
                                    np.arange(ind_k_to_remove, -1),
                                )
                            )
                        ]

            for i in np.arange(1, k.size + 1):
                m_avg_w_length = min(2 * i - 1, 2 * (k.size - i) + 1, 19)
                m_avg_imp_resp = np.ones(m_avg_w_length) / m_avg_w_length
                test = np.dot(
                    m_avg_imp_resp,
                    a_avg[
                        k[
                            int(i - np.floor(m_avg_w_length / 2) - 1) : int(
                                i + np.floor(m_avg_w_length / 2)
                            )
                        ]
                    ],
                )
                a_coeff[k[int(i - 1)]] = np.dot(
                    m_avg_imp_resp,
                    a_avg[
                        k[
                            int(i - np.floor(m_avg_w_length / 2) - 1) : int(
                                i + np.floor(m_avg_w_length / 2)
                            )
                        ]
                    ],
                )
                phi_coeff[k[int(i - 1)]] = np.dot(
                    m_avg_imp_resp,
                    phi_avg[
                        k[
                            int(i - np.floor(m_avg_w_length / 2) - 1) : int(
                                i + np.floor(m_avg_w_length / 2)
                            )
                        ]
                    ],
                )

            # perform linear interpolation to compute coefficients for each subcarrier
            a_coeff = np.interp(np.arange(1, dl["Nsc"] + 1), k + 1, a_coeff[k])
            phi_coeff = np.interp(np.arange(1, dl["Nsc"] + 1), k + 1, phi_coeff[k])

            cnstl = np.zeros((RE.shape), complex)
            # equalize resource elements and return
            for i in np.arange(0, dl["Nsc"]):
                cnstl[i, :] = RE[i, :] / (a_coeff[i] * np.exp(1j * phi_coeff[i]))

            cnstl_of_carrier.append(cnstl)

        return cnstl_of_carrier

    def measEVMdownlink(self, **kwargs):
        """Method for calculation EVM.

        Parameters
        ----------
        BW : integer
            Bandwidth
        BWP : array of certain structure
            Bandwidth part of corresponding carrier. [a,b,c,d]
            where a=numerology,b=number of OFDM symbols, c=lowest used frequency of bandwidth in %
            d=highest used frequency of bandwidth in %, 0=<c<d=<1.
        cnstl : array
            Generated constellation points used as reference
        dem : array
            Recieved constellation points
        Example
        -------
        self.measEVMdownlink(BW=10e6,BWP=[1,14,0,1],cnstl=[1-i*0.6,...,-0,6+i*0.3],dem=[1-i*0.6,...,-0,6+i*0.3])

        """
        BW = kwargs.get("BW")
        BWP = kwargs.get("BWP")
        cnstl = kwargs.get("cnstl")
        dem = kwargs.get("dem")

        Nsc, N_symb = cnstl.shape  # get some info from constellation matrix dimensions
        mu = BWP[0]
        if dem.size == 0:
            return 0, 0
        Nsc_rx, N_symb_rx = dem.shape
        if Nsc != Nsc_rx or N_symb != N_symb_rx:
            return 0, 0
        dl = self.NRparameters(mu=mu, BW=BW)  # get NR parameters
        N_ID_1 = 0  # physical-layer cell-identity group
        N_ID_2 = 0  # physical-layer identity within the group
        N_ID_cell = 3 * N_ID_1 + N_ID_2  # cell identity
        mask = grid.data_re_mask(dl["RB"], N_symb, BWP[2], BWP[3])
        rxDataSymbols_nzero = grid.data_symbols(dem, mask).reshape(-1, 1)
        refDataSymbols_nzero = grid.data_symbols(cnstl, mask).reshape(-1, 1)
        # normalize constellation powers
        rxDataSymbols = self.normalize(x=rxDataSymbols_nzero, opt="pow", k=1)
        refDataSymbols = self.normalize(x=refDataSymbols_nzero, opt="pow", k=1)
        # rxDataSymbols = rxDataSymbols_nzero
        # refDataSymbols = refDataSymbols_nzero

        # calculate EVM using Matlab's built-in functions
        # Copied from https://github.com/TheSDK-blocks/f2_testbench/blob/master/f2_testbench/analyzers_mixin.py

        reference = refDataSymbols
        received = rxDataSymbols

        # Takes zeros into account
        pad = np.zeros(np.shape(received), complex)
        pad[: np.shape(reference)[0], : np.shape(reference)[1]] = reference
        reference = pad

        # Do not take zeros into account
        # received=np.delete(received,range(len(reference),len(received)))

        # Shape the vectors: time is row observation is column
        reference.shape = (-1, 1)
        received.shape = (-1, 1)

        # RMS for Scaling
        rmsref = np.std(reference)
        rmsreceived = np.std(received)
        EVM = (
            np.mean(
                np.mean(np.abs(received - reference) ** 2, axis=0)
                / np.mean(np.abs(reference) ** 2, axis=0)
            )
        ) ** (1 / 2)

        return EVM, rxDataSymbols
