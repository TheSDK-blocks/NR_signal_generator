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
import matplotlib.pyplot as plt
from plot_PSD import plot_PSD

from . import constellation
from . import equalizer
from . import filters
from . import grid
from . import measurements
from . import multicarrier
from . import numerology
from . import ofdm
from . import sequences
from . import sync


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
            "FR",
        ]

        self.IOS = Bundle()
        self.IOS.Members["in_dem"] = IO()  # Pointer for input data

        self.IOS.Members["out"] = IO()  # Pointer for output data

        self.BWP = np.array([[1, 7, 0, 273]])
        self.QAM = "64QAM"
        self.osr = 1
        self.BW = np.array([100e6])
        self.FR = "FR1"  # frequency range, "FR1" or "FR2-1"
        self.in_bits = np.array(["max"])

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
        self.tx_filter = True  # transmit channel filter in the generator
        self.rx_filter = True  # receive channel-select filter in the analyzer
        self.equalizer = "on"

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
        self.s_struct = self.genMultiNRdownlink()
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
        self.EVM = self.measMultiEVMdownlink(NR_car_id=NR_car_id)

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
            if BW[i] > 0 and (NR_car_id == -1 or i == NR_car_id):
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
        cnstl = self.cnstl

        N_BW = len(cnstl)
        EVM = []
        # measure EVM separately for each constellation
        for i in range(0, N_BW):
            if BW[i] > 0 and (NR_car_id == -1 or i == NR_car_id):
                # reference: the transmitted symbols, in the order they are mapped
                EVM.append(measurements.evm(cnstl[i], self.dem_cnstl_vec[i]))
            else:
                EVM.append(np.array([]))
        return EVM

    def demMultiNRdownlink(self, **kwargs):
        """Method for demodulate constellation points from recieved signal for multiple carriers.

        The received signal must be one period of a cyclic signal, as produced
        by the generator: the receive filter, the PSS alignment and the FFT
        windows all treat it as periodic. Non-cyclic signals, such as arbitrary
        captures, are not supported.

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
        plan = multicarrier.plan(self.FR, BW, BWP, self.osr)
        Fs = plan["Fs"]
        f_off = plan["f_off"]
        FFRwpos = 0
        N_BW = BW.size
        cnstl = []
        # t=np.arange(0,sign.size)/Fs # initialize time vector (for mixing)
        if sign.shape[1] == 2:
            s = sign[:, 0] + 1j * sign[:, 1]
            t = np.arange(0, len(s)) / Fs
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

                v_filt = v_mixed
                if self.rx_filter:
                    v_filt = filters.nr_filter(v_mixed, Fs, BWi, self.FR)

                if car_return == True:
                    t2 = np.arange(0, len(v_filt)) / Fs
                    v_filt = v_filt * np.exp(+1j * 2 * np.pi * carrier_offset * t2)

                a = self.demNRdownlink(s=v_filt, BW=BWi, BWP=BWP[i], osr=plan["osr"][i])
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

        BW = self.BW
        BWP = self.BWP
        cnstl = self.cnstl
        N_BW = BW.size
        plan = multicarrier.plan(self.FR, BW, BWP, self.osr)
        Fs = plan["Fs"]
        osr = plan["osr"]
        f_off = plan["f_off"]

        smatrix = np.zeros((plan["length"], int(N_BW)), complex)
        # generate carriers
        for i in range(0, N_BW):
            BWi = BW[i]
            if BWi > 0:
                out = self.genNRdownlink(BW=BWi, BWP=BWP[i], osr=osr[i], cnstl=cnstl[i])

                s = out["s"]
                self.testvar = s
                if self.tx_filter:
                    s = filters.nr_filter(s, Fs, BWi, self.FR)
                smatrix[0 : len(s), i] = self.normalize(s, "max")

        s_raw = np.zeros(len(smatrix), complex)

        # mix carriers to proper frequency offset
        t_vect = np.arange(0, len(smatrix)) / Fs
        for i in range(0, N_BW):
            BWi = BW[i]

            if BWi > 0:
                s_raw = s_raw + smatrix[:, i] * np.exp(
                    1j * 2 * np.pi * (self.Fc_gen + f_off[i]) * t_vect
                )
        s = self.normalize(s_raw, self.norm)
        output_format = np.transpose(np.vstack((t_vect, np.real(s), np.imag(s))))
        out = {
            "s": output_format,
            "Fs": Fs,
        }

        return out

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
                a, bit = self.genQAM(carrier_id=i, bits=bits[i], BW=BW[i], BWP=BWP[i])
                cnstl.append(a)
                gen_bits.append(bit)
            else:
                cnstl.append([])
                gen_bits.append([])
        return cnstl, gen_bits

    def genQAM(self, **kwargs):
        """Method for generating constellation point based on input bits.

        Parameters
        ----------
        bits : array of binary values
           Binary input data that exactly fill every data resource element of
           the BWP, or "max" for pseudorandom bits. "max" stands in for MAC
           padding and scrambling and is seeded from carrier_id and signal_id,
           so it is reproducible. User data shorter than the allocation must be
           padded by the caller, e.g. with pseudorandom bits.
        BW : integer
            Bandwidth of carrier
        BWP : array of certain structure
            Bandwidth part of corresponding carrier. [a,b,c,d]
            where a=numerology, b=number of OFDM symbols, c=first resource block of
            the BWP (N_BWP^start), d=number of resource blocks of the BWP
            (N_BWP^size), c + d <= number of resource blocks of the carrier.
        Example
        -------
        self.genQAM(bits=[1,0...1,2],BW=10e6, BWP=[0,14,0,52])

        """

        carrier_id = kwargs.get("carrier_id")
        bits = kwargs.get("bits")
        BW = kwargs.get("BW")
        BWP = kwargs.get("BWP")

        qam_type = self.QAM
        dl = numerology.nr_parameters(self.FR, BWP[0], BW)
        Nsymb = int(BWP[1])
        Ncnstl = int(grid.data_re_mask(dl["RB"], Nsymb, BWP[2], BWP[3]).sum())
        m = constellation.BITS_PER_SYMBOL[qam_type]

        if isinstance(bits, str) and bits == "max":
            seed = (carrier_id + self.signal_id * 13 + 1) * 123
            rng = np.random.RandomState(seed)
            bits = rng.randint(2, size=m * Ncnstl)
        elif len(bits) != m * Ncnstl:
            raise ValueError(
                f"carrier {carrier_id}: {len(bits)} bits do not fill the {m * Ncnstl}-bit allocation"
            )

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
            where a=numerology, b=number of OFDM symbols, c=first resource block of
            the BWP (N_BWP^start), d=number of resource blocks of the BWP
            (N_BWP^size), c + d <= number of resource blocks of the carrier.
        Example
        -------
        self.QAMtoBit(cnstl=[0.2+i*0.5,...,1-i*0,7], BW=10e6, BWP=[1,7,0,24])

        """

        cnstl = kwargs.get("cnstl")
        BW = kwargs.get("BW")
        BWP = kwargs.get("BWP")

        qam_type = self.QAM

        mu = BWP[0]
        dl = numerology.nr_parameters(self.FR, mu, BW)  # get NR parameters
        N_ID_1 = 0  # physical-layer cell-identity group
        N_ID_2 = 0  # physical-layer identity within the group
        N_ID_cell = 3 * N_ID_1 + N_ID_2  # cell identity
        Nsymb = int(BWP[1])
        mask = grid.data_re_mask(dl["RB"], Nsymb, BWP[2], BWP[3])
        DataSymbols_nzero = grid.data_symbols(cnstl, mask)
        vector = constellation.demodulate(DataSymbols_nzero, qam_type)
        return vector, DataSymbols_nzero

    def genNRdownlink(self, **kwargs):
        """Method for generating 5G NR signal based on constellation points.

        Parameters
        ----------
        BW : integer
            Bandwidth of carrier
        BWP : array of certain structure
            Bandwidth part of corresponding carrier. [a,b,c,d]
            where a=numerology, b=number of OFDM symbols, c=first resource block of
            the BWP (N_BWP^start), d=number of resource blocks of the BWP
            (N_BWP^size), c + d <= number of resource blocks of the carrier.
        osr : integer
            Oversampling factor
        cnstl : array
            Array of constellation points
        Example
        -------
        self.genNRdownlink(BW=15e6, BWP=[2,7,0,18],osr=1, cnstl=[1-i*0.6,...,-0,6+i*0.3])

        """

        BW = kwargs.get("BW")
        BWP = kwargs.get("BWP")
        osr = kwargs.get("osr")
        cnstl = kwargs.get("cnstl")

        mu = BWP[0]
        dl = numerology.nr_parameters(self.FR, mu, BW, osr)
        self.BW_conf.append(dl["RB"] * 12 * 2**mu * 15e3)
        self.ACLR_BW.append(numerology.max_bw_config(self.FR, BW))
        N_ID_1 = 0  # physical-layer cell-identity group
        N_ID_2 = 0  # physical-layer identity within the group
        N_ID_cell = 3 * N_ID_1 + N_ID_2  # cell identity
        Nsymb = int(BWP[1])
        ncp = numerology.nr_cp_lengths(Nsymb, mu, dl)

        values, is_ref = grid.reference_grid(
            dl["RB"], Nsymb, mu, BWP[2], BWP[3], N_ID_cell, N_ID_2
        )
        mask = grid.data_re_mask(dl["RB"], Nsymb, BWP[2], BWP[3])
        RE = grid.map_data(values, mask, cnstl)

        s = ofdm.modulate(RE, dl["NFFT"], ncp, dl["Lroll"] * osr)
        out = {"s": s, "Fs": dl["Fs"]}

        return out

    def normalize(self, x, opt):
        """
        Return x scaled to a peak of one.

        Parameters
        ----------
        x : array
            Complex signal
        opt : string ("max", "amp")
            "max" scales max(|Re(x)|, |Im(x)|) to one, "amp" scales max(|x|) to one

        Returns
        -------
        Scaled signal.
        """
        if opt == "max":
            parts = np.concatenate((np.abs(x.real), np.abs(x.imag)))
            peak = parts.max()
        elif opt == "amp":
            peak = np.abs(x).max()
        else:
            raise ValueError(f'normalization must be "max" or "amp", got {opt}')
        return x / peak

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
            where a=numerology, b=number of OFDM symbols, c=first resource block of
            the BWP (N_BWP^start), d=number of resource blocks of the BWP
            (N_BWP^size), c + d <= number of resource blocks of the carrier.
        osr : integer
            Oversampling factor

        Example
        -------
        self.demNRdownlink(s=[0.3,...,0.7], BW=10e6,BWP=[1,14,0,24], osr=1)

        """

        s = kwargs.get("s")
        BW = kwargs.get("BW")
        BWP = kwargs.get("BWP")
        osr = kwargs.get("osr")

        equalize = self.equalizer
        mu = BWP[0]
        N_symb_TOT = int(BWP[1])

        # get various parameters related to input signal
        dl = numerology.nr_parameters(self.FR, mu, BW, osr)  # general NR parameters
        ncp = numerology.nr_cp_lengths(N_symb_TOT, mu, dl)
        starts = ofdm.symbol_starts(ncp, dl["NFFT"])
        sign = s
        N_sampl = len(s)
        N_slots = np.floor(N_symb_TOT / 14)  # number of slots (integer)
        N_ID_1 = 0  # physical-layer cell-identity group
        N_ID_2 = 0  # physical-layer identity within the group
        N_ID_cell = 3 * N_ID_1 + N_ID_2  # cell identity
        if N_symb_TOT == 0:
            print("ERROR")
            return 0
        l_pss = grid.pss_symbol(N_symb_TOT)
        invect_aligned = sync.pss_align(sign, dl["NFFT"], starts[l_pss] + ncp[l_pss], N_ID_2)

        RE = ofdm.demodulate(invect_aligned, dl["NFFT"], ncp, dl["Nsc"])
        # if equalization is deactivated, return at this point already
        if equalize == "off":
            return RE
        # re-create DMRS/PSS grid (i.e. post-FFT ideal reference signal)
        RE_id = np.ones((RE.shape), complex)
        values, is_ref = grid.reference_grid(
            dl["RB"], N_symb_TOT, mu, BWP[2], BWP[3], N_ID_cell, N_ID_2
        )
        RE_id[is_ref] = values[is_ref]
        pilots = grid.dmrs_mask(dl["RB"], N_symb_TOT, BWP[2], BWP[3])

        return equalizer.equalize(RE, RE_id, pilots)
