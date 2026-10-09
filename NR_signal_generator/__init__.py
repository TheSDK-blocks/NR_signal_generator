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

from . import carrier
from . import constellation
from . import equalizer
from . import filters
from . import grid
from . import measurements
from . import multicarrier
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
            "carriers",
            "osr",
            "in_bits",
            "include_time_vector",
        ]

        self.IOS = Bundle()
        self.IOS.Members["in_dem"] = IO()  # Pointer for input data

        self.IOS.Members["out"] = IO()  # Pointer for output data

        self.carriers = [carrier.Carrier("FR1", 100e6, 1, 0.5e-3, "64QAM")]
        self.osr = 1
        self.in_bits = np.array(["max"])

        self.seed = 0
        self.include_time_vector = 0

        self.model = "py"
        # Can be set externally, but is not propagated
        self.par = False  # By default, no parallel processingi
        self.queue = []  # By default, no parallel processing
        self.IOS.Members["control_write"] = IO()
        self.tx_filter = True  # transmit channel filter in the generator
        self.rx_filter = True  # receive channel-select filter in the analyzer

        self.norm = (
            "max"  # max = normalize I & Q separately, amp = normalize amplitude to one, None = volts
        )

        self.signal_id = 0  # used for generating different seeds for different signals

        if len(arg) >= 1:
            parent = arg[0]
            self.copy_propval(parent, self.proplist)
            self.parent = parent

    def main_gen(self):
        self.cnstl, self.gen_bits = self.genMultiQAM(self.carriers)
        self.s_struct = self.genMultiNRdownlink(self.carriers)
        if self.include_time_vector == 1:
            self.IOS.Members["out"].Data = self.s_struct[
                "s"
            ]  # output signal, matrix with  columns time, I signal, Q signal
        else:
            self.IOS.Members["out"].Data = np.transpose(
                np.vstack((self.s_struct["s"][:, 1], self.s_struct["s"][:, 2]))
            )

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
        self.dem = self.demMultiNRdownlink(
            self.carriers, car_return=car_return, sig_offset=sig_offset, NR_car_id=NR_car_id
        )
        self.dem_bits, self.dem_cnstl_vec = self.MultiQAMtoBit(self.carriers, NR_car_id=NR_car_id)

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

    def MultiQAMtoBit(self, carriers, **kwargs):
        """Method for calculate binary data based on recieved constellation points for multiple carriers.


        Example
        -------
        self.MultiQAMtoBit()

        """
        NR_car_id = kwargs.get("NR_car_id", -1)

        dem = self.dem
        bits = []
        dem_cnstl_vec = []
        for i, c in enumerate(carriers):
            if NR_car_id == -1 or i == NR_car_id:
                temp1, temp2 = self.QAMtoBit(dem[i], c)
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

        cnstl = self.cnstl

        EVM = []
        # measure EVM separately for each constellation
        for i in range(0, len(cnstl)):
            if NR_car_id == -1 or i == NR_car_id:
                # reference: the transmitted symbols, in the order they are mapped
                EVM.append(measurements.evm(cnstl[i], self.dem_cnstl_vec[i]))
            else:
                EVM.append(np.array([]))
        return EVM

    def demMultiNRdownlink(self, carriers, **kwargs):
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
        plan = multicarrier.plan(carriers, self.osr)
        Fs = plan["Fs"]
        cnstl = []
        # t=np.arange(0,sign.size)/Fs # initialize time vector (for mixing)
        if sign.shape[1] == 2:
            s = sign[:, 0] + 1j * sign[:, 1]
            t = np.arange(0, len(s)) / Fs
        else:
            t = sign[:, 0]
            s = sign[:, 1] + 1j * sign[:, 2]
        # demodulate carriers
        for i, c in enumerate(carriers):
            if NR_car_id == -1 or i == NR_car_id:

                # mix current carrier so that it is centered at 0 Hz
                if car_return == False:
                    v_mixed = s * np.exp(-1j * 2 * np.pi * c.offset * t)
                else:
                    carrier_offset = c.offset + sig_offset
                    v_mixed = s * np.exp(-1j * 2 * np.pi * carrier_offset * t)

                v_filt = v_mixed
                if self.rx_filter:
                    v_filt = filters.nr_filter(v_mixed, Fs, c.bw, c.fr)

                if car_return == True:
                    t2 = np.arange(0, len(v_filt)) / Fs
                    v_filt = v_filt * np.exp(+1j * 2 * np.pi * carrier_offset * t2)

                a = self.demNRdownlink(v_filt, c, plan["osr"][i])
                cnstl.append(a)
            else:
                cnstl.append(np.array([]))
        return cnstl

    def genMultiNRdownlink(self, carriers):
        """Method for generating signal from constellation points for multiple carriers.


        Example
        -------
        self.genMultiNRdownlink()

        """

        cnstl = self.cnstl
        plan = multicarrier.plan(carriers, self.osr)
        Fs = plan["Fs"]
        osr = plan["osr"]

        smatrix = np.zeros((plan["length"], len(carriers)), complex)
        # generate carriers
        for i, c in enumerate(carriers):
            s = self.genNRdownlink(c, osr[i], cnstl[i])
            self.testvar = s
            if self.tx_filter:
                s = filters.nr_filter(s, Fs, c.bw, c.fr)
            smatrix[:, i] = s * np.sqrt(c.mean_square / np.mean(np.abs(s) ** 2))

        s_raw = np.zeros(len(smatrix), complex)

        # mix carriers to proper frequency offset
        t_vect = np.arange(0, len(smatrix)) / Fs
        for i, c in enumerate(carriers):
            s_raw = s_raw + smatrix[:, i] * np.exp(1j * 2 * np.pi * c.offset * t_vect)
        s = self.normalize(s_raw, self.norm)
        output_format = np.transpose(np.vstack((t_vect, np.real(s), np.imag(s))))
        out = {
            "s": output_format,
            "Fs": Fs,
        }

        return out

    def genMultiQAM(self, carriers):
        """
        Return the data symbols and bits of each carrier.

        Parameters
        ----------
        carriers : list of carrier.Carrier
            Carriers of the signal

        Returns
        -------
        Data symbols and bits of each carrier.
        """
        cnstl = []
        gen_bits = []
        for i, c in enumerate(carriers):
            a, bit = self.genQAM(i, self.in_bits[i], c)
            cnstl.append(a)
            gen_bits.append(bit)
        return cnstl, gen_bits

    def genQAM(self, carrier_id, bits, c):
        """
        Return the data symbols of a carrier and the bits they carry.

        Parameters
        ----------
        carrier_id : integer
            Index of the carrier
        bits : array of binary values
           Binary input data that exactly fill every data resource element of
           the BWP, or "max" for pseudorandom bits. "max" stands in for MAC
           padding and scrambling and is seeded from carrier_id and signal_id,
           so it is reproducible. User data shorter than the allocation must be
           padded by the caller, e.g. with pseudorandom bits.
        c : carrier.Carrier
            Carrier

        Returns
        -------
        Data symbols, and the bits as an array of one row per symbol.
        """
        m = constellation.BITS_PER_SYMBOL[c.qam]
        if isinstance(bits, str) and bits == "max":
            seed = (carrier_id + self.signal_id * 13 + 1) * 123
            rng = np.random.RandomState(seed)
            bits = rng.randint(2, size=c.n_bits)
        elif len(bits) != c.n_bits:
            raise ValueError(
                f"carrier {carrier_id}: {len(bits)} bits do not fill the {c.n_bits}-bit allocation"
            )

        return constellation.modulate(bits, c.qam), np.reshape(bits, (-1, m))

    def QAMtoBit(self, cnstl, c):
        """
        Return the bits and data symbols of an equalized resource-element grid.

        Parameters
        ----------
        cnstl : array
            Equalized resource-element grid of the carrier
        c : carrier.Carrier
            Carrier

        Returns
        -------
        Demapped bits and data symbols.
        """
        data = grid.data_symbols(cnstl, c.data_mask)
        return constellation.demodulate(data, c.qam), data

    def genNRdownlink(self, c, osr, cnstl):
        """
        Return the OFDM signal of a carrier.

        Parameters
        ----------
        c : carrier.Carrier
            Carrier
        osr : integer
            Oversampling factor
        cnstl : array
            Data symbols of the carrier

        Returns
        -------
        Complex baseband signal.
        """
        dl = c.parameters(osr)
        values, is_ref = c.reference_grid
        RE = grid.map_data(values, c.data_mask, cnstl)
        return ofdm.modulate(RE, dl["NFFT"], c.cp_lengths(osr), dl["Lroll"] * osr)

    def normalize(self, x, opt):
        """
        Return x scaled to a peak of one, or unscaled.

        Parameters
        ----------
        x : array
            Complex signal
        opt : string ("max", "amp") or None
            "max" scales max(|Re(x)|, |Im(x)|) to one, "amp" scales max(|x|) to
            one, None returns x unscaled, in volts as set by the carrier powers

        Returns
        -------
        Scaled signal.
        """
        if opt is None:
            return x
        if opt == "max":
            parts = np.concatenate((np.abs(x.real), np.abs(x.imag)))
            peak = parts.max()
        elif opt == "amp":
            peak = np.abs(x).max()
        else:
            raise ValueError(f'normalization must be "max" or "amp", got {opt}')
        return x / peak

    def demNRdownlink(self, s, c, osr):
        """
        Return the equalized resource-element grid of a carrier.

        Parameters
        ----------
        s : array
            One period of the carrier signal at baseband
        c : carrier.Carrier
            Carrier
        osr : integer
            Oversampling factor

        Returns
        -------
        Equalized resource-element grid.
        """
        dl = c.parameters(osr)
        ncp = c.cp_lengths(osr)
        starts = ofdm.symbol_starts(ncp, dl["NFFT"])
        l_pss = grid.pss_symbol(c.n_symb)
        aligned = sync.pss_align(s, dl["NFFT"], starts[l_pss] + ncp[l_pss], c.n_id_2)
        RE = ofdm.demodulate(aligned, dl["NFFT"], ncp, dl["Nsc"])

        values, is_ref = c.reference_grid
        RE_id = np.ones(RE.shape, complex)
        RE_id[is_ref] = values[is_ref]
        return equalizer.equalize(RE, RE_id, c.pilot_mask)
