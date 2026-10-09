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

from . import analyzer
from . import carrier
from . import constellation
from . import generator
from . import multicarrier
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

        self.norm = "max"  # peak normalization, see generator.normalize
        self.backoff_db = 0.0  # peak back-off from one in dB, see generator.normalize

        self.signal_id = 0  # used for generating different seeds for different signals

        if len(arg) >= 1:
            parent = arg[0]
            self.copy_propval(parent, self.proplist)
            self.parent = parent

    def main_gen(self):
        bits = []
        for b in self.in_bits:
            if isinstance(b, str) and b == "max":
                bits.append(None)
            else:
                bits.append(b)
        x, self.cnstl = generator.generate(self.carriers, self.osr, self.tx_filter, bits, self.signal_id)
        self.gen_bits = [constellation.demodulate(s, c.qam) for c, s in zip(self.carriers, self.cnstl)]

        Fs = multicarrier.plan(self.carriers, self.osr)["Fs"]
        s = generator.normalize(x, self.norm, self.backoff_db)
        t = np.arange(len(s)) / Fs
        self.s_struct = {"s": np.transpose(np.vstack((t, np.real(s), np.imag(s)))), "Fs": Fs}
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

    def main_dem(self):
        # I and Q in the last two columns, after an optional time column
        data = self.IOS.Members["in_dem"].Data
        y = data[:, -2] + 1j * data[:, -1]
        self.dem_cnstl_vec, self.dem_bits, self.EVM = analyzer.analyze(
            y, self.cnstl, self.carriers, self.osr, self.rx_filter
        )

    def run_dem(self, *arg):
        if self.model == "py":
            self.main_dem()
