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


import numpy as np


def modulate(bits, Ncnstl, qam_type, carrier_id, signal_id):
    """Generate constellation points with the original bit preparation."""
    qam = np.array([], complex)
    if qam_type == "16QAM":
        M = 16

        if len(bits) > np.log2(M) * Ncnstl:
            raise Exception(
                "Not enough OFDM symbols. Max "
                + str(int(np.log2(M) * Ncnstl))
                + " bits or "
                + str(Ncnstl)
                + " constellation points"
            )

        if len(bits) != np.log2(M) * Ncnstl:
            bits = np.pad(
                bits, (0, int(np.log2(M) * Ncnstl - len(bits))), constant_values=0
            )
        for i in range(0, int(len(bits) / np.log2(M))):
            d = (
                1
                / np.sqrt(10)
                * (
                    (1 - 2 * bits[4 * i]) * (2 - (1 - 2 * bits[4 * i + 2]))
                    + 1j
                    * (1 - 2 * bits[4 * i + 1])
                    * (2 - (1 - 2 * bits[4 * i + 3]))
                )
            )
            qam = np.append(qam, d)

    elif qam_type == "4QAM":
        M = 4
        if len(bits) > np.log2(M) * Ncnstl:
            raise Exception(
                "Not enough OFDM symbols. Max "
                + str(int(np.log2(M) * Ncnstl))
                + " bits or "
                + str(Ncnstl)
                + " constellation points"
            )
        else:
            if len(bits) != np.log2(M) * Ncnstl:
                bits = np.pad(
                    bits,
                    (0, int(np.log2(M) * Ncnstl - len(bits))),
                    constant_values=0,
                )
            for i in range(0, int(len(bits) / np.log2(M))):
                d = (
                    1
                    / np.sqrt(2)
                    * ((1 - 2 * bits[2 * i]) + 1j * (1 - 2 * bits[2 * i + 1]))
                )
                qam = np.append(qam, d)

    elif qam_type == "BPSK":
        M = 2

        if len(bits) > np.log2(M) * Ncnstl:
            raise Exception(
                "Not enough OFDM symbols. Max "
                + str(int(np.log2(M) * Ncnstl))
                + " bits or "
                + str(Ncnstl)
                + " constellation points"
            )

        if len(bits) != np.log2(M) * Ncnstl:
            bits = np.pad(
                bits, (0, int(np.log2(M) * Ncnstl - len(bits))), constant_values=0
            )
        for i in range(0, int(len(bits) / np.log2(M))):
            d = 1 / np.sqrt(2) * ((1 - 2 * bits[i]) + 1j * (1 - 2 * bits[i]))
            qam = np.append(qam, d)

    elif qam_type == "256QAM":
        M = 256

        if len(bits) > np.log2(M) * Ncnstl:
            raise Exception(
                "Not enough OFDM symbols. Max "
                + str(int(np.log2(M) * Ncnstl))
                + " bits or "
                + str(Ncnstl)
                + " constellation points"
            )

        if len(bits) != np.log2(M) * Ncnstl:
            bits = np.pad(
                bits, (0, int(np.log2(M) * Ncnstl - len(bits))), constant_values=0
            )
        for i in range(0, int(len(bits) / np.log2(M))):
            d = (
                1
                / np.sqrt(170)
                * (
                    (1 - 2 * bits[8 * i])
                    * (
                        8
                        - (1 - 2 * bits[8 * i + 2])
                        * (
                            4
                            - (1 - 2 * bits[8 * i + 4])
                            * (2 - (1 - 2 * bits[8 * i + 6]))
                        )
                    )
                    + 1j
                    * (1 - 2 * bits[8 * i + 1])
                    * (
                        8
                        - (1 - 2 * bits[8 * i + 3])
                        * (
                            4
                            - (1 - 2 * bits[8 * i + 5])
                            * (2 - (1 - 2 * bits[8 * i + 7]))
                        )
                    )
                )
            )
            qam = np.append(qam, d)

    else:  # 64QAM
        M = 64

        if len(bits) > np.log2(M) * Ncnstl:
            raise Exception(
                "Not enough OFDM symbols. Max "
                + str(int(np.log2(M) * Ncnstl))
                + " bits or "
                + str(Ncnstl)
                + " constellation points"
            )

        if len(bits) != np.log2(M) * Ncnstl:
            bits = np.pad(
                bits, (0, int(np.log2(M) * Ncnstl - len(bits))), constant_values=0
            )
        for i in range(0, int(len(bits) / np.log2(M))):
            d = (
                1
                / np.sqrt(42)
                * (
                    (1 - 2 * bits[6 * i])
                    * (
                        4
                        - (1 - 2 * bits[6 * i + 2])
                        * (2 - (1 - 2 * bits[6 * i + 4]))
                    )
                    + 1j
                    * (1 - 2 * bits[6 * i + 1])
                    * (
                        4
                        - (1 - 2 * bits[6 * i + 3])
                        * (2 - (1 - 2 * bits[6 * i + 5]))
                    )
                )
            )
            qam = np.append(qam, d)

    return qam, bits


def demodulate(DataSymbols_nzero, qam_type):
    """Decode constellation points using the original nearest-point mapping."""
    bit_vect = []
    if qam_type == "16QAM":
        M = 16
        m = np.log2(M)
    elif qam_type == "4QAM":
        M = 4
        m = np.log2(M)
    elif qam_type == "BPSK":
        M = 2
        m = np.log2(M)
    elif qam_type == "256QAM":
        M = 256
        m = np.log2(M)
    else:  # 64QAM
        M = 64
        m = np.log2(M)
    points = []
    points_as_bits = []
    for i in range(0, M):
        bits = np.array(list(np.binary_repr(i, int(m))), dtype=int)
        if qam_type == "16QAM":
            d = (
                1
                / np.sqrt(10)
                * (
                    (1 - 2 * bits[0]) * (2 - (1 - 2 * bits[2]))
                    + 1j * (1 - 2 * bits[1]) * (2 - (1 - 2 * bits[3]))
                )
            )
        elif qam_type == "4QAM":
            d = 1 / np.sqrt(2) * ((1 - 2 * bits[0]) + 1j * (1 - 2 * bits[1]))

        elif qam_type == "BPSK":
            d = 1 / np.sqrt(2) * ((1 - 2 * bits[0]) + 1j * (1 - 2 * bits[0]))

        elif qam_type == "256QAM":
            d = (
                1
                / np.sqrt(170)
                * (
                    (1 - 2 * bits[0])
                    * (
                        8
                        - (1 - 2 * bits[2])
                        * (4 - (1 - 2 * bits[4]) * (2 - (1 - 2 * bits[6])))
                    )
                    + 1j
                    * (1 - 2 * bits[1])
                    * (
                        8
                        - (1 - 2 * bits[3])
                        * (4 - (1 - 2 * bits[5]) * (2 - (1 - 2 * bits[7])))
                    )
                )
            )

        else:  # 64QAM
            d = (
                1
                / np.sqrt(42)
                * (
                    (1 - 2 * bits[0])
                    * (4 - (1 - 2 * bits[2]) * (2 - (1 - 2 * bits[4])))
                    + 1j
                    * (1 - 2 * bits[1])
                    * (4 - (1 - 2 * bits[3]) * (2 - (1 - 2 * bits[5])))
                )
            )
        points.append(d)
        points_as_bits.append(bits)
    for j in range(0, len(DataSymbols_nzero)):
        index = np.argmin(np.abs(np.array(points) - DataSymbols_nzero[j]))
        bit_vect.extend(points_as_bits[index])
    return np.array(bit_vect)
