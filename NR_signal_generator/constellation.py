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


BITS_PER_SYMBOL = {
    "BPSK": 1,
    "QPSK": 2,
    "16QAM": 4,
    "64QAM": 6,
    "256QAM": 8,
    "1024QAM": 10,
}


def _constellation(modulation):
    """
    Return NR constellation points and their corresponding bit labels.

    points: complex array of shape (M,)
    labels: integer array of shape (M, bits_per_symbol)

    Labels are MSB-first, in ascending binary order.
    The complete constellation has unit average symbol energy.

    Implements TS 38.211 sections 5.1.2 through 5.1.7.
    Symbol-dependent pi/2-BPSK is not supported.
    """
    if not isinstance(modulation, str) or modulation not in BITS_PER_SYMBOL:
        raise ValueError(f"Unsupported modulation: {modulation!r}")

    m = BITS_PER_SYMBOL[modulation]
    M = 1 << m

    # Signed integers ensure 1 - 2*b produces +1 or -1 without wrapping.
    indices = np.arange(M, dtype=np.int64)
    shifts = np.arange(m - 1, -1, -1)
    labels = (indices[:, None] >> shifts) & 1
    signs = 1 - 2 * labels

    if modulation == "BPSK":
        # NR BPSK uses the diagonal constellation, not the real axis.
        axis = signs[:, 0]
        points = (axis + 1j * axis) / np.sqrt(2.0)
    else:
        # Even bit positions define I, odd positions define Q.
        # Evaluate the nested NR PAM mapping from the innermost bits outward.
        amplitude = np.ones((M, 2), dtype=np.int64)
        for offset in range(m - 2, 0, -2):
            amplitude = (1 << ((m - offset) // 2)) - signs[:, offset : offset + 2] * amplitude

        iq = signs[:, :2] * amplitude

        # Squared normalization: 2, 10, 42, 170, 682.
        normalization = 2 * (M - 1) // 3
        points = (iq[:, 0] + 1j * iq[:, 1]) / np.sqrt(normalization)

    return points, labels


def modulate(bits, modulation):
    """
    Map bits to NR constellation symbols.
    """
    points, labels = _constellation(modulation)
    m = labels.shape[1]
    bits = np.asarray(bits)

    if bits.ndim != 1 or not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must be a one-dimensional sequence of 0 and 1")
    if bits.size % m:
        raise ValueError(f"bit count must be divisible by {m}")

    groups = bits.astype(np.int64).reshape(-1, m)
    weights = 1 << np.arange(m - 1, -1, -1)
    return points[groups @ weights]


def demodulate(symbols, modulation):
    """
    Map equalized NR constellation symbols to bits.
    """
    points, labels = _constellation(modulation)
    symbols = np.asarray(symbols, dtype=np.complex128)

    if symbols.ndim != 1 or not np.all(np.isfinite(symbols)):
        raise ValueError("symbols must be a one-dimensional finite sequence")

    distances = np.abs(symbols[:, None] - points[None, :])
    nearest = np.argmin(distances, axis=1)
    return labels[nearest].reshape(-1)
