import numpy as np


def gold_sequence(c_init, Mpn):
    """Pseudo-random sequence c(n), 0 <= n < Mpn (TS 38.211, Section 5.2.1).

        c(n) = (x1(n + Nc) + x2(n + Nc)) mod 2,  Nc = 1600
        x1(n + 31) = (x1(n + 3) + x1(n)) mod 2,  x1(0) = 1, x1(1..30) = 0
        x2(n + 31) = (x2(n + 3) + x2(n + 2) + x2(n + 1) + x2(n)) mod 2
        c_init = sum_i x2(i) 2^i

    Parameters
    ----------
    c_init : integer
        Initial value used for sequence generation
    Mpn : integer
        Second initial value that sets length of the sequence
    """
    Nc = 1600
    x1 = np.zeros(Mpn + Nc)
    x2 = np.zeros(Mpn + Nc)
    x1[0] = 1
    c_init_bit = np.binary_repr(int(c_init), 31)
    for i in range(0, 31):
        x2[i] = float(c_init_bit[30 - i])
    # calculate m-sequences x1 and x2
    for i in range(0, Mpn + Nc - 31):
        x1[i + 31] = (x1[i + 3] + x1[i]) % 2
        x2[i + 31] = (x2[i + 3] + x2[i + 2] + x2[i + 1] + x2[i]) % 2
    c = np.zeros(Mpn)
    # calculate pseudo-random sequence
    for n in range(0, Mpn):
        c[n] = (x1[n + Nc] + x2[n + Nc]) % 2
    return c


def pss(n_id2):
    """Primary synchronization signal d_PSS(n), 0 <= n < 127 (TS 38.211, Section 7.4.2.2.1).

        d_PSS(n) = 1 - 2 x(m),  m = (n + 43 N_ID^(2)) mod 127
        x(i + 7) = (x(i + 4) + x(i)) mod 2
        [x(6) ... x(0)] = [1 1 1 0 1 1 0]

    Parameters
    ----------
    n_id2 : integer
        Second cell ID defined for 5G NR, N_ID^(2) in {0, 1, 2}
    """
    if n_id2 not in (0, 1, 2):
        raise ValueError(f"N_ID_2 must be 0, 1 or 2, got {n_id2}")
    x = np.zeros(127, dtype=int)
    x[:7] = [0, 1, 1, 0, 1, 1, 1]
    for i in range(120):
        x[i + 7] = (x[i + 4] + x[i]) % 2
    return np.roll(1.0 - 2.0 * x, -43 * n_id2)


def dmrs(N_ID_cell, Nsymb, RB):
    """Demodulation Reference Signal r(m) for each OFDM symbol (TS 38.211, Section 7.4.1.1.1).

        r(m) = (1 - 2 c(2m)) / sqrt(2) + j (1 - 2 c(2m + 1)) / sqrt(2)
        c_init = (2^17 (14 n_s + l + 1)(2 N_ID + 1) + 2 N_ID) mod 2^31

    Parameters
    ----------
    N_ID_cell : integer
       Cell ID
    Nsymb : integer
        Number of OFDM symbols
    RB : integer
        Number of used Resource Blocks

    Returns
    -------
    Complex array of shape (6 * RB, Nsymb).
    """
    N_RB = int(RB * 12 / 2)
    r = np.zeros((N_RB, int(Nsymb)), complex)
    for i in range(0, int(Nsymb)):
        # get slot number & symbol number within the slot
        ns = np.floor(i / 14)
        l = i % 14

        c_init = np.mod(
            (2**17 * (14 * ns + l + 1) * (2 * N_ID_cell + 1) + 2 * N_ID_cell),
            2**31,
        )
        c = gold_sequence(c_init, 2 * N_RB)
        # create reference-signal sequence
        ones = np.ones(int(len(c) / 2))
        a = np.reshape(np.array(c), (-1, 2))
        r[:, i] = np.add(
            (1 / (np.sqrt(2))) * np.subtract(ones, 2 * a[:, 0]),
            1j * (1 / (np.sqrt(2))) * np.subtract(ones, 2 * a[:, 1]),
        )

    return r
