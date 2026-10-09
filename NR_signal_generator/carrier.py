from dataclasses import dataclass
from functools import cached_property

import numpy as np

from . import constellation
from . import grid
from . import numerology


@dataclass(frozen=True)
class Carrier:
    """
    One NR downlink carrier with a single bandwidth part.

    Parameters
    ----------
    fr : string ("FR1", "FR2-1")
        Frequency range
    bw : float
        Channel bandwidth in Hz
    mu : integer
        Numerology
    duration : float
        Signal duration in seconds, a multiple of a half subframe (0.5 ms);
        see numerology.duration
    qam : string
        Modulation of the data resource elements
    offset : float
        Carrier centre frequency relative to the signal centre in Hz
    bwp_start : integer
        First resource block of the BWP, N_BWP^start
    bwp_size : integer or None
        Number of resource blocks of the BWP, N_BWP^size, None for the rest of
        the carrier from bwp_start
    n_id_cell : integer
        Physical-layer cell identity
    """

    fr: str
    bw: float
    mu: int
    duration: float
    qam: str
    offset: float = 0.0
    bwp_start: int = 0
    bwp_size: int = None
    n_id_cell: int = 0

    def __post_init__(self):
        half_subframes = self.duration / 0.5e-3
        if half_subframes < 1 or not np.isclose(half_subframes, np.round(half_subframes), rtol=0, atol=1e-9):
            raise ValueError(f"duration {self.duration * 1e3:g} ms is not a multiple of 0.5 ms")
        if self.qam not in constellation.BITS_PER_SYMBOL:
            raise ValueError(f"unsupported modulation {self.qam!r}")
        if self.bwp_size is None:
            object.__setattr__(self, "bwp_size", self.n_rb - self.bwp_start)
        # Building the data mask checks fr, mu, bw and the BWP
        self.data_mask

    def parameters(self, osr=1):
        """
        Return the NR parameters of the carrier, as numerology.nr_parameters.

        Parameters
        ----------
        osr : integer
            Oversampling factor

        Returns
        -------
        Dictionary with keys Fs, NFFT, Ncp1, Ncp2 (scaled by osr), RB, Nsc and Lroll.
        """
        return numerology.nr_parameters(self.fr, self.mu, self.bw, osr)

    def cp_lengths(self, osr=1):
        """
        Return the cyclic prefix length of each OFDM symbol.

        Parameters
        ----------
        osr : integer
            Oversampling factor

        Returns
        -------
        Array of n_symb cyclic prefix lengths in samples.
        """
        return numerology.nr_cp_lengths(self.n_symb, self.mu, self.parameters(osr))

    @cached_property
    def n_symb(self):
        """Number of OFDM symbols, 7 * 2^mu per half subframe."""
        return 7 * 2**self.mu * int(np.round(self.duration / 0.5e-3))

    @cached_property
    def n_rb(self):
        """Number of resource blocks of the carrier."""
        return self.parameters()["RB"]

    @cached_property
    def n_id_2(self):
        """Physical-layer identity within the cell-identity group, carried by the PSS."""
        return self.n_id_cell % 3

    @cached_property
    def data_mask(self):
        """Data resource elements, as grid.data_re_mask."""
        return grid.data_re_mask(self.n_rb, self.n_symb, self.bwp_start, self.bwp_size)

    @cached_property
    def pilot_mask(self):
        """DMRS resource elements, as grid.dmrs_mask."""
        return grid.dmrs_mask(self.n_rb, self.n_symb, self.bwp_start, self.bwp_size)

    @cached_property
    def reference_grid(self):
        """DMRS and PSS values and positions, as grid.reference_grid."""
        return grid.reference_grid(
            self.n_rb, self.n_symb, self.mu, self.bwp_start, self.bwp_size, self.n_id_cell, self.n_id_2
        )

    @cached_property
    def n_bits(self):
        """Number of bits that fill the data resource elements."""
        return int(self.data_mask.sum()) * constellation.BITS_PER_SYMBOL[self.qam]

    @cached_property
    def bw_config(self):
        """Transmission bandwidth configuration N_RB * 12 * SCS in Hz."""
        return self.n_rb * 12 * 2**self.mu * 15e3

    @cached_property
    def aclr_bw(self):
        """ACLR measurement bandwidth in Hz, as numerology.max_bw_config."""
        return numerology.max_bw_config(self.fr, self.bw)
