"""Analytical marginalization of Polynomial Chaos Expansions.

Exploits PCE orthonormality: E[psi_j(x_k)] = delta_{j,0}, so only
terms where marginalized variables have degree 0 survive. This gives
exact marginalization without quadrature.
"""

from typing import Generic, List

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.affine.expansions.pce import (
    PolynomialChaosExpansion,
)
from pyapprox.surrogates.affine.basis import (
    OrthonormalPolynomialBasis,
)
from pyapprox.interface.functions.marginalize import ReducedFunction


class PCEDimensionReducer(Generic[Array]):
    """Analytically marginalizes a PCE using multi-index structure.

    Satisfies ``DimensionReducerProtocol``.

    For a PCE f(x) = sum_i c_i psi_i(x), marginalizing out variable x_k
    uses E[psi_j(x_k)] = delta_{j,0} (orthonormality). Only terms where
    all marginalized variables have degree 0 survive, and the surviving
    terms form a lower-dimensional PCE in the kept variables.

    Parameters
    ----------
    pce : PolynomialChaosExpansion[Array]
        Fitted PCE with coefficients set.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        pce: PolynomialChaosExpansion[Array],
        bkd: Backend[Array],
    ):
        self._pce = pce
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables in the original PCE."""
        return self._pce.nvars()

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        return self._pce.nqoi()

    def reduce(
        self, keep_indices: List[int]
    ) -> ReducedFunction[Array]:
        """Reduce to the specified variables via analytical marginalization.

        Parameters
        ----------
        keep_indices : List[int]
            Indices of variables to keep (0-based).

        Returns
        -------
        ReducedFunction[Array]
            A function with ``nvars = len(keep_indices)`` and the same
            ``nqoi``.
        """
        reduced_pce = self.reduce_pce(keep_indices)
        return ReducedFunction(
            len(keep_indices), self.nqoi(), reduced_pce.__call__, self._bkd
        )

    def reduce_pce(
        self, keep_indices: List[int]
    ) -> PolynomialChaosExpansion[Array]:
        """Build a reduced PCE by marginalizing out non-kept variables.

        Parameters
        ----------
        keep_indices : List[int]
            Indices of variables to keep (0-based).

        Returns
        -------
        PolynomialChaosExpansion[Array]
            A new PCE in the kept variables only, with surviving
            coefficients.
        """
        bkd = self._bkd
        pce = self._pce
        indices_arr = pce.get_indices()  # (nvars, nterms)
        coef = pce.get_coefficients()  # (nterms, nqoi)
        nvars = pce.nvars()

        marginalize_indices = sorted(
            set(range(nvars)) - set(keep_indices)
        )

        if not marginalize_indices:
            return pce

        # Use numpy for multi-index filtering (integer metadata, not
        # part of autograd graph)
        indices_np = bkd.to_numpy(indices_arr)

        # Find surviving terms: all marginalized dims must have degree 0
        surviving_mask = np.ones(indices_np.shape[1], dtype=bool)
        for dim in marginalize_indices:
            surviving_mask &= (indices_np[dim, :] == 0)

        surviving_idx = np.where(surviving_mask)[0]

        # Sub-indices for kept dims only: (n_keep, n_surviving)
        sub_indices_np = indices_np[np.array(keep_indices), :][
            :, surviving_idx
        ]
        sub_indices = bkd.asarray(sub_indices_np, dtype=bkd.int64_dtype())

        # Sub-coefficients: (n_surviving, nqoi)
        sub_coef = bkd.zeros((len(surviving_idx), pce.nqoi()))
        for jj, term_idx in enumerate(surviving_idx):
            sub_coef[jj, :] = coef[term_idx, :]

        # Build sub-basis from kept dimensions' 1d bases
        basis = pce._basis
        sub_bases_1d = [basis._bases_1d[i] for i in keep_indices]
        sub_basis = OrthonormalPolynomialBasis(sub_bases_1d, bkd, sub_indices)

        # Build new PCE with sub-basis and sub-coefficients
        sub_pce = PolynomialChaosExpansion(
            sub_basis, bkd, nqoi=pce.nqoi()
        )
        sub_pce.set_coefficients(sub_coef)

        return sub_pce
