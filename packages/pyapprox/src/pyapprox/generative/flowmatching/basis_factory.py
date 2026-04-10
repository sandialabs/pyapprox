"""Basis factory for time-evolving flow matching bases.

Builds StieltjesBasisState from weighted 1D samples using the
Lanczos algorithm for computing orthonormal polynomial recurrence
coefficients.
"""

from typing import Generic

from pyapprox.generative.flowmatching.basis_state import StieltjesBasisState
from pyapprox.surrogates.affine.univariate.globalpoly.numeric import (
    lanczos_recursion,
)
from pyapprox.util.backends.protocols import Array, Backend


class StieltjesBasisFactory(Generic[Array]):
    """Builds StieltjesBasisState from weighted 1D samples.

    Parameters
    ----------
    nterms : int
        Number of orthonormal polynomial terms to compute.
    bkd : Backend[Array]
        Computational backend.
    ortho_tol : float
        Tolerance for Lanczos orthonormality check.
    """

    def __init__(
        self,
        nterms: int,
        bkd: Backend[Array],
        ortho_tol: float = 1e-10,
    ) -> None:
        self._nterms = nterms
        self._bkd = bkd
        self._ortho_tol = ortho_tol

    def nterms(self) -> int:
        """Number of basis terms produced."""
        return self._nterms

    def build(
        self, samples_1d: Array, weights: Array
    ) -> StieltjesBasisState[Array]:
        """Build basis state from 1D samples and weights.

        Parameters
        ----------
        samples_1d : Array
            Sample points, shape ``(n,)``.
        weights : Array
            Quadrature weights, shape ``(n,)``. Need not sum to 1.

        Returns
        -------
        StieltjesBasisState
            Basis state with computed recurrence coefficients.
        """
        rcoefs = lanczos_recursion(
            samples_1d,
            weights,
            self._nterms,
            self._bkd,
            self._ortho_tol,
        )
        return StieltjesBasisState(rcoefs, self._bkd)
