"""Basis state for time-evolving flow matching bases.

Holds recurrence coefficients for orthonormal polynomials at a single
time slice, with evaluation and derivative methods.
"""

from typing import Generic

from pyapprox.surrogates.affine.univariate.globalpoly.orthopoly_base import (
    evaluate_orthonormal_polynomial_1d,
    evaluate_orthonormal_polynomial_derivatives_1d,
)
from pyapprox.util.backends.protocols import Array, Backend


class StieltjesBasisState(Generic[Array]):
    """Recurrence coefficients and evaluator for one time slice.

    Parameters
    ----------
    rcoefs : Array
        Three-term recurrence coefficients, shape ``(nterms, 2)``.
        Column 0: alpha coefficients, column 1: beta coefficients.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, rcoefs: Array, bkd: Backend[Array]) -> None:
        self._rcoefs = rcoefs
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def rcoefs(self) -> Array:
        """Recurrence coefficients, shape ``(nterms, 2)``."""
        return self._rcoefs

    def n_basis(self) -> int:
        """Number of basis functions."""
        return self._rcoefs.shape[0]

    def eval(self, x: Array) -> Array:
        """Evaluate orthonormal polynomials at x.

        Parameters
        ----------
        x : Array
            Sample points, shape ``(1, nsamples)``.

        Returns
        -------
        Array
            Basis values, shape ``(nsamples, n_basis)``.
        """
        return evaluate_orthonormal_polynomial_1d(
            self._rcoefs, self._bkd, x
        )

    def eval_derivatives(self, x: Array, order: int = 1) -> Array:
        """Evaluate derivatives of orthonormal polynomials at x.

        Parameters
        ----------
        x : Array
            Sample points, shape ``(1, nsamples)``.
        order : int
            Derivative order (1 = first derivative).

        Returns
        -------
        Array
            Derivative values, shape ``(nsamples, n_basis)``.
        """
        return evaluate_orthonormal_polynomial_derivatives_1d(
            self._rcoefs, self._bkd, x, order
        )
