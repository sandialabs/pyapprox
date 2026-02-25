"""Trigonometric basis expansion for 1D periodic functions.

Linear combination of trigonometric basis functions:
f(x) = sum_i c_i * phi_i(x)
where phi_i are [1, cos(kx), sin(kx)] on [-pi, pi].
"""

from typing import Generic, Optional

from pyapprox.surrogates.affine.univariate.trigonometric import (
    TrigonometricPolynomial1D,
)
from pyapprox.util.backends.protocols import Array, Backend


class TrigonometricExpansion(Generic[Array]):
    """Linear combination of trigonometric basis functions.

    Wraps a TrigonometricPolynomial1D basis and manages coefficients
    for evaluation.

    Parameters
    ----------
    basis : TrigonometricPolynomial1D[Array]
        The trigonometric basis.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        basis: TrigonometricPolynomial1D[Array],
        bkd: Backend[Array],
    ):
        self._basis = basis
        self._bkd = bkd
        self._nqoi = 1
        self._coef: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of input variables (always 1)."""
        return 1

    def nterms(self) -> int:
        """Return the number of basis terms."""
        return self._basis.nterms()

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        return self._nqoi

    def set_coefficients(self, coef: Array) -> None:
        """Set expansion coefficients.

        Parameters
        ----------
        coef : Array, shape (nterms, nqoi)
            Coefficients for each basis function.
        """
        if coef.shape[0] != self.nterms():
            raise ValueError(f"coef.shape[0]={coef.shape[0]} != nterms={self.nterms()}")
        self._coef = coef
        self._nqoi = coef.shape[1]

    def get_coefficients(self) -> Array:
        """Return the current expansion coefficients.

        Returns
        -------
        Array, shape (nterms, nqoi)
        """
        if self._coef is None:
            raise RuntimeError("Coefficients have not been set.")
        return self._coef

    def basis_matrix(self, samples: Array) -> Array:
        """Evaluate the basis matrix at given samples.

        Parameters
        ----------
        samples : Array, shape (1, nsamples)
            Sample points.

        Returns
        -------
        Array, shape (nsamples, nterms)
            Basis function values at each sample.
        """
        return self._basis(samples)

    def __call__(self, samples: Array) -> Array:
        """Evaluate the expansion at given samples.

        Parameters
        ----------
        samples : Array, shape (1, nsamples)
            Sample points.

        Returns
        -------
        Array, shape (nqoi, nsamples)
            Expansion values.
        """
        if self._coef is None:
            raise RuntimeError("Coefficients have not been set.")
        basis_vals = self._basis(samples)  # (nsamples, nterms)
        return (basis_vals @ self._coef).T  # (nqoi, nsamples)
