"""
Cholesky-based correlation matrix parameterization.

Parameterizes a correlation matrix via its Cholesky factor L where
Sigma = L L^T. The strict lower-triangular elements of L are free
parameters, while the diagonal is derived from the unit-diagonal
constraint: L_ii = sqrt(1 - sum_{j<i} L_ij^2).
"""

from typing import Generic

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import (
    HyperParameter,
    HyperParameterList,
)


class CholeskyCorrelationParameterization(Generic[Array]):
    """
    Correlation matrix parameterized by a Cholesky factor.

    For a d-dimensional correlation matrix, there are d*(d-1)/2 free
    parameters: the strict lower-triangular elements of L. The diagonal
    is derived from the constraint diag(L L^T) = 1.

    Parameters
    ----------
    chol_lower_values : Array
        Strict lower-triangular elements of L, stored row-by-row.
        Shape: (d*(d-1)/2,)
        For d=3, the elements are [L_{10}, L_{20}, L_{21}].
    nvars : int
        Dimension of the correlation matrix.
    bkd : Backend[Array]
        Computational backend.

    Raises
    ------
    ValueError
        If chol_lower_values has wrong length for the given nvars.
    """

    def __init__(
        self,
        chol_lower_values: Array,
        nvars: int,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._nvars = nvars
        expected_nparams = nvars * (nvars - 1) // 2

        if chol_lower_values.shape != (expected_nparams,):
            raise ValueError(
                f"chol_lower_values must have shape ({expected_nparams},), "
                f"got {chol_lower_values.shape}"
            )

        self._hyp = HyperParameter(
            name="chol_lower",
            nparams=expected_nparams,
            values=chol_lower_values,
            bounds=(-1.0, 1.0),
            bkd=bkd,
        )
        self._hyp_list = HyperParameterList([self._hyp])

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the dimension of the correlation matrix."""
        return self._nvars

    def nparams(self) -> int:
        """Return the number of free parameters."""
        return self._hyp.nparams()

    def hyp_list(self) -> HyperParameterList:
        """Return the hyperparameter list for optimization."""
        return self._hyp_list

    def _build_cholesky_factor(self) -> Array:
        """
        Build the full Cholesky factor L from free parameters.

        The diagonal is derived: L_ii = sqrt(1 - sum_{j<i} L_ij^2).

        Returns
        -------
        Array
            Lower triangular Cholesky factor. Shape: (nvars, nvars)
        """
        d = self._nvars
        L = self._bkd.zeros((d, d))
        values = self._hyp.get_values()

        idx = 0
        for i in range(d):
            # Fill strict lower-triangular elements
            for j in range(i):
                L[i, j] = values[idx]
                idx += 1
            # Derive diagonal from unit-diagonal constraint
            row_sum = self._bkd.sum(L[i, :i] ** 2)
            L[i, i] = self._bkd.sqrt(self._bkd.clip(1.0 - row_sum, 1e-15, 1.0))

        return L

    def correlation_matrix(self) -> Array:
        """
        Compute the full correlation matrix Sigma = L L^T.

        Returns
        -------
        Array
            Correlation matrix. Shape: (nvars, nvars)
        """
        L = self._build_cholesky_factor()
        return L @ L.T

    def log_det(self) -> Array:
        """
        Compute log|Sigma| = 2 * sum(log(L_ii)).

        Returns
        -------
        Array
            Log determinant as a scalar Array (preserves autograd).
        """
        L = self._build_cholesky_factor()
        diag_L = self._bkd.get_diagonal(L)
        return 2.0 * self._bkd.sum(self._bkd.log(diag_L))

    def quad_form(self, z: Array) -> Array:
        """
        Compute z^T (Sigma^{-1} - I) z for each sample column.

        Uses the identity: z^T Sigma^{-1} z = ||L^{-1} z||^2
        so quad_form = ||L^{-1} z||^2 - ||z||^2.

        Parameters
        ----------
        z : Array
            Standard normal samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Quadratic form values. Shape: (nsamples,)
        """
        if z.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got {z.ndim}D"
            )
        if z.shape[0] != self._nvars:
            raise ValueError(
                f"Expected {self._nvars} variables, got {z.shape[0]}"
            )

        L = self._build_cholesky_factor()
        # w = L^{-1} z via forward substitution
        w = self._bkd.solve_triangular(L, z, lower=True)
        # ||w||^2 - ||z||^2 per column
        return self._bkd.sum(w ** 2, axis=0) - self._bkd.sum(z ** 2, axis=0)

    def sample_transform(self, eps: Array) -> Array:
        """
        Map independent standard normals to correlated samples: z = L @ eps.

        Parameters
        ----------
        eps : Array
            Independent standard normal samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Correlated samples. Shape: (nvars, nsamples)
        """
        if eps.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got {eps.ndim}D"
            )
        if eps.shape[0] != self._nvars:
            raise ValueError(
                f"Expected {self._nvars} variables, got {eps.shape[0]}"
            )

        L = self._build_cholesky_factor()
        return L @ eps

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CholeskyCorrelationParameterization("
            f"nvars={self._nvars}, nparams={self.nparams()})"
        )
