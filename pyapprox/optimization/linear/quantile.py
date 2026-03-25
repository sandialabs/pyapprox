"""Quantile regression solvers for basis expansion fitting.

This module provides solvers for quantile regression:
- QuantileRegressionSolver: Quantile regression via linear programming
"""

from typing import Generic, Optional

from pyapprox.optimization.linear.base import (
    LinearSystemSolver,
    SingleQoiSolverMixin,
)
from pyapprox.util.backends.protocols import Array, Backend


class QuantileRegressionSolver(
    SingleQoiSolverMixin, LinearSystemSolver[Array], Generic[Array]
):
    """Quantile regression solver via linear programming.

    Solves: min_c Σ_i ρ_τ(y_i - Φ_i c)

    where ρ_τ(u) = u(τ - I(u < 0)) is the check function (pinball loss).

    Uses linear programming reformulation.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    quantile : float
        Target quantile τ in [0, 1].
    options : dict, optional
        Options passed to scipy.optimize.linprog.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        quantile: float,
        options: Optional[dict[str, Any]] = None,
    ):
        super().__init__(bkd)
        self._validate_quantile(quantile)
        self._quantile = quantile
        self._options = options or {}

    def _validate_quantile(self, quantile: float) -> None:
        """Validate quantile is in [0, 1]."""
        if not 0 <= quantile <= 1:
            raise ValueError(f"Quantile must be in [0, 1], got {quantile}")

    def set_quantile(self, quantile: float) -> None:
        """Set target quantile.

        Parameters
        ----------
        quantile : float
            Quantile level in [0, 1].
        """
        self._validate_quantile(quantile)
        self._quantile = quantile

    def set_options(self, options: dict[str, Any]) -> None:
        """Set solver options.

        Parameters
        ----------
        options : dict
            Options for scipy.optimize.linprog.
        """
        self._options = options

    def _solve(self, basis_matrix: Array, values: Array) -> Array:
        """Solve quantile regression via linear programming.

        The quantile regression problem:
            min_c Σ_i ρ_τ(y_i - Φ_i c)

        is reformulated as LP:
            min_{c,u,v} τ 1^T u + (1-τ) 1^T v
            s.t. Φc + u - v = y
                 u, v >= 0

        where the residual r = y - Φc = v - u.

        Parameters
        ----------
        basis_matrix : Array
            Basis matrix Φ. Shape: (nsamples, nterms)
        values : Array
            Target values y. Shape: (nsamples, 1)

        Returns
        -------
        Array
            Coefficients c. Shape: (nterms, 1)
        """
        self._validate_single_qoi(values)

        import numpy as np
        from scipy.optimize import linprog
        # TODO: are lazy imports necessary here

        bkd = self._bkd
        A = bkd.to_numpy(basis_matrix)
        y = bkd.to_numpy(values)[:, 0]
        nsamples, nterms = A.shape
        tau = self._quantile

        # Decision variables: [c (nterms), u (nsamples), v (nsamples)]
        # Objective: 0*c + τ*u + (1-τ)*v
        c_obj = np.concatenate(
            [
                np.zeros(nterms),
                tau * np.ones(nsamples),
                (1 - tau) * np.ones(nsamples),
            ]
        )

        # Equality constraints: Φc + u - v = y
        # [A, I, -I] @ [c; u; v] = y
        A_eq = np.hstack(
            [
                A,
                np.eye(nsamples),
                -np.eye(nsamples),
            ]
        )
        b_eq = y

        # Bounds: c unbounded, u >= 0, v >= 0
        bounds = (
            [(None, None)] * nterms + [(0, None)] * nsamples + [(0, None)] * nsamples
        )

        result = linprog(
            c_obj,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options=self._options,
        )

        if not result.success:
            raise RuntimeError(f"Quantile regression failed: {result.message}")

        coef = result.x[:nterms]
        return bkd.asarray(coef.reshape(-1, 1))


class ExpectileRegressionSolver(
    SingleQoiSolverMixin, LinearSystemSolver[Array], Generic[Array]
):
    """Expectile regression solver via iteratively reweighted least squares.

    Solves: min_c Σ_i |τ - I(y_i < Φ_i c)| (y_i - Φ_i c)²

    Expectiles are to means what quantiles are to medians.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    expectile : float
        Target expectile τ in (0, 1).
    max_iter : int
        Maximum IRLS iterations. Default: 100.
    tol : float
        Convergence tolerance. Default: 1e-6.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        expectile: float,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        super().__init__(bkd)
        self._validate_expectile(expectile)
        self._expectile = expectile
        self._max_iter = max_iter
        self._tol = tol

    def _validate_expectile(self, expectile: float) -> None:
        """Validate expectile is in (0, 1)."""
        if not 0 < expectile < 1:
            raise ValueError(f"Expectile must be in (0, 1), got {expectile}")

    def set_expectile(self, expectile: float) -> None:
        """Set target expectile.

        Parameters
        ----------
        expectile : float
            Expectile level in (0, 1).
        """
        self._validate_expectile(expectile)
        self._expectile = expectile

    def _solve(self, basis_matrix: Array, values: Array) -> Array:
        """Solve expectile regression via IRLS.

        Parameters
        ----------
        basis_matrix : Array
            Basis matrix Φ. Shape: (nsamples, nterms)
        values : Array
            Target values y. Shape: (nsamples, 1)

        Returns
        -------
        Array
            Coefficients c. Shape: (nterms, 1)
        """
        self._validate_single_qoi(values)

        bkd = self._bkd
        A = basis_matrix
        y = values[:, 0]
        tau = self._expectile

        # Initialize with OLS
        coef = bkd.lstsq(A, values, rcond=None)[:, 0]

        for _ in range(self._max_iter):
            coef_old = bkd.copy(coef)

            # Compute residuals
            residual = y - bkd.dot(A, coef)

            # Compute asymmetric weights
            # w_i = τ if r_i >= 0 else (1-τ)
            weights = bkd.where(
                residual >= 0,
                tau * bkd.ones_like(residual),
                (1 - tau) * bkd.ones_like(residual),
            )

            # Weighted least squares
            sqrt_w = bkd.sqrt(weights)
            A_weighted = A * bkd.reshape(sqrt_w, (-1, 1))
            y_weighted = y * sqrt_w
            coef = bkd.lstsq(A_weighted, bkd.reshape(y_weighted, (-1, 1)), rcond=None)[
                :, 0
            ]

            # Check convergence
            if bkd.norm(coef - coef_old) < self._tol:
                break

        return bkd.reshape(coef, (-1, 1))
