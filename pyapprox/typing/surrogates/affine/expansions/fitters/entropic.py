"""Entropic fitter for basis expansions.

The entropic loss function is the error measure from the Entropic Risk Quadrangle.
It is used to fit surrogates that minimize the entropic risk of residuals.

Unlike other risk measures, the entropic risk is not positively homogeneous,
meaning R[t*X] != t*R[X], so it cannot be used with conservative adjustment
patterns. Instead, it requires direct nonlinear optimization.
"""

from typing import Generic, Optional

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import BasisExpansionProtocol
from pyapprox.typing.surrogates.affine.expansions.fitters.results import (
    DirectSolverResult,
)
from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)


class EntropicLoss(Generic[Array]):
    """Entropic loss function for basis expansion regression.

    Minimizes the entropic error measure with strength parameter beta:
        L(coef) = sum_i w_i * (exp(beta*r_i) - beta*r_i - 1) / beta^2

    where r_i = y_i - Phi_i @ coef are residuals and w_i are quadrature
    weights (default: 1/nsamples for Monte Carlo).

    When beta=1, this simplifies to: sum_i w_i * (exp(r_i) - r_i - 1)

    This implements FunctionWithJacobianAndHVPProtocol for use with
    trust-region optimizers.

    Parameters
    ----------
    basis_matrix : Array
        Basis matrix from expansion. Shape: (nsamples, nterms)
    train_values : Array
        Target values. Shape: (nsamples, 1)
    bkd : Backend[Array]
        Computational backend.
    weights : Array, optional
        Quadrature weights. Shape: (nsamples, 1). Default: 1/nsamples.
    strength : float, optional
        Strength parameter beta. Default: 1.0.
    """

    def __init__(
        self,
        basis_matrix: Array,
        train_values: Array,
        bkd: Backend[Array],
        weights: Optional[Array] = None,
        strength: float = 1.0,
    ):
        self._bkd = bkd
        self._basis_mat = basis_matrix
        self._train_values = train_values
        self._beta = strength

        if weights is None:
            nsamples = basis_matrix.shape[0]
            self._weights = bkd.full((nsamples, 1), 1.0 / nsamples)
        else:
            if weights.shape != (basis_matrix.shape[0], 1):
                raise ValueError(
                    f"weights must have shape ({basis_matrix.shape[0]}, 1), "
                    f"got {weights.shape}"
                )
            self._weights = weights

        if train_values.shape != (basis_matrix.shape[0], 1):
            raise ValueError(
                f"train_values must have shape ({basis_matrix.shape[0]}, 1), "
                f"got {train_values.shape}"
            )

    def strength(self) -> float:
        """Return the strength parameter beta."""
        return self._beta

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of variables (coefficients)."""
        return int(self._basis_mat.shape[1])

    def nqoi(self) -> int:
        """Return number of quantities of interest (always 1)."""
        return 1

    def __call__(self, samples: Array) -> Array:
        """Evaluate entropic loss at coefficient values.

        Parameters
        ----------
        samples : Array
            Coefficient values. Shape: (nterms, nsamples_eval)

        Returns
        -------
        Array
            Loss values. Shape: (1, nsamples_eval)
        """
        # samples shape: (nterms, nsamples_eval)
        nsamples_eval = samples.shape[1]
        beta = self._beta
        results = []
        for ii in range(nsamples_eval):
            coef = samples[:, ii : ii + 1]  # (nterms, 1)
            pred_values = self._bkd.dot(self._basis_mat, coef)  # (nsamples, 1)
            residuals = self._train_values - pred_values  # (nsamples, 1)
            # Entropic error: sum w_i * (exp(beta*r_i) - beta*r_i - 1) / beta^2
            beta_r = beta * residuals
            loss = self._bkd.dot(
                (self._bkd.exp(beta_r) - beta_r - 1.0).T, self._weights
            ) / (beta * beta)  # (1, 1)
            results.append(loss[0, 0])
        return self._bkd.reshape(self._bkd.stack(results, axis=0), (1, -1))

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian (gradient) at a single coefficient point.

        Parameters
        ----------
        sample : Array
            Coefficient values. Shape: (nterms, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (1, nterms)
        """
        if sample.ndim != 2 or sample.shape[1] != 1:
            raise ValueError(
                f"sample must have shape (nvars, 1), got {sample.shape}"
            )
        beta = self._beta
        pred_values = self._bkd.dot(self._basis_mat, sample)  # (nsamples, 1)
        residuals = self._train_values - pred_values  # (nsamples, 1)
        beta_r = beta * residuals
        # d/d(coef) [(exp(beta*r) - beta*r - 1) / beta^2]
        # = (beta * exp(beta*r) * (-1) - beta * (-1)) / beta^2
        # = (-exp(beta*r) + 1) / beta
        # = (1 - exp(beta*r)) / beta
        # grad = Phi.T @ (w * (1 - exp(beta*r))) / beta
        grad = self._bkd.einsum(
            "i,ij->j",
            (self._weights * (1.0 - self._bkd.exp(beta_r)))[:, 0],
            self._basis_mat,
        ) / beta
        return self._bkd.reshape(grad, (1, -1))

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product at a single coefficient point.

        Parameters
        ----------
        sample : Array
            Coefficient values. Shape: (nterms, 1)
        vec : Array
            Direction vector. Shape: (nterms, 1)

        Returns
        -------
        Array
            Hessian-vector product. Shape: (nterms, 1)
        """
        if sample.ndim != 2 or sample.shape[1] != 1:
            raise ValueError(
                f"sample must have shape (nvars, 1), got {sample.shape}"
            )
        if vec.ndim != 2 or vec.shape[1] != 1:
            raise ValueError(
                f"vec must have shape (nvars, 1), got {vec.shape}"
            )
        beta = self._beta
        pred_values = self._bkd.dot(self._basis_mat, sample)  # (nsamples, 1)
        residuals = self._train_values - pred_values  # (nsamples, 1)
        beta_r = beta * residuals
        # d^2/d(coef)^2 [(exp(beta*r) - beta*r - 1) / beta^2]
        # Second derivative of (1 - exp(beta*r)) / beta w.r.t. coef
        # = beta * exp(beta*r) * Phi.T @ Phi / beta = exp(beta*r) * Phi.T @ Phi
        # But we need to account for the chain rule correctly
        # Hessian: H = Phi.T @ diag(w * exp(beta*r)) @ Phi
        # HVP: H @ v = Phi.T @ (w * exp(beta*r) * (Phi @ v))
        Phi_v = self._bkd.dot(self._basis_mat, vec)  # (nsamples, 1)
        weighted = self._weights * self._bkd.exp(beta_r) * Phi_v
        return self._bkd.dot(self._basis_mat.T, weighted)  # (nterms, 1)


class EntropicFitter(Generic[Array]):
    """Entropic fitter for basis expansions.

    Fits expansion coefficients by minimizing the entropic error measure,
    which corresponds to the error from the Entropic Risk Quadrangle.

    At the optimum, the entropic risk statistic of the residuals equals zero.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    strength : float, optional
        Strength parameter beta for the entropic loss. Default: 1.0.
    gtol : float, optional
        Gradient tolerance for optimizer. Default: 1e-8.
    maxiter : int, optional
        Maximum iterations for optimizer. Default: 1000.
    verbosity : int, optional
        Verbosity level for optimizer. Default: 0.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        strength: float = 1.0,
        gtol: float = 1e-8,
        maxiter: int = 1000,
        verbosity: int = 0,
    ):
        self._bkd = bkd
        self._strength = strength
        self._gtol = gtol
        self._maxiter = maxiter
        self._verbosity = verbosity

    def strength(self) -> float:
        """Return the strength parameter beta."""
        return self._strength

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
        weights: Optional[Array] = None,
    ) -> DirectSolverResult[Array, BasisExpansionProtocol[Array]]:
        """Fit via entropic loss minimization.

        Parameters
        ----------
        expansion : BasisExpansionProtocol
            Must have basis_matrix() and with_params() methods.
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        values : Array
            Target values. Shape: (nqoi, nsamples) or (nsamples,) for nqoi=1.
        weights : Array, optional
            Quadrature weights. Shape: (nsamples, 1). Default: uniform.

        Returns
        -------
        DirectSolverResult
            Result containing fitted expansion.

        Raises
        ------
        ValueError
            If nqoi > 1 (entropic fitting only supports single QoI).
        """
        # Handle 1D values
        if values.ndim == 1:
            values = self._bkd.reshape(values, (1, -1))

        nqoi = values.shape[0]
        if nqoi != 1:
            raise ValueError(
                f"EntropicFitter only supports nqoi=1, got nqoi={nqoi}"
            )

        # Get basis matrix: (nsamples, nterms)
        Phi = expansion.basis_matrix(samples)
        nterms = Phi.shape[1]

        # Prepare train values: (nsamples, 1)
        train_values = values.T  # (nsamples, 1)

        # Create loss function
        loss = EntropicLoss(Phi, train_values, self._bkd, weights, self._strength)

        # Create optimizer
        optimizer = ScipyTrustConstrOptimizer[Array](
            verbosity=self._verbosity,
            maxiter=self._maxiter,
            gtol=self._gtol,
        )

        # Set up bounds (unbounded coefficients)
        bounds = self._bkd.stack(
            (
                self._bkd.full((nterms,), -np.inf),
                self._bkd.full((nterms,), np.inf),
            ),
            axis=1,
        )

        # Bind and minimize
        optimizer.bind(loss, bounds)
        init_guess = self._bkd.ones((nterms, 1))
        result = optimizer.minimize(init_guess)

        # Extract coefficients: (nterms, 1)
        params = result.optima()

        # Create fitted expansion
        fitted_expansion = expansion.with_params(params)

        return DirectSolverResult(
            surrogate=fitted_expansion,
            params=params,
        )
