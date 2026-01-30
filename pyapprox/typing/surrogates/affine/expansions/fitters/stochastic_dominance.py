"""Stochastic dominance fitters for basis expansions.

This module provides fitters that ensure the fitted surrogate satisfies
stochastic dominance constraints relative to the training data:

- FSDFitter: First-order Stochastic Dominance
  Ensures P(f(X) <= eta) <= P(Y <= eta) for all eta

- SSDFitter: Second-order Stochastic Dominance
  Ensures E[max(0, eta - f(X))] <= E[max(0, eta - Y)] for all eta

Both use smooth approximations to make the non-differentiable indicator
functions usable with gradient-based optimization.
"""

from typing import Generic, Optional

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import BasisExpansionProtocol
from pyapprox.typing.surrogates.affine.expansions.fitters.results import (
    DirectSolverResult,
)
from pyapprox.typing.surrogates.affine.expansions.fitters.least_squares import (
    LeastSquaresFitter,
)
from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.typing.optimization.minimize.differentiable_approximations import (
    SmoothLogBasedMaxFunction,
    SmoothLogBasedLeftHeavisideFunction,
    DifferentiableApproximationBase,
)


class FSDObjective(Generic[Array]):
    """Objective function for stochastic dominance regression.

    Minimizes weighted least squares:
        L(coef) = 0.5 * sum_i w_i * (y_i - Phi_i @ coef)^2

    This implements FunctionWithJacobianAndHVPProtocol.

    Parameters
    ----------
    basis_matrix : Array
        Basis matrix. Shape: (nsamples, nterms)
    train_values : Array
        Target values. Shape: (nsamples, 1)
    bkd : Backend[Array]
        Computational backend.
    weights : Array, optional
        Sample weights. Shape: (nsamples, 1). Default: uniform 1/nsamples.
    """

    def __init__(
        self,
        basis_matrix: Array,
        train_values: Array,
        bkd: Backend[Array],
        weights: Optional[Array] = None,
    ):
        self._bkd = bkd
        self._basis_mat = basis_matrix
        self._train_values = train_values

        nsamples = basis_matrix.shape[0]
        if weights is None:
            self._weights = bkd.full((nsamples, 1), 1.0 / nsamples)
        else:
            self._weights = weights

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
        """Evaluate objective at coefficient values.

        Parameters
        ----------
        samples : Array
            Coefficient values. Shape: (nterms, nsamples_eval)

        Returns
        -------
        Array
            Objective values. Shape: (1, nsamples_eval)
        """
        bkd = self._bkd
        nsamples_eval = samples.shape[1]
        results = []
        for ii in range(nsamples_eval):
            coef = samples[:, ii : ii + 1]  # (nterms, 1)
            pred = bkd.dot(self._basis_mat, coef)  # (nsamples, 1)
            residuals = self._train_values - pred  # (nsamples, 1)
            val = 0.5 * bkd.sum(self._weights * residuals**2)
            results.append(val)
        return bkd.reshape(bkd.stack(results, axis=0), (1, -1))

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
        bkd = self._bkd
        pred = bkd.dot(self._basis_mat, sample)  # (nsamples, 1)
        residuals = self._train_values - pred  # (nsamples, 1)
        # d/d(coef) [0.5 * sum w * (y - Phi @ coef)^2]
        # = -Phi.T @ (w * (y - Phi @ coef))
        grad = -bkd.dot(self._basis_mat.T, self._weights * residuals)  # (nterms, 1)
        return grad.T  # (1, nterms)

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
        bkd = self._bkd
        # Hessian: H = Phi.T @ diag(w) @ Phi
        # HVP: H @ v = Phi.T @ (w * (Phi @ v))
        Phi_v = bkd.dot(self._basis_mat, vec)  # (nsamples, 1)
        return bkd.dot(self._basis_mat.T, self._weights * Phi_v)  # (nterms, 1)


class StochasticDominanceConstraint(Generic[Array]):
    """Nonlinear constraint for stochastic dominance.

    For FSD (using left Heaviside):
        c(coef) = sum_m w_m * [H(f_m - f_n) - H(y_m - f_n)] <= 0 for all n

    For SSD (using smooth max):
        c(coef) = sum_m w_m * [max(0, f_n - f_m) - max(0, f_n - y_m)] >= 0 for all n

    This implements NonlinearConstraintProtocolWithJacobianAndWHVP.

    Parameters
    ----------
    basis_matrix : Array
        Basis matrix. Shape: (nsamples, nterms)
    train_values : Array
        Target values. Shape: (nsamples, 1)
    smooth_function : DifferentiableApproximationBase
        Smooth approximation (Heaviside for FSD, max for SSD).
    bkd : Backend[Array]
        Computational backend.
    weights : Array, optional
        Sample weights. Shape: (nsamples, 1). Default: uniform 1/nsamples.
    constraint_indices : Array, optional
        Indices for constraint evaluation. Shape: (nconstraints,).
        Default: all samples.
    lb : Array, optional
        Lower bounds. Default: -inf for FSD, 0 for SSD.
    ub : Array, optional
        Upper bounds. Default: 0 for FSD, inf for SSD.
    """

    def __init__(
        self,
        basis_matrix: Array,
        train_values: Array,
        smooth_function: DifferentiableApproximationBase[Array],
        bkd: Backend[Array],
        weights: Optional[Array] = None,
        constraint_indices: Optional[Array] = None,
        lb: Optional[Array] = None,
        ub: Optional[Array] = None,
    ):
        self._bkd = bkd
        self._basis_mat = basis_matrix
        self._train_values = train_values
        self._smooth_func = smooth_function

        nsamples = basis_matrix.shape[0]
        if weights is None:
            self._weights = bkd.full((nsamples, 1), 1.0 / nsamples)
        else:
            self._weights = weights

        if constraint_indices is None:
            self._constraint_indices = bkd.arange(nsamples)
        else:
            self._constraint_indices = constraint_indices

        nconstraints = int(self._constraint_indices.shape[0])
        if lb is None:
            self._lb = bkd.full((nconstraints,), -np.inf)
        else:
            self._lb = lb
        if ub is None:
            self._ub = bkd.zeros((nconstraints,))
        else:
            self._ub = ub

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of variables (coefficients)."""
        return int(self._basis_mat.shape[1])

    def nqoi(self) -> int:
        """Return number of constraints."""
        return int(self._constraint_indices.shape[0])

    def lb(self) -> Array:
        """Return lower bounds."""
        return self._lb

    def ub(self) -> Array:
        """Return upper bounds."""
        return self._ub

    def __call__(self, samples: Array) -> Array:
        """Evaluate constraints at coefficient values.

        Parameters
        ----------
        samples : Array
            Coefficient values. Shape: (nterms, 1)

        Returns
        -------
        Array
            Constraint values. Shape: (nconstraints, 1)
        """
        bkd = self._bkd
        coef = samples  # (nterms, 1)
        surrogate_values = bkd.dot(self._basis_mat, coef)  # (nsamples, 1)

        # f_n values at constraint indices
        f_n = surrogate_values[self._constraint_indices]  # (nconstraints, 1)

        # Compute h(f_m - f_n) for all m and constraint indices n
        # surrogate_values: (nsamples, 1), f_n.T: (1, nconstraints)
        # diff: (nsamples, nconstraints)
        diff_surrogate = surrogate_values - f_n.T
        tmp1 = self._smooth_func(diff_surrogate.T)  # (nconstraints, nsamples)

        # Compute h(y_m - f_n)
        diff_data = self._train_values - f_n.T  # (nsamples, nconstraints)
        tmp2 = self._smooth_func(diff_data.T)  # (nconstraints, nsamples)

        # Weighted sum over samples: shape (nconstraints, 1)
        val = bkd.dot(tmp1 - tmp2, self._weights)
        return val

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian of constraints.

        Parameters
        ----------
        sample : Array
            Coefficient values. Shape: (nterms, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (nconstraints, nterms)
        """
        bkd = self._bkd
        coef = sample
        surrogate_values = bkd.dot(self._basis_mat, coef)  # (nsamples, 1)
        f_n = surrogate_values[self._constraint_indices]  # (nconstraints, 1)

        # Compute h'(f_m - f_n) * w_m
        diff_surrogate = surrogate_values - f_n.T  # (nsamples, nconstraints)
        hder1 = (
            self._smooth_func.first_derivative(diff_surrogate.T)  # (ncons, nsamp)
            * self._weights.T  # (1, nsamples)
        )  # (nconstraints, nsamples)

        # Gradient of surrogate w.r.t. coefficients
        # surrogate_jac[m, :] = Phi[m, :] is (nsamples, nterms)
        surrogate_jac = self._basis_mat  # (nsamples, nterms)

        # fder1[c, m, :] = grad_coef(f_m) - grad_coef(f_n[c])
        # = Phi[m, :] - Phi[constraint_indices[c], :]
        # Shape: (nconstraints, nsamples, nterms)
        fder1 = (
            surrogate_jac[None, :, :]  # (1, nsamples, nterms)
            - surrogate_jac[self._constraint_indices, None, :]  # (ncons, 1, nterms)
        )

        # con_jac[c, d] = sum_m hder1[c, m] * fder1[c, m, d]
        con_jac = bkd.einsum("cn,cnd->cd", hder1, fder1)

        # Second term: h'(y_m - f_n) * (-grad_coef(f_n))
        diff_data = self._train_values - f_n.T  # (nsamples, nconstraints)
        hder2 = (
            self._smooth_func.first_derivative(diff_data.T)  # (ncons, nsamp)
            * self._weights.T
        )

        # fder2[c, m, :] = 0 - Phi[constraint_indices[c], :]
        fder2 = -surrogate_jac[self._constraint_indices, None, :]  # (ncons, 1, nterms)
        # Broadcast to (nconstraints, nsamples, nterms)
        fder2 = bkd.zeros((self.nqoi(), self._basis_mat.shape[0], self.nvars())) + fder2

        con_jac -= bkd.einsum("cn,cnd->cd", hder2, fder2)

        return con_jac  # (nconstraints, nterms)

    def whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        """Compute weighted Hessian-vector product.

        Parameters
        ----------
        sample : Array
            Coefficient values. Shape: (nterms, 1)
        vec : Array
            Direction vector. Shape: (nterms, 1)
        weights : Array
            Lagrange multipliers. Shape: (nconstraints, 1)

        Returns
        -------
        Array
            Weighted HVP. Shape: (nterms, 1)
        """
        bkd = self._bkd
        coef = sample
        surrogate_values = bkd.dot(self._basis_mat, coef)  # (nsamples, 1)
        f_n = surrogate_values[self._constraint_indices]  # (nconstraints, 1)
        surrogate_jac = self._basis_mat  # (nsamples, nterms)

        # h''(f_m - f_n) term
        diff_surrogate = surrogate_values - f_n.T  # (nsamples, nconstraints)
        hder2_surr = self._smooth_func.second_derivative(diff_surrogate.T)  # (ncons, nsamp)

        # fder1: (nconstraints, nsamples, nterms)
        fder1 = (
            surrogate_jac[None, :, :]
            - surrogate_jac[self._constraint_indices, None, :]
        )

        # Compute Phi @ v for each sample
        Phi_v = bkd.dot(self._basis_mat, vec)  # (nsamples, 1)
        Phi_n_v = Phi_v[self._constraint_indices]  # (nconstraints, 1)

        # fder1 @ v: (nconstraints, nsamples)
        fder1_v = Phi_v.T - Phi_n_v  # (nconstraints, nsamples)

        # Weight by sample weights and Lagrange multipliers
        # weighted: (nconstraints, nsamples)
        weighted = hder2_surr * self._weights.T * weights * fder1_v

        # Accumulate: sum over constraints and samples
        # hvp = sum_c sum_m weighted[c, m] * fder1[c, m, :]
        hvp = bkd.einsum("cn,cnd->d", weighted, fder1)

        # h''(y_m - f_n) term
        diff_data = self._train_values - f_n.T
        hder2_data = self._smooth_func.second_derivative(diff_data.T)

        # For this term, fder2 only depends on f_n
        # fder2_v = -Phi_n @ v: (nconstraints, 1)
        fder2_v = -Phi_n_v  # (nconstraints, 1)

        # weighted2: (nconstraints, nsamples)
        weighted2 = hder2_data * self._weights.T * weights * fder2_v

        # fder2[c, m, :] = -Phi[constraint_indices[c], :] for all m
        hvp -= bkd.einsum(
            "cn,cd->d",
            weighted2,
            -surrogate_jac[self._constraint_indices, :],
        )

        return bkd.reshape(hvp, (-1, 1))


class FSDFitter(Generic[Array]):
    """First-order Stochastic Dominance (FSD) fitter.

    Fits a surrogate such that P(f(X) <= eta) <= P(Y <= eta) for all eta,
    where f is the surrogate and Y is the training data.

    This ensures the surrogate's CDF is dominated by the data's CDF.

    Only supports nqoi=1 (single quantity of interest).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    eps : float, optional
        Smoothing parameter for Heaviside approximation. Default: 0.5.
    shift : float, optional
        Shift parameter for Heaviside approximation. Default: 0.0.
        Can improve numerical stability near the transition region.
    verbosity : int, optional
        Verbosity level for optimizer. Default: 0.
    maxiter : int, optional
        Maximum iterations. Default: 1000.
    gtol : float, optional
        Gradient tolerance. Default: 1e-8.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        eps: float = 0.5,
        shift: float = 0.0,
        verbosity: int = 0,
        maxiter: int = 1000,
        gtol: float = 1e-8,
    ):
        self._bkd = bkd
        self._eps = eps
        self._shift = shift
        self._verbosity = verbosity
        self._maxiter = maxiter
        self._gtol = gtol

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def eps(self) -> float:
        """Return smoothing parameter."""
        return self._eps

    def shift(self) -> float:
        """Return shift parameter."""
        return self._shift

    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
        weights: Optional[Array] = None,
    ) -> DirectSolverResult[Array, BasisExpansionProtocol[Array]]:
        """Fit with FSD constraint.

        Parameters
        ----------
        expansion : BasisExpansionProtocol
            Must have basis_matrix() and with_params() methods.
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        values : Array
            Target values. Shape: (1, nsamples) or (nsamples,).
        weights : Array, optional
            Sample weights. Shape: (nsamples, 1). Default: uniform.

        Returns
        -------
        DirectSolverResult
            Result containing fitted expansion.
        """
        bkd = self._bkd

        # Handle 1D values
        if values.ndim == 1:
            values = bkd.reshape(values, (1, -1))

        if values.shape[0] != 1:
            raise ValueError(
                f"FSDFitter only supports nqoi=1, got {values.shape[0]}"
            )

        # Get basis matrix: (nsamples, nterms)
        Phi = expansion.basis_matrix(samples)
        nsamples, nterms = Phi.shape

        # Prepare training values: (nsamples, 1)
        train_values = values.T

        # Prepare weights
        if weights is None:
            weights = bkd.full((nsamples, 1), 1.0 / nsamples)

        # Compute initial guess using conservative least squares
        lstsq = LeastSquaresFitter(bkd)
        init_result = lstsq.fit(expansion, samples, values)
        init_coef = bkd.copy(init_result.params())
        # Shift to satisfy constraints initially
        residuals = train_values - bkd.dot(Phi, init_coef)
        shift = bkd.max(residuals)
        init_coef[0, 0] = init_coef[0, 0] + shift

        # Create smooth Heaviside function
        smooth_heaviside = SmoothLogBasedLeftHeavisideFunction(
            bkd, self._eps, shift=self._shift
        )

        # Create objective and constraint
        objective = FSDObjective(Phi, train_values, bkd, weights)
        constraint = StochasticDominanceConstraint(
            Phi,
            train_values,
            smooth_heaviside,
            bkd,
            weights,
            lb=bkd.full((nsamples,), -np.inf),
            ub=bkd.zeros((nsamples,)),
        )

        # Set up optimizer
        optimizer = ScipyTrustConstrOptimizer[Array](
            verbosity=self._verbosity,
            maxiter=self._maxiter,
            gtol=self._gtol,
        )

        # Set bounds (unbounded coefficients)
        bounds = bkd.stack(
            (
                bkd.full((nterms,), -np.inf),
                bkd.full((nterms,), np.inf),
            ),
            axis=1,
        )

        # Bind and minimize
        optimizer.bind(objective, bounds, constraints=[constraint])
        result = optimizer.minimize(init_coef)

        # Extract coefficients
        params = result.optima()

        # Create fitted expansion
        fitted_expansion = expansion.with_params(params)

        return DirectSolverResult(
            surrogate=fitted_expansion,
            params=params,
        )


class SSDFitter(Generic[Array]):
    """Second-order Stochastic Dominance (SSD) fitter.

    Fits a surrogate such that E[max(0, eta - f(X))] <= E[max(0, eta - Y)]
    for all eta, where f is the surrogate and Y is the training data.

    This is a weaker condition than FSD but still provides a risk-averse
    approximation.

    Only supports nqoi=1 (single quantity of interest).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    eps : float, optional
        Smoothing parameter for max approximation. Default: 0.5.
    shift : float, optional
        Shift parameter for max approximation. Default: 0.0.
        Can improve numerical stability near the transition region.
    verbosity : int, optional
        Verbosity level for optimizer. Default: 0.
    maxiter : int, optional
        Maximum iterations. Default: 1000.
    gtol : float, optional
        Gradient tolerance. Default: 1e-8.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        eps: float = 0.5,
        shift: float = 0.0,
        verbosity: int = 0,
        maxiter: int = 1000,
        gtol: float = 1e-8,
    ):
        self._bkd = bkd
        self._eps = eps
        self._shift = shift
        self._verbosity = verbosity
        self._maxiter = maxiter
        self._gtol = gtol

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def eps(self) -> float:
        """Return smoothing parameter."""
        return self._eps

    def shift(self) -> float:
        """Return shift parameter."""
        return self._shift

    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
        weights: Optional[Array] = None,
    ) -> DirectSolverResult[Array, BasisExpansionProtocol[Array]]:
        """Fit with SSD constraint.

        Parameters
        ----------
        expansion : BasisExpansionProtocol
            Must have basis_matrix() and with_params() methods.
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        values : Array
            Target values. Shape: (1, nsamples) or (nsamples,).
        weights : Array, optional
            Sample weights. Shape: (nsamples, 1). Default: uniform.

        Returns
        -------
        DirectSolverResult
            Result containing fitted expansion.
        """
        bkd = self._bkd

        # Handle 1D values
        if values.ndim == 1:
            values = bkd.reshape(values, (1, -1))

        if values.shape[0] != 1:
            raise ValueError(
                f"SSDFitter only supports nqoi=1, got {values.shape[0]}"
            )

        # Get basis matrix: (nsamples, nterms)
        Phi = expansion.basis_matrix(samples)
        nsamples, nterms = Phi.shape

        # Prepare training values: (nsamples, 1)
        train_values = values.T

        # Prepare weights
        if weights is None:
            weights = bkd.full((nsamples, 1), 1.0 / nsamples)

        # Compute initial guess using conservative least squares
        lstsq = LeastSquaresFitter(bkd)
        init_result = lstsq.fit(expansion, samples, values)
        init_coef = bkd.copy(init_result.params())
        # Shift to satisfy constraints initially
        residuals = train_values - bkd.dot(Phi, init_coef)
        shift = bkd.max(residuals)
        init_coef[0, 0] = init_coef[0, 0] + shift

        # Create smooth max function
        smooth_max = SmoothLogBasedMaxFunction(bkd, self._eps, shift=self._shift)

        # Create objective and constraint
        # For SSD, constraints are >= 0, so bounds are (0, inf)
        objective = FSDObjective(Phi, train_values, bkd, weights)
        constraint = StochasticDominanceConstraint(
            Phi,
            train_values,
            smooth_max,
            bkd,
            weights,
            lb=bkd.zeros((nsamples,)),
            ub=bkd.full((nsamples,), np.inf),
        )

        # Set up optimizer
        optimizer = ScipyTrustConstrOptimizer[Array](
            verbosity=self._verbosity,
            maxiter=self._maxiter,
            gtol=self._gtol,
        )

        # Set bounds (unbounded coefficients)
        bounds = bkd.stack(
            (
                bkd.full((nterms,), -np.inf),
                bkd.full((nterms,), np.inf),
            ),
            axis=1,
        )

        # Bind and minimize
        optimizer.bind(objective, bounds, constraints=[constraint])
        result = optimizer.minimize(init_coef)

        # Extract coefficients
        params = result.optima()

        # Create fitted expansion
        fitted_expansion = expansion.with_params(params)

        return DirectSolverResult(
            surrogate=fitted_expansion,
            params=params,
        )
