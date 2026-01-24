"""
Gaussian Process statistics module for computing moments.

This module provides the GaussianProcessStatistics class which computes
statistical quantities from fitted Gaussian Processes:

- E[mu_f]: Mean of the GP mean (expected value of posterior mean)
- Var[mu_f]: Variance of the GP mean (uncertainty in posterior mean)
- E[gamma_f]: Mean of the GP variance (expected posterior variance)

These statistics quantify uncertainty in GP predictions integrated over
the input space according to the input distribution.
"""

from typing import Generic
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.gaussianprocess.protocols import (
    PredictiveGPProtocol,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.protocols import (
    KernelIntegralCalculatorProtocol,
)


class GaussianProcessStatistics(Generic[Array]):
    """
    Compute statistical quantities from a fitted Gaussian Process.

    This class provides methods to compute integrated statistics of GP
    predictions over the input space. These are useful for:

    - Uncertainty quantification: How much uncertainty remains?
    - Design of experiments: Which regions need more samples?
    - Sensitivity analysis: Which inputs matter most?

    Mathematical Background
    -----------------------
    For a GP f with posterior mean mu(x) and variance gamma(x), this class
    computes:

    1. Mean of mean: eta = E[mu(X)] = tau^T A^{-1} y
       where tau_i = int C(x, x^(i)) rho(x) dx

    2. Variance of mean: Var[mu(X)] = s^2 * varsigma^2
       where varsigma^2 = u - tau^T A^{-1} tau
       and u = int int C(x, z) rho(x) rho(z) dx dz

    3. Mean of variance: E[gamma(X)] = zeta + s^2 * v^2 - eta^2 - s^2 * varsigma^2
       where zeta = y^T A^{-1} P A^{-1} y
       and v^2 = 1 - Tr[P A^{-1}]

    Here:
    - A = K(X, X) + nugget * I is the noisy kernel matrix
    - s^2 is the kernel variance hyperparameter
    - rho(x) is the input probability density

    Parameters
    ----------
    gp : PredictiveGPProtocol[Array]
        A fitted Gaussian Process.
    integral_calculator : KernelIntegralCalculatorProtocol[Array]
        Calculator for kernel integrals (tau, P, u, etc.).

    Raises
    ------
    RuntimeError
        If the GP has not been fitted.

    Examples
    --------
    >>> from pyapprox.typing.surrogates.gaussianprocess.statistics import (
    ...     SeparableKernelIntegralCalculator,
    ...     GaussianProcessStatistics,
    ... )
    >>> # Assume gp is a fitted GP with separable kernel
    >>> # and marginals is a list of marginal distributions
    >>> calc = SeparableKernelIntegralCalculator(gp, marginals, nquad_points=50, bkd=bkd)
    >>> stats = GaussianProcessStatistics(gp, calc)
    >>> mean_of_mean = stats.mean_of_mean()
    >>> var_of_mean = stats.variance_of_mean()
    >>> mean_of_var = stats.mean_of_variance()
    """

    def __init__(
        self,
        gp: PredictiveGPProtocol[Array],
        integral_calculator: KernelIntegralCalculatorProtocol[Array],
    ):
        # Validate GP is fitted
        if not gp.is_fitted():
            raise RuntimeError(
                "GP must be fitted before computing statistics. "
                "Call gp.fit(X_train, y_train) first."
            )

        self._gp = gp
        self._calc = integral_calculator
        self._bkd = gp.bkd()

        # Get training data
        self._y = gp._data.y()  # Shape: (n_train, nqoi)
        self._n_train = gp._data.n_samples()

        # Get the Cholesky factor for efficient solves
        # A^{-1} v = L^{-T} L^{-1} v
        self._cholesky = gp._cholesky

        # Cache computed quantities
        self._cache: dict = {}

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def _solve(self, b: Array) -> Array:
        """
        Solve A x = b using the precomputed Cholesky factor.

        Parameters
        ----------
        b : Array
            Right-hand side, shape (n_train,) or (n_train, k).

        Returns
        -------
        x : Array
            Solution x = A^{-1} b, same shape as b.
        """
        return self._cholesky.solve(b)

    def _get_kernel_variance(self) -> Array:
        """
        Get the kernel variance hyperparameter (s^2).

        For standard kernels, this is typically 1.0 (built into length scale).
        For kernels with explicit variance, extract it from hyperparameters.

        Returns
        -------
        s2 : Array
            Kernel variance (scalar).
        """
        # Most kernels in pyapprox.typing have unit variance by default
        # The variance is absorbed into the kernel evaluation
        # For now, assume s^2 = 1.0 (this is correct for SE kernels
        # where the variance is 1 and length scales control correlation)
        #
        # TODO: Support explicit variance hyperparameter for kernels that have it
        return self._bkd.asarray(1.0)

    def mean_of_mean(self) -> Array:
        """
        Compute the expected value of the GP posterior mean.

        eta = E[mu(X)] = tau^T A^{-1} y

        This is the mean prediction integrated over the input space.

        Returns
        -------
        Array
            Scalar or shape (nqoi,) for multi-output GPs.
        """
        if 'mean_of_mean' in self._cache:
            return self._cache['mean_of_mean']

        tau = self._calc.tau()  # Shape: (n_train,)

        # Compute A^{-1} y (already computed in GP as alpha)
        alpha = self._gp._alpha  # Shape: (n_train, nqoi)

        # eta = tau^T * alpha
        # For single output: tau shape (n_train,), alpha shape (n_train, 1)
        # Result: (1,) or scalar
        eta = tau @ alpha  # Shape: (nqoi,)

        # Squeeze to scalar if nqoi=1
        if eta.shape[0] == 1:
            eta = eta[0]

        self._cache['mean_of_mean'] = eta
        return eta

    def variance_of_mean(self) -> Array:
        """
        Compute the variance of the GP posterior mean.

        Var[mu(X)] = s^2 * varsigma^2
        where varsigma^2 = u - tau^T A^{-1} tau

        This quantifies uncertainty in the mean prediction due to limited data.

        Returns
        -------
        Array
            Scalar (non-negative).
        """
        if 'variance_of_mean' in self._cache:
            return self._cache['variance_of_mean']

        tau = self._calc.tau()  # Shape: (n_train,)
        u = self._calc.u()      # Scalar

        # Compute A^{-1} tau
        # Need to reshape tau to (n_train, 1) for solve
        tau_col = self._bkd.reshape(tau, (-1, 1))
        A_inv_tau = self._solve(tau_col)  # Shape: (n_train, 1)
        A_inv_tau = self._bkd.reshape(A_inv_tau, (-1,))  # Shape: (n_train,)

        # varsigma^2 = u - tau^T A^{-1} tau
        varsigma_sq = u - tau @ A_inv_tau

        # Var[mu] = s^2 * varsigma^2
        s2 = self._get_kernel_variance()
        var_of_mean = s2 * varsigma_sq

        # Ensure non-negative (numerical stability)
        var_of_mean = var_of_mean * (var_of_mean >= 0.0)

        self._cache['variance_of_mean'] = var_of_mean
        return var_of_mean

    def mean_of_variance(self) -> Array:
        """
        Compute the expected value of the GP posterior variance.

        E[gamma(X)] = zeta + s^2 * v^2 - eta^2 - s^2 * varsigma^2

        where:
        - zeta = y^T A^{-1} P A^{-1} y
        - v^2 = 1 - Tr[P A^{-1}]
        - eta = E[mu(X)] (mean of mean)
        - varsigma^2 = u - tau^T A^{-1} tau

        This quantifies the expected prediction uncertainty over the input space.

        Returns
        -------
        Array
            Scalar (non-negative).
        """
        if 'mean_of_variance' in self._cache:
            return self._cache['mean_of_variance']

        # Get required quantities
        P = self._calc.P()      # Shape: (n_train, n_train)
        tau = self._calc.tau()  # Shape: (n_train,)
        u = self._calc.u()      # Scalar
        s2 = self._get_kernel_variance()

        # Compute A^{-1} y (already have as alpha)
        alpha = self._gp._alpha  # Shape: (n_train, nqoi)

        # For single output, squeeze to 1D
        if alpha.shape[1] == 1:
            alpha_1d = self._bkd.reshape(alpha, (-1,))  # Shape: (n_train,)
        else:
            raise NotImplementedError(
                "mean_of_variance currently only supports single-output GPs (nqoi=1)"
            )

        # Compute zeta = y^T A^{-1} P A^{-1} y = alpha^T P alpha
        P_alpha = P @ alpha_1d  # Shape: (n_train,)
        zeta = alpha_1d @ P_alpha  # Scalar

        # Compute v^2 = 1 - Tr[P A^{-1}]
        # A^{-1} = L^{-T} L^{-1}, so Tr[P A^{-1}] = Tr[P L^{-T} L^{-1}]
        # We can compute this as: Tr[P A^{-1}] = sum_i (A^{-1} P)_ii
        # But more efficiently: solve A Z = P, then Tr[Z] = Tr[A^{-1} P]
        A_inv_P = self._solve(P)  # Shape: (n_train, n_train)
        trace_P_A_inv = self._bkd.trace(A_inv_P)
        v_sq = 1.0 - trace_P_A_inv

        # Compute eta = mean of mean
        eta = self.mean_of_mean()

        # Compute varsigma^2 = u - tau^T A^{-1} tau
        tau_col = self._bkd.reshape(tau, (-1, 1))
        A_inv_tau = self._solve(tau_col)
        A_inv_tau = self._bkd.reshape(A_inv_tau, (-1,))
        varsigma_sq = u - tau @ A_inv_tau

        # E[gamma] = zeta + s^2 * v^2 - eta^2 - s^2 * varsigma^2
        mean_of_var = zeta + s2 * v_sq - eta * eta - s2 * varsigma_sq

        # Ensure non-negative (numerical stability)
        mean_of_var = mean_of_var * (mean_of_var >= 0.0)

        self._cache['mean_of_variance'] = mean_of_var
        return mean_of_var
