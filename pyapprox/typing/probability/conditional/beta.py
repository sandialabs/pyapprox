"""
Conditional Beta distribution.

Provides a conditional Beta distribution where the log-shape parameters
are functions of the conditioning variable.
"""

import math
from typing import Generic, Tuple

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import HyperParameterList


class ConditionalBeta(Generic[Array]):
    """
    Conditional Beta distribution.

    p(y | x) = Beta(y; exp(log_alpha_func(x)), exp(log_beta_func(x)))

    The log-shape parameters are used to ensure positivity of alpha and beta
    without constrained optimization.

    Parameters
    ----------
    log_alpha_func : callable
        Function mapping x to log(alpha). Must have:
        - __call__(x: Array) -> Array with shapes (nvars, n) -> (1, n)
        - nvars() -> int
        - nqoi() -> int (must be 1)
        Optionally: jacobian(x), jacobian_wrt_params(x), hyp_list()
    log_beta_func : callable
        Function mapping x to log(beta). Same interface as log_alpha_func.
    bkd : Backend[Array]
        Computational backend.
    lb : float, optional
        Lower bound of the support. Default is 0.0.
    ub : float, optional
        Upper bound of the support. Default is 1.0.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.affine.expansions import BasisExpansion
    >>>
    >>> bkd = NumpyBkd()
    >>> # Create log_alpha and log_beta as polynomial expansions
    >>> log_alpha_func = BasisExpansion(basis, bkd, nqoi=1)
    >>> log_beta_func = BasisExpansion(basis, bkd, nqoi=1)
    >>>
    >>> cond = ConditionalBeta(log_alpha_func, log_beta_func, bkd)
    >>> x = bkd.array([[0.5, 0.7]])  # Shape: (nvars=1, nsamples=2)
    >>> y = bkd.array([[0.3, 0.6]])  # Shape: (nqoi=1, nsamples=2), must be in (lb, ub)
    >>> log_probs = cond.logpdf(x, y)  # Shape: (1, 2)
    """

    def __init__(
        self,
        log_alpha_func: Generic[Array],
        log_beta_func: Generic[Array],
        bkd: Backend[Array],
        lb: float = 0.0,
        ub: float = 1.0,
    ):
        self._log_alpha_func = log_alpha_func
        self._log_beta_func = log_beta_func
        self._bkd = bkd

        # Validate and store bounds (NOT hyperparameters - fixed values)
        if lb >= ub:
            raise ValueError(f"lb must be < ub, got lb={lb}, ub={ub}")
        self._lb = float(lb)
        self._ub = float(ub)
        self._scale = self._ub - self._lb
        self._log_scale = math.log(self._scale)

        # Validate that both functions have nqoi=1
        if log_alpha_func.nqoi() != 1:
            raise ValueError(
                f"log_alpha_func must have nqoi=1, got {log_alpha_func.nqoi()}"
            )
        if log_beta_func.nqoi() != 1:
            raise ValueError(
                f"log_beta_func must have nqoi=1, got {log_beta_func.nqoi()}"
            )
        # Validate same nvars
        if log_alpha_func.nvars() != log_beta_func.nvars():
            raise ValueError(
                f"log_alpha_func and log_beta_func must have same nvars, "
                f"got {log_alpha_func.nvars()} and {log_beta_func.nvars()}"
            )

        # Setup optional methods based on capabilities
        self._setup_methods()

    def _setup_methods(self) -> None:
        """Bind optional methods based on component capabilities."""
        # Combine hyp_lists if both funcs have them
        if hasattr(self._log_alpha_func, "hyp_list") and hasattr(
            self._log_beta_func, "hyp_list"
        ):
            self._hyp_list = (
                self._log_alpha_func.hyp_list() + self._log_beta_func.hyp_list()
            )
            self.hyp_list = self._get_hyp_list
            self.nparams = self._get_nparams

        # Bind jacobian_wrt_x if both funcs support jacobian
        if hasattr(self._log_alpha_func, "jacobian") and hasattr(
            self._log_beta_func, "jacobian"
        ):
            self.logpdf_jacobian_wrt_x = self._logpdf_jacobian_wrt_x

        # Bind jacobian_wrt_params if both funcs support jacobian_wrt_params
        if hasattr(self._log_alpha_func, "jacobian_wrt_params") and hasattr(
            self._log_beta_func, "jacobian_wrt_params"
        ):
            self.logpdf_jacobian_wrt_params = self._logpdf_jacobian_wrt_params

    def _get_hyp_list(self) -> HyperParameterList:
        """Return the combined hyperparameter list."""
        return self._hyp_list

    def _get_nparams(self) -> int:
        """Return the total number of parameters."""
        return self._hyp_list.nparams()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of conditioning variables."""
        return self._log_alpha_func.nvars()

    def nqoi(self) -> int:
        """Return the number of output variables (always 1)."""
        return 1

    def lower(self) -> float:
        """Return the lower bound of the support."""
        return self._lb

    def upper(self) -> float:
        """Return the upper bound of the support."""
        return self._ub

    def bounds(self) -> Tuple[float, float]:
        """Return the support bounds (lb, ub)."""
        return (self._lb, self._ub)

    def _validate_inputs(self, x: Array, y: Array) -> None:
        """Validate input shapes."""
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got {x.ndim}D")
        if y.ndim != 2:
            raise ValueError(f"y must be 2D, got {y.ndim}D")
        if x.shape[0] != self.nvars():
            raise ValueError(
                f"x first dimension must be {self.nvars()}, got {x.shape[0]}"
            )
        if y.shape[0] != 1:
            raise ValueError(f"y first dimension must be 1, got {y.shape[0]}")
        if x.shape[1] != y.shape[1]:
            raise ValueError(
                f"x and y must have same number of samples, "
                f"got {x.shape[1]} and {y.shape[1]}"
            )

    def logpdf(self, x: Array, y: Array) -> Array:
        """
        Evaluate the log probability density function.

        Beta logpdf: (alpha-1)*log(y) + (beta-1)*log(1-y) - log(B(alpha,beta))
        where log(B(alpha,beta)) = gammaln(alpha) + gammaln(beta) - gammaln(alpha+beta)

        Parameters
        ----------
        x : Array
            Conditioning variable values. Shape: (nvars, nsamples)
        y : Array
            Output variable values. Shape: (1, nsamples). Must be in (lb, ub).

        Returns
        -------
        Array
            Log PDF values. Shape: (1, nsamples)
        """
        self._validate_inputs(x, y)

        # Transform y from [lb, ub] to [0, 1]
        y_01 = (y - self._lb) / self._scale

        log_alpha = self._log_alpha_func(x)  # (1, nsamples)
        log_beta = self._log_beta_func(x)  # (1, nsamples)
        alpha = self._bkd.exp(log_alpha)
        beta = self._bkd.exp(log_beta)

        # log(B(a,b)) = gammaln(a) + gammaln(b) - gammaln(a+b)
        log_beta_func = (
            self._bkd.gammaln(alpha)
            + self._bkd.gammaln(beta)
            - self._bkd.gammaln(alpha + beta)
        )

        log_y = self._bkd.log(y_01)
        log_1my = self._bkd.log(1.0 - y_01)

        # Include Jacobian correction: -log(scale)
        return (
            (alpha - 1.0) * log_y
            + (beta - 1.0) * log_1my
            - log_beta_func
            - self._log_scale
        )

    def rvs(self, x: Array) -> Array:
        """
        Generate random samples given conditioning variable.

        Parameters
        ----------
        x : Array
            Conditioning variable values. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Random samples. Shape: (1, nsamples). Values are in (lb, ub).
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got {x.ndim}D")
        if x.shape[0] != self.nvars():
            raise ValueError(
                f"x first dimension must be {self.nvars()}, got {x.shape[0]}"
            )

        alpha = self._bkd.to_numpy(
            self._bkd.exp(self._log_alpha_func(x))
        ).flatten()
        beta = self._bkd.to_numpy(
            self._bkd.exp(self._log_beta_func(x))
        ).flatten()

        # Generate samples on [0, 1] and transform to [lb, ub]
        samples_01 = np.random.beta(alpha, beta)
        samples = self._lb + samples_01 * self._scale
        return self._bkd.reshape(self._bkd.asarray(samples), (1, -1))

    def _logpdf_jacobian_wrt_x(self, x: Array, y: Array) -> Array:
        """
        Compute Jacobian of log PDF w.r.t. conditioning variable x.

        Uses chain rule:
        d(logpdf)/dx = d(logpdf)/d(log_alpha) * d(log_alpha)/dx
                     + d(logpdf)/d(log_beta) * d(log_beta)/dx

        Parameters
        ----------
        x : Array
            Conditioning variable values. Shape: (nvars, 1)
        y : Array
            Output variable values. Shape: (1, 1). Must be in (lb, ub).

        Returns
        -------
        Array
            Jacobian. Shape: (1, nvars)
        """
        self._validate_inputs(x, y)

        # Transform y from [lb, ub] to [0, 1]
        y_01 = (y - self._lb) / self._scale

        log_alpha = self._log_alpha_func(x)  # (1, 1)
        log_beta = self._log_beta_func(x)  # (1, 1)
        alpha = self._bkd.exp(log_alpha)
        beta = self._bkd.exp(log_beta)

        log_y = self._bkd.log(y_01)
        log_1my = self._bkd.log(1.0 - y_01)

        # Digamma terms
        psi_alpha = self._bkd.digamma(alpha)
        psi_beta = self._bkd.digamma(beta)
        psi_sum = self._bkd.digamma(alpha + beta)

        # d(logpdf)/d(alpha) = log(y_01) - psi(alpha) + psi(alpha+beta)
        # d(logpdf)/d(log_alpha) = alpha * d(logpdf)/d(alpha)
        dlogpdf_dlogalpha = alpha * (log_y - psi_alpha + psi_sum)

        # d(logpdf)/d(beta) = log(1-y_01) - psi(beta) + psi(alpha+beta)
        # d(logpdf)/d(log_beta) = beta * d(logpdf)/d(beta)
        dlogpdf_dlogbeta = beta * (log_1my - psi_beta + psi_sum)

        # Get Jacobians of log_alpha and log_beta w.r.t. x
        # jacobian returns (nqoi, nvars) for single sample
        dlogalpha_dx = self._log_alpha_func.jacobian(x)  # (1, nvars)
        dlogbeta_dx = self._log_beta_func.jacobian(x)  # (1, nvars)

        # Chain rule (Jacobian correction from scale is constant, doesn't affect derivative)
        result = dlogpdf_dlogalpha * dlogalpha_dx + dlogpdf_dlogbeta * dlogbeta_dx

        return result  # (1, nvars)

    def _logpdf_jacobian_wrt_params(self, x: Array, y: Array) -> Array:
        """
        Compute Jacobian of log PDF w.r.t. active parameters.

        Uses chain rule to propagate gradients through parameter functions.

        Parameters
        ----------
        x : Array
            Conditioning variable values. Shape: (nvars, nsamples)
        y : Array
            Output variable values. Shape: (1, nsamples). Must be in (lb, ub).

        Returns
        -------
        Array
            Jacobian. Shape: (nsamples, nactive_params)
        """
        self._validate_inputs(x, y)

        # Transform y from [lb, ub] to [0, 1]
        y_01 = (y - self._lb) / self._scale

        nsamples = x.shape[1]
        log_alpha = self._log_alpha_func(x)  # (1, nsamples)
        log_beta = self._log_beta_func(x)  # (1, nsamples)
        alpha = self._bkd.exp(log_alpha)
        beta = self._bkd.exp(log_beta)

        log_y = self._bkd.log(y_01)
        log_1my = self._bkd.log(1.0 - y_01)

        # Digamma terms
        psi_alpha = self._bkd.digamma(alpha)
        psi_beta = self._bkd.digamma(beta)
        psi_sum = self._bkd.digamma(alpha + beta)

        # d(logpdf)/d(log_alpha) = alpha * (log(y_01) - psi(alpha) + psi(alpha+beta))
        dlogpdf_dlogalpha = alpha * (log_y - psi_alpha + psi_sum)

        # d(logpdf)/d(log_beta) = beta * (log(1-y_01) - psi(beta) + psi(alpha+beta))
        dlogpdf_dlogbeta = beta * (log_1my - psi_beta + psi_sum)

        # Get Jacobians of log_alpha and log_beta w.r.t. their params
        # jacobian_wrt_params returns (nsamples, nqoi, nactive_params_i)
        dlogalpha_dparams = self._log_alpha_func.jacobian_wrt_params(
            x
        )  # (nsamples, 1, n_alpha_params)
        dlogbeta_dparams = self._log_beta_func.jacobian_wrt_params(
            x
        )  # (nsamples, 1, n_beta_params)

        # Chain rule (Jacobian correction from scale is constant, doesn't affect derivative)
        # dlogpdf_dlogalpha: (1, nsamples) -> need (nsamples, 1, 1) for broadcasting
        dlogpdf_dlogalpha_expanded = self._bkd.reshape(
            dlogpdf_dlogalpha.T, (nsamples, 1, 1)
        )
        jac_alpha_params = (
            dlogpdf_dlogalpha_expanded * dlogalpha_dparams
        )  # (nsamples, 1, n_alpha_params)

        dlogpdf_dlogbeta_expanded = self._bkd.reshape(
            dlogpdf_dlogbeta.T, (nsamples, 1, 1)
        )
        jac_beta_params = (
            dlogpdf_dlogbeta_expanded * dlogbeta_dparams
        )  # (nsamples, 1, n_beta_params)

        # Concatenate along parameter axis
        # Remove the nqoi=1 dimension for final output
        jac_alpha = jac_alpha_params[:, 0, :]  # (nsamples, n_alpha_params)
        jac_beta = jac_beta_params[:, 0, :]  # (nsamples, n_beta_params)

        return self._bkd.hstack([jac_alpha, jac_beta])  # (nsamples, nparams)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ConditionalBeta(nvars={self.nvars()}, nqoi={self.nqoi()}, "
            f"lb={self._lb}, ub={self._ub}, "
            f"log_alpha_func={self._log_alpha_func}, "
            f"log_beta_func={self._log_beta_func})"
        )
