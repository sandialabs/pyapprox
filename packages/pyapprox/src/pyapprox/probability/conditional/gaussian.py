"""
Conditional Gaussian distribution.

Provides a conditional Gaussian distribution where the mean and log-standard
deviation are functions of the conditioning variable.
"""

import math
from typing import TYPE_CHECKING, Generic

import numpy as np

from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList

if TYPE_CHECKING:
    from pyapprox.probability.univariate.gaussian import GaussianMarginal


class ConditionalGaussian(Generic[Array]):
    """
    Conditional Gaussian distribution.

    p(y | x) = N(y; mean_func(x), exp(log_stdev_func(x))^2)

    The log-standard deviation is used to ensure positivity of the
    standard deviation without constrained optimization.

    Parameters
    ----------
    mean_func : callable
        Function mapping x to mean(s). Must have:
        - __call__(x: Array) -> Array with shapes (nvars, n) -> (1, n)
        - nvars() -> int
        - nqoi() -> int (must be 1)
        Optionally: jacobian(x), jacobian_wrt_params(x), hyp_list()
    log_stdev_func : callable
        Function mapping x to log(stdev). Same interface as mean_func.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.surrogates.affine.expansions import BasisExpansion
    >>>
    >>> bkd = NumpyBkd()
    >>> # Create mean and log_stdev as polynomial expansions
    >>> mean_func = BasisExpansion(basis, bkd, nqoi=1)
    >>> log_stdev_func = BasisExpansion(basis, bkd, nqoi=1)
    >>>
    >>> cond = ConditionalGaussian(mean_func, log_stdev_func, bkd)
    >>> x = bkd.array([[0.5, 0.7]])  # Shape: (nvars=1, nsamples=2)
    >>> y = bkd.array([[1.0, 2.0]])  # Shape: (nqoi=1, nsamples=2)
    >>> log_probs = cond.logpdf(x, y)  # Shape: (1, 2)
    """

    def __init__(
        self,
        mean_func: FunctionProtocol[Array],
        log_stdev_func: FunctionProtocol[Array],
        bkd: Backend[Array],
    ):
        self._mean_func = mean_func
        self._log_stdev_func = log_stdev_func
        self._bkd = bkd

        # Validate that both functions have nqoi=1
        if mean_func.nqoi() != 1:
            raise ValueError(f"mean_func must have nqoi=1, got {mean_func.nqoi()}")
        if log_stdev_func.nqoi() != 1:
            raise ValueError(
                f"log_stdev_func must have nqoi=1, got {log_stdev_func.nqoi()}"
            )
        # Validate same nvars
        if mean_func.nvars() != log_stdev_func.nvars():
            raise ValueError(
                f"mean_func and log_stdev_func must have same nvars, "
                f"got {mean_func.nvars()} and {log_stdev_func.nvars()}"
            )

        # Constant for log PDF computation
        self._log_2pi = math.log(2.0 * math.pi)

        # Setup optional methods based on capabilities
        self._setup_methods()

    def _setup_methods(self) -> None:
        """Bind optional methods based on component capabilities."""
        # Combine hyp_lists if both funcs have them
        if hasattr(self._mean_func, "hyp_list") and hasattr(
            self._log_stdev_func, "hyp_list"
        ):
            self._hyp_list: HyperParameterList[Array] = (
                self._mean_func.hyp_list()  # type: ignore[attr-defined]
                + self._log_stdev_func.hyp_list()  # type: ignore[attr-defined]
            )
            self.hyp_list = self._get_hyp_list
            self.nparams = self._get_nparams

        # Bind jacobian_wrt_x if both funcs support jacobian
        if hasattr(self._mean_func, "jacobian") and hasattr(
            self._log_stdev_func, "jacobian"
        ):
            self.logpdf_jacobian_wrt_x = self._logpdf_jacobian_wrt_x

        # Bind jacobian_wrt_params if both funcs support jacobian_wrt_params
        if hasattr(self._mean_func, "jacobian_wrt_params") and hasattr(
            self._log_stdev_func, "jacobian_wrt_params"
        ):
            self.logpdf_jacobian_wrt_params = self._logpdf_jacobian_wrt_params
            self.reparameterize_jacobian_wrt_params = (
                self._reparameterize_jacobian_wrt_params
            )

    def _sync_param_funcs(self) -> None:
        """Sync parameter functions from hyp_list values.

        BasisExpansion.__call__ does not call sync_params, so if
        hyp_list values have been updated (e.g., by an optimizer), the
        coefficients used by __call__ may be stale.  This method ensures
        consistency before every evaluation.
        """
        if hasattr(self._mean_func, "sync_params"):
            self._mean_func.sync_params()
        if hasattr(self._log_stdev_func, "sync_params"):
            self._log_stdev_func.sync_params()

    def _get_hyp_list(self) -> HyperParameterList[Array]:
        """Return the combined hyperparameter list."""
        return self._hyp_list

    def _get_nparams(self) -> int:
        """Return the total number of parameters."""
        return int(self._hyp_list.nparams())

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of conditioning variables."""
        return int(self._mean_func.nvars())

    def nqoi(self) -> int:
        """Return the number of output variables (always 1)."""
        return 1

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

        Parameters
        ----------
        x : Array
            Conditioning variable values. Shape: (nvars, nsamples)
        y : Array
            Output variable values. Shape: (1, nsamples)

        Returns
        -------
        Array
            Log PDF values. Shape: (1, nsamples)
        """
        self._validate_inputs(x, y)
        self._sync_param_funcs()

        mean = self._mean_func(x)  # (1, nsamples)
        log_stdev = self._log_stdev_func(x)  # (1, nsamples)

        z = (y - mean) / self._bkd.exp(log_stdev)
        log_const = -0.5 * self._log_2pi - log_stdev
        return log_const - 0.5 * z**2

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
            Random samples. Shape: (1, nsamples)
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got {x.ndim}D")
        if x.shape[0] != self.nvars():
            raise ValueError(
                f"x first dimension must be {self.nvars()}, got {x.shape[0]}"
            )

        nsamples = x.shape[1]
        base = self._bkd.asarray(np.random.randn(1, nsamples))
        return self.reparameterize(x, base)

    def _logpdf_jacobian_wrt_x(self, x: Array, y: Array) -> Array:
        """
        Compute Jacobian of log PDF w.r.t. conditioning variable x.

        Uses chain rule:
        d(logpdf)/dx = d(logpdf)/d(mean) * d(mean)/dx
                     + d(logpdf)/d(log_stdev) * d(log_stdev)/dx

        Parameters
        ----------
        x : Array
            Conditioning variable values. Shape: (nvars, 1)
        y : Array
            Output variable values. Shape: (1, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (1, nvars)
        """
        self._validate_inputs(x, y)

        mean = self._mean_func(x)  # (1, 1)
        log_stdev = self._log_stdev_func(x)  # (1, 1)
        stdev = self._bkd.exp(log_stdev)

        z = (y - mean) / stdev

        # d(logpdf)/d(mean) = z / stdev
        dlogpdf_dmean = z / stdev  # (1, 1)

        # d(logpdf)/d(log_stdev) = -1 + z^2
        dlogpdf_dlogstdev = -1.0 + z**2  # (1, 1)

        # Get Jacobians of mean and log_stdev w.r.t. x
        # jacobian returns (nqoi, nvars) for single sample
        dmean_dx = self._mean_func.jacobian(x)  # (1, nvars)
        dlogstdev_dx = self._log_stdev_func.jacobian(x)  # (1, nvars)

        # Chain rule: d(logpdf)/dx = dlogpdf/dmean * dmean/dx + ...
        result = dlogpdf_dmean * dmean_dx + dlogpdf_dlogstdev * dlogstdev_dx

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
            Output variable values. Shape: (1, nsamples)

        Returns
        -------
        Array
            Jacobian. Shape: (nsamples, nactive_params)
            Where nactive_params = mean_func.nactive_params() +
                                   log_stdev_func.nactive_params()
        """
        self._validate_inputs(x, y)

        nsamples = x.shape[1]
        mean = self._mean_func(x)  # (1, nsamples)
        log_stdev = self._log_stdev_func(x)  # (1, nsamples)
        stdev = self._bkd.exp(log_stdev)

        z = (y - mean) / stdev

        # d(logpdf)/d(mean) = z / stdev, shape (1, nsamples)
        dlogpdf_dmean = z / stdev

        # d(logpdf)/d(log_stdev) = -1 + z^2, shape (1, nsamples)
        dlogpdf_dlogstdev = -1.0 + z**2

        # Get Jacobians of mean and log_stdev w.r.t. their params
        # jacobian_wrt_params returns (nsamples, nqoi, nactive_params_i)
        dmean_dparams = self._mean_func.jacobian_wrt_params(
            x
        )  # (nsamples, 1, n_mean_params)
        dlogstdev_dparams = self._log_stdev_func.jacobian_wrt_params(
            x
        )  # (nsamples, 1, n_stdev_params)

        # Chain rule: d(logpdf)/d(params_mean) = dlogpdf/dmean * dmean/dparams
        # dlogpdf_dmean: (1, nsamples) -> need (nsamples, 1, 1) for broadcasting
        dlogpdf_dmean_expanded = self._bkd.reshape(dlogpdf_dmean.T, (nsamples, 1, 1))
        jac_mean_params = (
            dlogpdf_dmean_expanded * dmean_dparams
        )  # (nsamples, 1, n_mean_params)

        # Similarly for log_stdev params
        dlogpdf_dlogstdev_expanded = self._bkd.reshape(
            dlogpdf_dlogstdev.T, (nsamples, 1, 1)
        )
        jac_stdev_params = (
            dlogpdf_dlogstdev_expanded * dlogstdev_dparams
        )  # (nsamples, 1, n_stdev_params)

        # Concatenate along parameter axis
        # Remove the nqoi=1 dimension for final output
        jac_mean = jac_mean_params[:, 0, :]  # (nsamples, n_mean_params)
        jac_stdev = jac_stdev_params[:, 0, :]  # (nsamples, n_stdev_params)

        return self._bkd.hstack([jac_mean, jac_stdev])  # (nsamples, nparams)

    def reparameterize(self, x: Array, base_samples: Array) -> Array:
        """Transform N(0,1) base samples to N(mean(x), stdev(x)^2).

        Parameters
        ----------
        x : Array
            Conditioning variable values. Shape: (nvars, nsamples)
        base_samples : Array
            Standard normal samples. Shape: (1, nsamples)

        Returns
        -------
        Array
            Reparameterized samples. Shape: (1, nsamples)
        """
        self._sync_param_funcs()
        mean = self._mean_func(x)  # (1, nsamples)
        log_s = self._log_stdev_func(x)  # (1, nsamples)
        return mean + self._bkd.exp(log_s) * base_samples

    def kl_divergence(self, x: Array, prior: "GaussianMarginal[Array]") -> Array:
        """Compute KL(q(.|x) || prior) per sample.

        Parameters
        ----------
        x : Array
            Conditioning variable values. Shape: (nvars, nsamples)
        prior : GaussianMarginal
            Prior distribution (unconditional).

        Returns
        -------
        Array
            Per-sample KL divergence. Shape: (1, nsamples)
        """
        self._sync_param_funcs()
        mean_q = self._mean_func(x)  # (1, nsamples)
        log_s_q = self._log_stdev_func(x)  # (1, nsamples)
        s_q = self._bkd.exp(log_s_q)
        m_p = prior.mean_value()  # float
        s_p = prior.std()  # float
        return (
            math.log(s_p)
            - log_s_q
            + (s_q**2 + (mean_q - m_p) ** 2) / (2.0 * s_p**2)
            - 0.5
        )

    def base_distribution(self) -> "GaussianMarginal[Array]":
        """Return the base distribution for reparameterization (standard normal)."""
        from pyapprox.probability.univariate.gaussian import (
            GaussianMarginal,
        )

        return GaussianMarginal(0.0, 1.0, self._bkd)

    def _reparameterize_jacobian_wrt_params(
        self, x: Array, base_samples: Array
    ) -> Array:
        """Compute Jacobian of reparameterize w.r.t. active parameters.

        Uses chain rule:
        z = mean_func(x) + exp(log_stdev_func(x)) * base_samples
        dz/d(mean_params) = d(mean_func)/d(params)
        dz/d(log_stdev_params) = exp(log_stdev) * base * d(log_stdev_func)/d(params)

        Parameters
        ----------
        x : Array
            Conditioning variable values. Shape: (nvars, nsamples)
        base_samples : Array
            Standard normal samples. Shape: (1, nsamples)

        Returns
        -------
        Array
            Jacobian. Shape: (nsamples, 1, nactive_params)
        """
        self._sync_param_funcs()
        nsamples = x.shape[1]
        log_s = self._log_stdev_func(x)  # (1, nsamples)
        stdev = self._bkd.exp(log_s)  # (1, nsamples)

        # d(mean_func)/d(params): (nsamples, 1, n_mean_params)
        dmean_dparams = self._mean_func.jacobian_wrt_params(x)

        # d(log_stdev_func)/d(params): (nsamples, 1, n_stdev_params)
        dlogstdev_dparams = self._log_stdev_func.jacobian_wrt_params(x)

        # dz/d(log_stdev_params) = stdev * base * d(log_stdev)/d(params)
        # stdev * base: (1, nsamples) -> (nsamples, 1, 1)
        scale = self._bkd.reshape((stdev * base_samples).T, (nsamples, 1, 1))
        jac_stdev_params = scale * dlogstdev_dparams

        return self._bkd.concatenate([dmean_dparams, jac_stdev_params], axis=2)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ConditionalGaussian(nvars={self.nvars()}, nqoi={self.nqoi()}, "
            f"mean_func={self._mean_func}, log_stdev_func={self._log_stdev_func})"
        )
