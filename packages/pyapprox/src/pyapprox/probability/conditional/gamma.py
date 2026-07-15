"""
Conditional Gamma distribution.

Provides a conditional Gamma distribution where the log-shape and log-scale
parameters are functions of the conditioning variable.
"""

from typing import Generic, Optional

import numpy as np

from pyapprox.interface.functions.derivatives import JacobianFn
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.interface.functions.protocols.objective import (
    ObjectiveProtocol,
)
from pyapprox.probability.conditional.protocols import (
    ComponentWithHypListProtocol,
    ComponentWithParamJacobianProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


class ConditionalGamma(Generic[Array]):
    """
    Conditional Gamma distribution.

    p(y | x) = Gamma(y; exp(log_shape_func(x)), exp(log_scale_func(x)))

    The log-parameters are used to ensure positivity of shape and scale
    without constrained optimization.

    The Gamma distribution has PDF:
        f(y; k, theta) = y^{k-1} * exp(-y/theta) / (theta^k * Gamma(k))
    where k is the shape parameter and theta is the scale parameter.

    Parameters
    ----------
    log_shape_func : callable
        Function mapping x to log(shape). Must have:
        - __call__(x: Array) -> Array with shapes (nvars, n) -> (1, n)
        - nvars() -> int
        - nqoi() -> int (must be 1)
        Optionally: jacobian(x), jacobian_wrt_params(x), hyp_list()
    log_scale_func : callable
        Function mapping x to log(scale). Same interface as log_shape_func.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.surrogates.affine.expansions import BasisExpansion
    >>>
    >>> bkd = NumpyBkd()
    >>> # Create log_shape and log_scale as polynomial expansions
    >>> log_shape_func = BasisExpansion(basis, bkd, nqoi=1)
    >>> log_scale_func = BasisExpansion(basis, bkd, nqoi=1)
    >>>
    >>> cond = ConditionalGamma(log_shape_func, log_scale_func, bkd)
    >>> x = bkd.array([[0.5, 0.7]])  # Shape: (nvars=1, nsamples=2)
    >>> y = bkd.array([[1.0, 2.0]])  # Shape: (nqoi=1, nsamples=2), must be > 0
    >>> log_probs = cond.logpdf(x, y)  # Shape: (1, 2)
    """

    def __init__(
        self,
        log_shape_func: FunctionProtocol[Array],
        log_scale_func: FunctionProtocol[Array],
        bkd: Backend[Array],
    ):
        self._log_shape_func = log_shape_func
        self._log_scale_func = log_scale_func
        self._bkd = bkd

        # Validate that both functions have nqoi=1
        if log_shape_func.nqoi() != 1:
            raise ValueError(
                f"log_shape_func must have nqoi=1, got {log_shape_func.nqoi()}"
            )
        if log_scale_func.nqoi() != 1:
            raise ValueError(
                f"log_scale_func must have nqoi=1, got {log_scale_func.nqoi()}"
            )
        # Validate same nvars
        if log_shape_func.nvars() != log_scale_func.nvars():
            raise ValueError(
                f"log_shape_func and log_scale_func must have same nvars, "
                f"got {log_shape_func.nvars()} and {log_scale_func.nvars()}"
            )

        # Setup optional methods based on capabilities
        self._capture_component_capabilities()

    def _capture_component_capabilities(self) -> None:
        """Capture component capability once at construction.

        Absent capability is None (or a False predicate), never a
        missing attribute.
        """
        f1 = self._log_shape_func
        f2 = self._log_scale_func

        self._hyp_list: Optional[HyperParameterList[Array]] = None
        if isinstance(f1, ComponentWithHypListProtocol) and isinstance(
            f2, ComponentWithHypListProtocol
        ):
            self._hyp_list = f1.hyp_list() + f2.hyp_list()

        self._log_shape_jac: Optional[JacobianFn[Array]] = None
        self._log_scale_jac: Optional[JacobianFn[Array]] = None
        if isinstance(f1, ObjectiveProtocol) and isinstance(
            f2, ObjectiveProtocol
        ):
            jac1 = f1.derivatives().jacobian
            jac2 = f2.derivatives().jacobian
            if jac1 is not None and jac2 is not None:
                self._log_shape_jac = jac1
                self._log_scale_jac = jac2

        self._log_shape_params: Optional[
            ComponentWithParamJacobianProtocol[Array]
        ] = None
        self._log_scale_params: Optional[
            ComponentWithParamJacobianProtocol[Array]
        ] = None
        if isinstance(f1, ComponentWithParamJacobianProtocol) and isinstance(
            f2, ComponentWithParamJacobianProtocol
        ):
            self._log_shape_params = f1
            self._log_scale_params = f2

    def has_hyp_list(self) -> bool:
        """Whether both component functions expose hyperparameters."""
        return self._hyp_list is not None

    def has_logpdf_jacobian_wrt_x(self) -> bool:
        """Whether both component functions declare an input jacobian."""
        return self._log_shape_jac is not None

    def has_logpdf_jacobian_wrt_params(self) -> bool:
        """Whether both component functions provide parameter jacobians."""
        return self._log_shape_params is not None

    def hyp_list(self) -> HyperParameterList[Array]:
        """Return the combined hyperparameter list.

        Raises
        ------
        RuntimeError
            If not both component functions expose hyp_list().
        """
        if self._hyp_list is None:
            raise RuntimeError(
                "hyp_list is unavailable; check has_hyp_list before calling"
            )
        return self._hyp_list

    def nparams(self) -> int:
        """Return the total number of parameters.

        Raises
        ------
        RuntimeError
            If not both component functions expose hyp_list().
        """
        return int(self.hyp_list().nparams())
    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of conditioning variables."""
        return int(self._log_shape_func.nvars())

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

        Gamma logpdf: (shape-1)*log(y) - y/scale - shape*log(scale) - gammaln(shape)

        Parameters
        ----------
        x : Array
            Conditioning variable values. Shape: (nvars, nsamples)
        y : Array
            Output variable values. Shape: (1, nsamples). Must be > 0.

        Returns
        -------
        Array
            Log PDF values. Shape: (1, nsamples)
        """
        self._validate_inputs(x, y)

        log_shape = self._log_shape_func(x)  # (1, nsamples)
        log_scale = self._log_scale_func(x)  # (1, nsamples)
        shape = self._bkd.exp(log_shape)
        scale = self._bkd.exp(log_scale)

        log_y = self._bkd.log(y)

        return (
            (shape - 1.0) * log_y
            - y / scale
            - shape * log_scale
            - self._bkd.gammaln(shape)
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
            Random samples. Shape: (1, nsamples). Values are > 0.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got {x.ndim}D")
        if x.shape[0] != self.nvars():
            raise ValueError(
                f"x first dimension must be {self.nvars()}, got {x.shape[0]}"
            )

        shape = self._bkd.to_numpy(self._bkd.exp(self._log_shape_func(x))).flatten()
        scale = self._bkd.to_numpy(self._bkd.exp(self._log_scale_func(x))).flatten()

        samples = np.random.gamma(shape, scale)
        return self._bkd.reshape(self._bkd.asarray(samples), (1, -1))

    def logpdf_jacobian_wrt_x(self, x: Array, y: Array) -> Array:
        """
        Compute Jacobian of log PDF w.r.t. conditioning variable x.

        Uses chain rule:
        d(logpdf)/dx = d(logpdf)/d(log_shape) * d(log_shape)/dx
                     + d(logpdf)/d(log_scale) * d(log_scale)/dx

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
        log_shape_jac = self._log_shape_jac
        log_scale_jac = self._log_scale_jac
        if log_shape_jac is None or log_scale_jac is None:
            raise RuntimeError(
                "logpdf_jacobian_wrt_x is unavailable; check "
                "has_logpdf_jacobian_wrt_x before calling"
            )

        log_shape = self._log_shape_func(x)  # (1, 1)
        log_scale = self._log_scale_func(x)  # (1, 1)
        shape = self._bkd.exp(log_shape)
        scale = self._bkd.exp(log_scale)

        log_y = self._bkd.log(y)

        # Digamma term
        psi_shape = self._bkd.digamma(shape)

        # d(logpdf)/d(shape) = log(y) - log(scale) - psi(shape)
        # d(logpdf)/d(log_shape) = shape * d(logpdf)/d(shape)
        dlogpdf_dlogshape = shape * (log_y - log_scale - psi_shape)

        # d(logpdf)/d(scale) = y/scale^2 - shape/scale
        # d(logpdf)/d(log_scale) = scale * d(logpdf)/d(scale) = y/scale - shape
        dlogpdf_dlogscale = y / scale - shape

        # Get Jacobians of log_shape and log_scale w.r.t. x
        # jacobian returns (nqoi, nvars) for single sample
        dlogshape_dx = log_shape_jac(x)  # (1, nvars)
        dlogscale_dx = log_scale_jac(x)  # (1, nvars)

        # Chain rule
        result = dlogpdf_dlogshape * dlogshape_dx + dlogpdf_dlogscale * dlogscale_dx

        return result  # (1, nvars)

    def logpdf_jacobian_wrt_params(self, x: Array, y: Array) -> Array:
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
        """
        self._validate_inputs(x, y)
        log_shape_params = self._log_shape_params
        log_scale_params = self._log_scale_params
        if log_shape_params is None or log_scale_params is None:
            raise RuntimeError(
                "logpdf_jacobian_wrt_params is unavailable; check "
                "has_logpdf_jacobian_wrt_params before calling"
            )

        nsamples = x.shape[1]
        log_shape = self._log_shape_func(x)  # (1, nsamples)
        log_scale = self._log_scale_func(x)  # (1, nsamples)
        shape = self._bkd.exp(log_shape)
        scale = self._bkd.exp(log_scale)

        log_y = self._bkd.log(y)

        # Digamma term
        psi_shape = self._bkd.digamma(shape)

        # d(logpdf)/d(log_shape) = shape * (log(y) - log(scale) - psi(shape))
        dlogpdf_dlogshape = shape * (log_y - log_scale - psi_shape)

        # d(logpdf)/d(log_scale) = y/scale - shape
        dlogpdf_dlogscale = y / scale - shape

        # Get Jacobians of log_shape and log_scale w.r.t. their params
        # jacobian_wrt_params returns (nsamples, nqoi, nactive_params_i)
        dlogshape_dparams = log_shape_params.jacobian_wrt_params(
            x
        )  # (nsamples, 1, n_shape_params)
        dlogscale_dparams = log_scale_params.jacobian_wrt_params(
            x
        )  # (nsamples, 1, n_scale_params)

        # Chain rule
        # dlogpdf_dlogshape: (1, nsamples) -> need (nsamples, 1, 1) for broadcasting
        dlogpdf_dlogshape_expanded = self._bkd.reshape(
            dlogpdf_dlogshape.T, (nsamples, 1, 1)
        )
        jac_shape_params = (
            dlogpdf_dlogshape_expanded * dlogshape_dparams
        )  # (nsamples, 1, n_shape_params)

        dlogpdf_dlogscale_expanded = self._bkd.reshape(
            dlogpdf_dlogscale.T, (nsamples, 1, 1)
        )
        jac_scale_params = (
            dlogpdf_dlogscale_expanded * dlogscale_dparams
        )  # (nsamples, 1, n_scale_params)

        # Concatenate along parameter axis
        # Remove the nqoi=1 dimension for final output
        jac_shape = jac_shape_params[:, 0, :]  # (nsamples, n_shape_params)
        jac_scale = jac_scale_params[:, 0, :]  # (nsamples, n_scale_params)

        return self._bkd.hstack([jac_shape, jac_scale])  # (nsamples, nparams)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ConditionalGamma(nvars={self.nvars()}, nqoi={self.nqoi()}, "
            f"log_shape_func={self._log_shape_func}, "
            f"log_scale_func={self._log_scale_func})"
        )
