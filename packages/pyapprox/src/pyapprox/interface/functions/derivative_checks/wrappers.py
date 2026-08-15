"""JVP-exposing views used by DerivativeChecker.

Capability is read from each function's Derivatives bundle (via the
migration shim ``resolve_bundle``), never via attribute probing.
"""

from typing import Generic, Optional

from pyapprox.interface.functions.derivative_checks.resolve import (
    resolve_bundle,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.util.backends.protocols import Array, Backend


class FunctionWithJVP(Generic[Array]):
    """Expose a jvp for first-order checking, from jvp or jacobian."""

    def __init__(self, function: FunctionProtocol[Array]):
        derivs = resolve_bundle(function)
        if derivs.jvp is None and derivs.jacobian is None:
            raise ValueError(
                "The provided function must declare a jacobian or jvp in "
                "its Derivatives bundle. "
                f"Got an object of type {type(function).__name__}."
            )
        self._fun = function
        self._jvp = derivs.jvp
        self._jacobian = derivs.jacobian

    def bkd(self) -> Backend[Array]:
        return self._fun.bkd()

    def nvars(self) -> int:
        return self._fun.nvars()

    def nqoi(self) -> int:
        return self._fun.nqoi()

    def __call__(self, samples: Array) -> Array:
        return self._fun(samples)

    def jvp(self, sample: Array, vec: Array) -> Array:
        if self._jvp is not None:
            return self._jvp(sample, vec)
        jacobian = self._jacobian
        if jacobian is None:
            raise RuntimeError(
                "jvp requires the function to declare a jacobian or jvp"
            )
        return jacobian(sample) @ vec

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the object for debugging.
        """
        return (
            f"{self.__class__.__name__}("
            f"nvars={self.nvars()}, "
            f"bkd={type(self.bkd()).__name__})"
        )


class FunctionWithJVPFromHVP(Generic[Array]):
    """
    Used to check hessian vector products with DerivativeChecker.

    Views the (weighted) gradient as the function and the (w)hvp as its
    jvp, so second derivatives are checked as first derivatives of the
    gradient.
    """

    def __init__(
        self,
        function: FunctionProtocol[Array],
        weights: Optional[Array] = None,
    ):
        derivs = resolve_bundle(function)
        if derivs.hvp is None and derivs.whvp is None:
            raise ValueError(
                "The provided function must declare an hvp or whvp in its "
                "Derivatives bundle. "
                f"Got an object of type {type(function).__name__}."
            )
        if derivs.jvp is None and derivs.jacobian is None:
            raise ValueError(
                "The provided function must declare a jacobian or jvp in "
                "its Derivatives bundle. "
                f"Got an object of type {type(function).__name__}."
            )
        if weights is None and derivs.hvp is None:
            raise AttributeError(
                "weights must be provided if testing the weighted hessian of a function"
            )
        if weights is not None and weights.shape != (function.nqoi(), 1):
            raise ValueError(
                "weights must have shape (nqoi, 1) = "
                f"({function.nqoi()}, 1), got {tuple(weights.shape)}. This "
                "is the orientation documented in "
                "pyapprox.interface.functions.derivatives and passed to "
                "production whvps by the optimizer adapters; a whvp that "
                "reads the whole weight vector rather than weights[0, 0] "
                "silently uses one weight if handed a row."
            )
        self._fun = function
        self._jacobian = derivs.jacobian
        self._explicit_jvp = derivs.jvp
        self._hvp = derivs.hvp
        self._whvp = derivs.whvp
        self._weights = weights

    def bkd(self) -> Backend[Array]:
        return self._fun.bkd()

    def nvars(self) -> int:
        return self._fun.nvars()

    def nqoi(self) -> int:
        return self._fun.nqoi()

    def _jacobian_from_apply(self, sample: Array) -> Array:
        explicit_jvp = self._explicit_jvp
        if explicit_jvp is None:
            raise RuntimeError(
                "_jacobian_from_apply requires the function to declare a "
                "jvp"
            )
        nvars = sample.shape[0]
        actions = []
        for ii in range(nvars):
            vec = self.bkd().zeros((nvars, 1))
            vec[ii] = 1.0
            actions.append(explicit_jvp(sample, vec))
        return self.bkd().hstack(actions)

    def __call__(self, samples: Array) -> Array:
        jacobian = self._jacobian
        if jacobian is None:
            return self._jacobian_from_apply(samples)
        if self.nqoi() == 1:
            return jacobian(samples)
        weights = self._weights
        if weights is None:
            raise RuntimeError(
                "weights are required for multi-QoI hessian checks"
            )
        # weights is (nqoi, 1) and the jacobian is (nqoi, nvars), so the
        # weighted jacobian contracts over the QoI axis.
        return weights.T @ jacobian(samples)

    def jvp(self, sample: Array, vec: Array) -> Array:
        hvp = self._hvp
        if self.nqoi() == 1 and hvp is not None:
            return hvp(sample, vec)
        whvp = self._whvp
        weights = self._weights
        if whvp is None or weights is None:
            raise RuntimeError(
                "jvp requires an hvp (nqoi == 1) or a whvp with weights"
            )
        return whvp(sample, vec, weights)

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the object for debugging.
        """
        return (
            f"{self.__class__.__name__}("
            f"nvars={self.nvars()}, "
            f"bkd={type(self.bkd()).__name__})"
        )


class SingleSampleFromBatchJacobian(Generic[Array]):
    """Expose a single-sample jacobian view of a batch jacobian.

    This allows testing jacobian_batch through DerivativeChecker by
    extracting results for a single sample.
    """

    def __init__(self, function: FunctionProtocol[Array]):
        derivs = resolve_bundle(function)
        if derivs.jacobian_batch is None:
            raise ValueError(
                "Function must declare jacobian_batch in its Derivatives "
                f"bundle. Got {type(function).__name__}."
            )
        self._fun = function
        self._jacobian_batch = derivs.jacobian_batch
        self._derivs: Derivatives[Array] = Derivatives.first_order(
            jacobian=self.jacobian
        )

    def derivatives(self) -> Derivatives[Array]:
        return self._derivs

    def bkd(self) -> Backend[Array]:
        return self._fun.bkd()

    def nvars(self) -> int:
        return self._fun.nvars()

    def nqoi(self) -> int:
        return self._fun.nqoi()

    def __call__(self, samples: Array) -> Array:
        # Evaluate function at the given samples
        return self._fun(samples)  # (nqoi, 1)

    def jacobian(self, sample: Array) -> Array:
        # Use jacobian_batch and extract single result
        jac_batch = self._jacobian_batch(sample)  # (1, nqoi, nvars)
        return jac_batch[0, :, :]  # (nqoi, nvars)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nvars={self.nvars()}, nqoi={self.nqoi()})"


class SingleSampleFromBatchHessian(Generic[Array]):
    """Expose single-sample jacobian/hvp views of batch derivatives.

    This allows testing hessian_batch through DerivativeChecker by
    extracting results for a single sample. Only for nqoi=1.
    """

    def __init__(self, function: FunctionProtocol[Array]):
        derivs = resolve_bundle(function)
        if derivs.hessian_batch is None:
            raise ValueError(
                "Function must declare hessian_batch in its Derivatives "
                f"bundle. Got {type(function).__name__}."
            )
        if derivs.jacobian_batch is None:
            raise ValueError(
                "Function must declare jacobian_batch in its Derivatives "
                f"bundle. Got {type(function).__name__}."
            )
        if function.nqoi() != 1:
            raise ValueError(
                f"hessian_batch only supported for nqoi=1. Got nqoi={function.nqoi()}."
            )
        self._fun = function
        self._jacobian_batch = derivs.jacobian_batch
        self._hessian_batch = derivs.hessian_batch
        # jacobian + hvp + materialized hessian is an unusual combination,
        # so the raw constructor is used instead of a named one.
        self._derivs: Derivatives[Array] = Derivatives(
            jacobian=self.jacobian, hvp=self.hvp, hessian=self.hessian
        )

    def derivatives(self) -> Derivatives[Array]:
        return self._derivs

    def bkd(self) -> Backend[Array]:
        return self._fun.bkd()

    def nvars(self) -> int:
        return self._fun.nvars()

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        return self._fun(samples)  # (1, 1)

    def jacobian(self, sample: Array) -> Array:
        # Use jacobian_batch for the gradient
        jac_batch = self._jacobian_batch(sample)  # (1, 1, nvars)
        return jac_batch[0, :, :]  # (1, nvars)

    def hessian(self, sample: Array) -> Array:
        # Use hessian_batch and extract single result
        hess_batch = self._hessian_batch(sample)  # (1, nvars, nvars)
        return hess_batch[0, :, :]  # (nvars, nvars)

    def hvp(self, sample: Array, vec: Array) -> Array:
        # Compute HVP from hessian
        hess = self.hessian(sample)  # (nvars, nvars)
        return hess @ vec  # (nvars, 1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nvars={self.nvars()})"
