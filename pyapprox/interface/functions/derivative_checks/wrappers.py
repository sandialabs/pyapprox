from typing import Generic, Optional, Protocol, Union, runtime_checkable

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend

from pyapprox.interface.functions.protocols.jacobian import (
    FunctionWithJVPProtocol,
    FunctionWithJacobianOrJVPProtocol,
    function_has_jacobian_or_jvp,
)
from pyapprox.interface.functions.protocols.hessian import (
    FunctionWithHVPAndJacobianOrJVPProtocol,
    FunctionWithJacobianAndWHVPProtocol,
    function_has_hvp_and_jacobian_or_jvp,
)


class FunctionWithJVP(Generic[Array]):
    def __init__(self, function: FunctionWithJacobianOrJVPProtocol[Array]):
        if not function_has_jacobian_or_jvp(function):
            raise ValueError(
                "The provided function must satisfy "
                "'FunctionWithJacobianOrJVPProtocol'. "
                f"Got an object of type {type(function).__name__}."
            )
        self._fun = function

    def bkd(self) -> Backend[Array]:
        return self._fun.bkd()

    def nvars(self) -> int:
        return self._fun.nvars()

    def nqoi(self) -> int:
        return self._fun.nqoi()

    def __call__(self, samples: Array) -> Array:
        return self._fun(samples)

    def jvp(self, sample: Array, vec: Array) -> Array:
        if isinstance(self._fun, FunctionWithJVPProtocol):
            return self._fun.jvp(sample, vec)
        return self._fun.jacobian(sample) @ vec

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
    Used to check hessian vector products with DerivativeChecker
    """

    def __init__(
        self,
        function: Union[
            FunctionWithHVPAndJacobianOrJVPProtocol[Array],
            FunctionWithJacobianAndWHVPProtocol,
        ],
        weights: Optional[Array] = None,
    ):
        if not function_has_hvp_and_jacobian_or_jvp(function):
            raise ValueError(
                "The provided function must satisfy either "
                "'FunctionWithJacobianAndHVPProtocol' or "
                "'FunctionWithJVPAndHVPProtocol' or "
                "'FunctionWithJacobianAndWHVPProtocol'."
                f"Got an object of type {type(function).__name__}."
            )
        self._fun = function
        if weights is None and not hasattr(self._fun, "hvp"):
            raise AttributeError(
                "weights must be provided if testing the weighted hessian of "
                "a function"
            )
        self._weights = weights

    def bkd(self) -> Backend[Array]:
        return self._fun.bkd()

    def nvars(self) -> int:
        return self._fun.nvars()

    def nqoi(self) -> int:
        return self._fun.nqoi()

    def _jacobian_from_apply(self, sample: Array) -> Array:
        nvars = sample.shape[0]
        actions = []
        for ii in range(nvars):
            vec = self.bkd().zeros((nvars, 1))
            vec[ii] = 1.0
            actions.append(self.jvp(sample, vec))
        return self.bkd().hstack(actions)

    def __call__(self, samples: Array) -> Array:
        if isinstance(self._fun, FunctionWithJVPProtocol):
            return self._jacobian_from_apply(samples)
        if self.nqoi() == 1:
            return self._fun.jacobian(samples)
        return self._weights @ self._fun.jacobian(samples)  # type: ignore

    def jvp(self, sample: Array, vec: Array) -> Array:
        if self.nqoi() == 1 and hasattr(self._fun, "hvp"):
            return self._fun.hvp(sample, vec)
        return self._fun.whvp(sample, vec, self._weights)

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
    """Wrap batch jacobian to expose single-sample jacobian interface.

    This allows testing jacobian_batch through DerivativeChecker by
    extracting results for a single sample.
    """

    def __init__(self, function: "BatchJacobianProtocol[Array]"):
        """Initialize wrapper.

        Parameters
        ----------
        function : BatchJacobianProtocol[Array]
            Function with jacobian_batch method.
        """
        if not hasattr(function, "jacobian_batch"):
            raise ValueError(
                "Function must have jacobian_batch method. "
                f"Got {type(function).__name__}."
            )
        self._fun = function

    def bkd(self) -> Backend[Array]:
        return self._fun.bkd()

    def nvars(self) -> int:
        return self._fun.nvars()

    def nqoi(self) -> int:
        return self._fun.nqoi()

    def __call__(self, sample: Array) -> Array:
        # Evaluate function at the given sample
        return self._fun(sample)  # (nqoi, 1)

    def jacobian(self, sample: Array) -> Array:
        # Use jacobian_batch and extract single result
        jac_batch = self._fun.jacobian_batch(sample)  # (1, nqoi, nvars)
        return jac_batch[0, :, :]  # (nqoi, nvars)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nvars={self.nvars()}, nqoi={self.nqoi()})"
        )


class SingleSampleFromBatchHessian(Generic[Array]):
    """Wrap batch hessian to expose single-sample hessian interface.

    This allows testing hessian_batch through DerivativeChecker by
    extracting results for a single sample. Only for nqoi=1.
    """

    def __init__(self, function: "BatchHessianProtocol[Array]"):
        """Initialize wrapper.

        Parameters
        ----------
        function : BatchHessianProtocol[Array]
            Function with hessian_batch method. Must have nqoi=1.
        """
        if not hasattr(function, "hessian_batch"):
            raise ValueError(
                "Function must have hessian_batch method. "
                f"Got {type(function).__name__}."
            )
        if not hasattr(function, "jacobian_batch"):
            raise ValueError(
                "Function must have jacobian_batch method. "
                f"Got {type(function).__name__}."
            )
        if function.nqoi() != 1:
            raise ValueError(
                f"hessian_batch only supported for nqoi=1. Got nqoi={function.nqoi()}."
            )
        self._fun = function

    def bkd(self) -> Backend[Array]:
        return self._fun.bkd()

    def nvars(self) -> int:
        return self._fun.nvars()

    def nqoi(self) -> int:
        return 1

    def __call__(self, sample: Array) -> Array:
        return self._fun(sample)  # (1, 1)

    def jacobian(self, sample: Array) -> Array:
        # Use jacobian_batch for the gradient
        jac_batch = self._fun.jacobian_batch(sample)  # (1, 1, nvars)
        return jac_batch[0, :, :]  # (1, nvars)

    def hessian(self, sample: Array) -> Array:
        # Use hessian_batch and extract single result
        hess_batch = self._fun.hessian_batch(sample)  # (1, nvars, nvars)
        return hess_batch[0, :, :]  # (nvars, nvars)

    def hvp(self, sample: Array, vec: Array) -> Array:
        # Compute HVP from hessian
        hess = self.hessian(sample)  # (nvars, nvars)
        return hess @ vec  # (nvars, 1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nvars={self.nvars()})"
        )


@runtime_checkable
class BatchJacobianProtocol(Protocol, Generic[Array]):
    """Protocol for functions with batch Jacobian."""

    def bkd(self) -> Backend[Array]: ...
    def nvars(self) -> int: ...
    def nqoi(self) -> int: ...
    def __call__(self, samples: Array) -> Array: ...
    def jacobian_batch(self, samples: Array) -> Array: ...


@runtime_checkable
class BatchHessianProtocol(Protocol, Generic[Array]):
    """Protocol for functions with batch Hessian."""

    def bkd(self) -> Backend[Array]: ...
    def nvars(self) -> int: ...
    def nqoi(self) -> int: ...
    def __call__(self, samples: Array) -> Array: ...
    def jacobian_batch(self, samples: Array) -> Array: ...
    def hessian_batch(self, samples: Array) -> Array: ...
