from typing import Generic

from pyapprox.typing.util.backend import Array, Backend

from pyapprox.typing.interface.functions.protocols.jacobian import (
    FunctionWithJVPProtocol,
    FunctionWithJacobianOrJVPProtocol,
    function_has_jacobian_or_jvp,
)
from pyapprox.typing.interface.functions.protocols.hessian import (
    FunctionWithHVPAndJacobianOrJVPProtocol,
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
        function: FunctionWithHVPAndJacobianOrJVPProtocol[Array],
    ):
        if not function_has_hvp_and_jacobian_or_jvp(function):
            raise ValueError(
                "The provided function must satisfy either "
                "'FunctionWithJacobianAndHVPProtocol' or "
                "'FunctionWithJVPAndHVPProtocol'. "
                f"Got an object of type {type(function).__name__}."
            )
        self._fun = function

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
        return self._fun.jacobian(samples)

    def jvp(self, sample: Array, vec: Array) -> Array:
        return self._fun.hvp(sample, vec)

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the object for debugging.
        """
        return (
            f"{self.__class__.__name__}("
            f"nvars={self.nvars()}, "
            f"bkd={type(self.bkd()).__name__})"
        )
