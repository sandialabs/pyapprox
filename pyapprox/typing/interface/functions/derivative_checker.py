from typing import Protocol, Optional, runtime_checkable, Generic, Union, List

import numpy as np

from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.interface.functions.function import validate_sample
from pyapprox.typing.interface.functions.hessian_protocols import (
    FunctionWithJacobianApplyHessianProtocol,
    FunctionWithApplyJacobianApplyHessianProtocol,
)


@runtime_checkable
class DerivativeCheckerFunctionWithJacobianProtocol(Protocol, Generic[Array]):
    """
    Protocol for function objects used in derivative checks.

    Defines the required components for the function object.
    """

    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def jacobian(self, sample: Array) -> Array: ...


@runtime_checkable
class DerivativeCheckerFunctionWithApplyJacobianProtocol(
    Protocol, Generic[Array]
):
    """
    Protocol for function objects used in derivative checks.

    Defines the required components for the function object.
    """

    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def apply_jacobian(self, sample: Array, vec: Array) -> Array: ...


class FunctionWithApplyJacobianFromJacobian(Generic[Array]):
    def __init__(
        self,
        function: DerivativeCheckerFunctionWithJacobianProtocol[Array],
    ):
        if not isinstance(
            function, DerivativeCheckerFunctionWithJacobianProtocol
        ):
            raise ValueError(
                "The provided function must satisfy "
                "'DerivativeCheckerFunctionWithJacobianProtocol'. "
                f"Got an object of type {type(function).__name__}."
            )
        self._fun = function

    def bkd(self) -> Backend[Array]:
        return self._fun.bkd()

    def nvars(self) -> int:
        return self._fun.nvars()

    def __call__(self, samples: Array) -> Array:
        print(samples, self._fun)
        return self._fun(samples)

    def apply_jacobian(self, sample: Array, vec: Array) -> Array:
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


# TODO make these protocols generic on array, avoid inheritance
FunctionWithApplyHessian = Union[
    FunctionWithJacobianApplyHessianProtocol,
    FunctionWithApplyJacobianApplyHessianProtocol,
]


def has_apply_hessian(function: FunctionWithApplyHessian) -> bool:
    if not isinstance(
        function, FunctionWithJacobianApplyHessianProtocol
    ) and not isinstance(
        function,
        FunctionWithApplyJacobianApplyHessianProtocol,
    ):
        return False
    return True


class DerivativeCheckerFunctionWithApplyJacobianFromApplyHessian(
    Generic[Array]
):
    """
    Used to check hessian vector products with DerivativeChecker
    """

    def __init__(
        self,
        function: FunctionWithApplyHessian,
    ):
        if not isinstance(
            function, FunctionWithJacobianApplyHessianProtocol
        ) and not isinstance(
            function,
            FunctionWithApplyJacobianApplyHessianProtocol,
        ):
            raise ValueError(
                "The provided function must satisfy either "
                "'FunctionWithJacobianApplyHessianProtocol' or "
                "'FunctionWithApplyJacobianApplyHessianProtocol'. "
                f"Got an object of type {type(function).__name__}."
            )
        self._fun = function

    def bkd(self) -> Backend[Array]:
        return self._fun.bkd()

    def nvars(self) -> int:
        return self._fun.nvars()

    def _jacobian_from_apply(self, sample: Array) -> Array:
        nvars = sample.shape[0]
        actions = []
        for ii in range(nvars):
            vec = self.bkd().zeros((nvars, 1))
            vec[ii] = 1.0
            actions.append(self.apply_jacobian(sample, vec))
        return self.bkd().hstack(actions)

    def __call__(self, samples: Array) -> Array:
        if not isinstance(self._fun, FunctionWithJacobianApplyHessianProtocol):
            return self._jacobian_from_apply(samples)
        return self._fun.jacobian(samples)

    def apply_jacobian(self, sample: Array, vec: Array) -> Array:
        return self._fun.apply_hessian(sample, vec)

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the object for debugging.
        """
        return (
            f"{self.__class__.__name__}("
            f"nvars={self.nvars()}, "
            f"bkd={type(self.bkd()).__name__})"
        )


class ApplyJacobianChecker(Generic[Array]):
    """
    Class for checking the correctness of jacobian vector products

    Parameters
    ----------
    function_object : DerivativeCheckerProtocol
        The function object satisfying the required protocol.
    backend : Backend
        The backend to use for computations.
    """

    def __init__(
        self,
        function: DerivativeCheckerFunctionWithApplyJacobianProtocol[Array],
        symb: str = "J",
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        verbosity: int = 0,
    ):
        print(function)
        self._validate_function(function)
        self._fun = function
        self._bkd = function.bkd()

        if fd_eps is None:
            fd_eps = self._bkd.flip(self._bkd.logspace(-13, 0, 14))
        if direction is None:
            nvars = self._fun.nvars()
            direction = self._bkd.asarray(np.random.normal(0, 1, (nvars, 1)))
            direction /= self._bkd.norm(direction)

            self._symb = symb
        self._fd_eps = fd_eps
        self._direction = direction
        self._relative = relative
        self._verbosity = verbosity

    def _validate_function(
        self,
        function: DerivativeCheckerFunctionWithApplyJacobianProtocol[Array],
    ) -> None:
        if not isinstance(
            function, DerivativeCheckerFunctionWithApplyJacobianProtocol
        ):
            raise ValueError(
                "The provided function must satisfy "
                "'DerivativeCheckerFunctionWithJacobianProtocol'. "
                f"Got an object of type {type(function).__name__}."
            )

    def check(self, sample: Array) -> Array:
        """
        Compare the result of an apply_jacobian with directional
        finite difference approximations.
        """
        validate_sample(self._fun.nvars(), sample)
        errors = []
        val = self._fun(sample)
        directional_grad = self._fun.apply_jacobian(sample, self._direction)
        relative = self._relative
        if self._bkd.norm(directional_grad) < 1e-16:
            if self._verbosity > 0:
                print("Gradient is zero so setting relative=False")
            relative = False

        row_format = "{:<12} {:<25} {:<25} {:<25}"
        headers = [
            "Eps",
            "norm({0}v)".format(self._symb),
            "norm({0}v_fd)".format(self._symb),
            "Rel. Errors" if relative else "Abs. Errors",
        ]
        if self._verbosity > 0:
            print(row_format.format(*headers))
        row_format = "{:<12.2e} {:<25} {:<25} {:<25}"

        for ii in range(self._fd_eps.shape[0]):
            sample_perturbed = (
                self._bkd.copy(sample) + self._fd_eps[ii] * self._direction
            )
            perturbed_val = self._fun(sample_perturbed)
            fd_directional_grad = (perturbed_val - val) / self._fd_eps[ii]
            errors.append(
                self._bkd.norm(
                    fd_directional_grad.reshape(directional_grad.shape)
                    - directional_grad
                )
            )
            if relative:
                if self._bkd.norm(directional_grad) < 1e-16:
                    raise RuntimeError(
                        "directional grad is zero thus grad is likely zero. "
                        "Set relative=False"
                    )
                errors[-1] /= self._bkd.norm(directional_grad)
            if self._verbosity > 0:
                print(
                    row_format.format(
                        self._fd_eps[ii],
                        self._bkd.norm(directional_grad),
                        self._bkd.norm(fd_directional_grad),
                        errors[ii],
                    )
                )
        return self._bkd.asarray(errors)


class DerivativeChecker(Generic[Array]):
    def __init__(
        self,
        function: Union[
            DerivativeCheckerFunctionWithJacobianProtocol[Array],
            DerivativeCheckerFunctionWithApplyJacobianProtocol[Array],
        ],
    ):
        self._validate_function(function)
        self._fun = function

    def _validate_function(
        self,
        function: Union[
            DerivativeCheckerFunctionWithJacobianProtocol[Array],
            DerivativeCheckerFunctionWithApplyJacobianProtocol[Array],
        ],
    ) -> None:
        if not isinstance(
            function, DerivativeCheckerFunctionWithJacobianProtocol
        ) and not isinstance(
            function, DerivativeCheckerFunctionWithApplyJacobianProtocol
        ):
            raise ValueError(
                "The provided function must satisfy either "
                "'DerivativeCheckerFunctionWithJacobianProtocol' or "
                "'DerivativeCheckerFunctionWithApplyJacobianProtocol'. "
                f"Got an object of type {type(function).__name__}."
            )

    def _get_function_with_apply_jacobian(
        self,
    ) -> DerivativeCheckerFunctionWithApplyJacobianProtocol[Array]:
        if isinstance(
            self._fun, DerivativeCheckerFunctionWithApplyJacobianProtocol
        ):
            return self._fun
        return FunctionWithApplyJacobianFromJacobian(self._fun)

    def check_derivatives(
        self,
        sample: Array,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        verbosity: int = 0,
    ) -> List[Array]:
        jacobian_checker = ApplyJacobianChecker(
            self._get_function_with_apply_jacobian(),
            "J",
            fd_eps,
            direction,
            relative,
            verbosity,
        )
        errors = [jacobian_checker.check(sample)]
        if not has_apply_hessian(self._fun):
            return errors
        hessian_checker = ApplyJacobianChecker(
            DerivativeCheckerFunctionWithApplyJacobianFromApplyHessian(
                self._fun
            ),
            "H",
            fd_eps,
            direction,
            relative,
            verbosity,
        )
        errors.append(hessian_checker.check(sample))
        return errors
