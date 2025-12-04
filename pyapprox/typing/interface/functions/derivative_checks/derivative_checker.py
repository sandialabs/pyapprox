from typing import (
    Protocol,
    Optional,
    runtime_checkable,
    Generic,
    Union,
    List,
    cast,
)

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.interface.functions.protocols.jacobian import (
    function_has_jacobian_or_jvp,
    FunctionWithJacobianOrJVPProtocol,
)
from pyapprox.typing.interface.functions.protocols.hessian import (
    FunctionWithHVPAndJacobianOrJVPProtocol,
    FunctionWithJacobianAndWHVPProtocol,
    function_has_hvp_and_jacobian_or_jvp,
)
from pyapprox.typing.interface.functions.derivative_checks.base import (
    JVPChecker,
)
from pyapprox.typing.interface.functions.derivative_checks.wrappers import (
    FunctionWithJVP,
    FunctionWithJVPFromHVP,
)


class DerivativeChecker(Generic[Array]):
    def __init__(self, function: FunctionWithJacobianOrJVPProtocol[Array]):
        self._validate_function(function)
        self._fun = function

    def bkd(self) -> Backend[Array]:
        return self._fun.bkd()

    def _validate_function(
        self,
        function: Union[
            FunctionWithJacobianOrJVPProtocol[Array],
            FunctionWithHVPAndJacobianOrJVPProtocol[Array],
            FunctionWithJacobianAndWHVPProtocol[Array],
        ],
    ) -> None:
        if not function_has_hvp_and_jacobian_or_jvp(
            function
        ) and not function_has_jacobian_or_jvp(function):
            raise ValueError(
                "The provided function must satisfy either "
                "'FunctionWithJacobianOrJVPProtocol. "
                f"Got an object of type {type(function).__name__}."
            )

    def check_derivatives(
        self,
        sample: Array,
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        verbosity: int = 0,
        weights: Optional[Array] = None,
    ) -> List[Array]:
        jacobian_checker = JVPChecker(
            FunctionWithJVP(self._fun),
            "J",
            fd_eps,
            direction,
            relative,
            verbosity,
        )
        errors = [jacobian_checker.check(sample)]
        if not function_has_hvp_and_jacobian_or_jvp(self._fun):
            return errors
        # use cast because type checker cannot determine that
        # self._fun is guaranteed to be this type if execution makes it here
        if weights is None and not hasattr(self._fun, "hvp"):
            weights = self.bkd().ones((self._fun.nqoi(), 1))
        hessian_checker = JVPChecker(
            FunctionWithJVPFromHVP(self._fun, weights),
            "H",
            fd_eps,
            direction,
            relative,
            verbosity,
        )
        errors.append(hessian_checker.check(sample))
        return errors

    def error_ratios_satisfied(self, errors: Array, tol: float) -> bool:
        print(self.bkd().min(errors) / self.bkd().max(errors))
        if self.bkd().min(errors) / self.bkd().max(errors) < tol:
            return True
        return False
