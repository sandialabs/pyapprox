"""Rosenbrock benchmark — known global minimum."""

from typing import Generic

from pyapprox.interface.functions.protocols import (
    FunctionWithJacobianProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend

from pyapprox_benchmarks.benchmark import BoxDomain
from pyapprox_benchmarks.functions.algebraic.rosenbrock import (
    RosenbrockFunction,
)
from pyapprox_benchmarks.problems.function_over_domain import (
    FunctionOverDomainProblem,
)


class RosenbrockBenchmark(Generic[Array]):
    """Rosenbrock benchmark — known global minimum.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nvars : int
        Number of variables (default 2).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        nvars: int = 2,
    ) -> None:
        func = RosenbrockFunction(bkd, nvars=nvars)
        domain = BoxDomain(
            _bounds=bkd.array([[-5.0, 10.0]] * nvars),
            _bkd=bkd,
        )
        self._problem = FunctionOverDomainProblem(
            f"rosenbrock_{nvars}d", func, domain,
            description=f"{nvars}D Rosenbrock function - banana-shaped valley",
        )
        self._global_minimum = 0.0
        self._global_minimizers = bkd.ones((nvars, 1))

    def problem(
        self,
    ) -> FunctionOverDomainProblem[FunctionWithJacobianProtocol[Array], Array]:
        return self._problem  # type: ignore[return-value]

    def global_minimum(self) -> float:
        return self._global_minimum

    def global_minimizers(self) -> Array:
        return self._global_minimizers
