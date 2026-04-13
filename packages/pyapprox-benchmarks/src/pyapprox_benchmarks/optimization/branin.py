"""Branin benchmark — three known global minima."""

from typing import Generic

from pyapprox.interface.functions.protocols import (
    FunctionWithJacobianProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend

from pyapprox_benchmarks.benchmark import BoxDomain
from pyapprox_benchmarks.functions.algebraic.branin import (
    BRANIN_GLOBAL_MINIMUM,
    BRANIN_MINIMIZERS,
    BraninFunction,
)
from pyapprox_benchmarks.problems.function_over_domain import (
    FunctionOverDomainProblem,
)


class BraninBenchmark(Generic[Array]):
    """Branin benchmark — three known global minima.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        func = BraninFunction(bkd)
        domain = BoxDomain(
            _bounds=bkd.array([[-5.0, 10.0], [0.0, 15.0]]),
            _bkd=bkd,
        )
        self._problem = FunctionOverDomainProblem(
            "branin_2d", func, domain,
            description="2D Branin function - three equivalent global minima",
        )
        self._global_minimum = BRANIN_GLOBAL_MINIMUM
        self._global_minimizers = bkd.array(
            [
                [m[0] for m in BRANIN_MINIMIZERS],
                [m[1] for m in BRANIN_MINIMIZERS],
            ]
        )

    def problem(
        self,
    ) -> FunctionOverDomainProblem[FunctionWithJacobianProtocol[Array], Array]:
        return self._problem  # type: ignore[return-value]

    def global_minimum(self) -> float:
        return self._global_minimum

    def global_minimizers(self) -> Array:
        return self._global_minimizers
