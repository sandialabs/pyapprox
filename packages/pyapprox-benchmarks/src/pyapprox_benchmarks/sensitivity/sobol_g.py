"""Sobol G-function benchmark — analytical Sobol indices."""

from typing import Dict, Generic, List, Tuple

from pyapprox.interface.functions.protocols import (
    FunctionWithJacobianProtocol,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend

from pyapprox_benchmarks.benchmark import BoxDomain
from pyapprox_benchmarks.functions.algebraic.sobol_g import (
    SobolGFunction,
    SobolGSensitivityIndices,
)
from pyapprox_benchmarks.problems.forward_uq import ForwardUQProblem


class SobolGBenchmark(Generic[Array]):
    """Sobol G-function benchmark — analytical Sobol indices.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    a : list[float]
        Importance parameters. Length determines dimension.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        a: List[float],
    ) -> None:
        nvars = len(a)
        func = SobolGFunction(bkd, a=a)
        prior = IndependentJoint(
            [UniformMarginal(0.0, 1.0, bkd) for _ in range(nvars)],
            bkd,
        )
        domain = BoxDomain(
            _bounds=bkd.array([[0.0, 1.0]] * nvars),
            _bkd=bkd,
        )
        self._problem = ForwardUQProblem(
            f"sobol_g_{nvars}d", func, prior,
            description=f"Sobol G-function - {nvars}D",
        )
        self._domain = domain

        indices = SobolGSensitivityIndices(bkd, a)
        self._mean = 1.0
        self._variance = float(indices.variance()[0])
        self._main_effects = indices.main_effects()
        self._total_effects = indices.total_effects()
        self._sobol_indices: Dict[Tuple[int, ...], float] = {
            (i,): float(self._main_effects[i, 0]) for i in range(nvars)
        }

    def problem(
        self,
    ) -> ForwardUQProblem[FunctionWithJacobianProtocol[Array], Array]:
        return self._problem  # type: ignore[return-value]

    def domain(self) -> BoxDomain[Array]:
        return self._domain

    def mean(self) -> float:
        return self._mean

    def variance(self) -> float:
        return self._variance

    def main_effects(self) -> Array:
        return self._main_effects

    def total_effects(self) -> Array:
        return self._total_effects

    def sobol_indices(self) -> Dict[Tuple[int, ...], float]:
        return self._sobol_indices
