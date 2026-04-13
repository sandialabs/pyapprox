"""Ishigami benchmark — analytical Sobol indices."""

import math
from typing import Dict, Generic, Tuple

from pyapprox.interface.functions.protocols import (
    FunctionWithJacobianProtocol,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend

from pyapprox_benchmarks.benchmark import BoxDomain
from pyapprox_benchmarks.functions.algebraic.ishigami import (
    IshigamiFunction,
    IshigamiSensitivityIndices,
)
from pyapprox_benchmarks.problems.forward_uq import ForwardUQProblem


class IshigamiBenchmark(Generic[Array]):
    """Ishigami benchmark — analytical Sobol indices.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    a : float
        Ishigami parameter a (default 7.0).
    b : float
        Ishigami parameter b (default 0.1).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        a: float = 7.0,
        b: float = 0.1,
    ) -> None:
        pi = math.pi
        func = IshigamiFunction(bkd, a=a, b=b)
        prior = IndependentJoint(
            [UniformMarginal(-pi, pi, bkd) for _ in range(3)],
            bkd,
        )
        domain = BoxDomain(
            _bounds=bkd.array([[-pi, pi], [-pi, pi], [-pi, pi]]),
            _bkd=bkd,
        )
        self._problem = ForwardUQProblem(
            "ishigami", func, prior,
            description="Ishigami function - standard sensitivity analysis benchmark",
        )
        self._domain = domain

        indices = IshigamiSensitivityIndices(bkd, a=a, b=b)
        self._mean = float(indices.mean()[0])
        self._variance = float(indices.variance()[0])
        self._main_effects = indices.main_effects()
        self._total_effects = indices.total_effects()
        sobol_all = indices.sobol_indices()
        self._sobol_indices: Dict[Tuple[int, ...], float] = {
            (0,): float(sobol_all[0, 0]),
            (1,): float(sobol_all[1, 0]),
            (2,): float(sobol_all[2, 0]),
            (0, 2): float(sobol_all[4, 0]),
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
