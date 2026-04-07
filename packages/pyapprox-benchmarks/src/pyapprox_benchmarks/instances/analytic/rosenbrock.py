"""Rosenbrock benchmark instances.

Standard Rosenbrock function benchmarks for optimization with known
global minimum.
"""

from typing import Generic, Optional

from pyapprox_benchmarks.benchmark import Benchmark, BoxDomain
from pyapprox_benchmarks.functions.algebraic.rosenbrock import (
    RosenbrockFunction,
)
from pyapprox_benchmarks.ground_truth import OptimizationGroundTruth
from pyapprox_benchmarks.protocols import DomainProtocol
from pyapprox_benchmarks.registry import BenchmarkRegistry
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.util.backends.protocols import Array, Backend


class RosenbrockBenchmark(Generic[Array]):
    """Rosenbrock benchmark wrapper.

    Satisfies: HasForwardModel, HasJacobian, HasGlobalMinimum,
    HasSmoothness, HasEstimatedEvaluationCost.
    """

    def __init__(self, inner: Benchmark[Array, OptimizationGroundTruth[Array]]) -> None:
        self._inner = inner

    def name(self) -> str:
        return self._inner.name()

    def function(self) -> FunctionProtocol[Array]:
        return self._inner.function()

    def domain(self) -> DomainProtocol[Array]:
        return self._inner.domain()

    def ground_truth(self) -> OptimizationGroundTruth[Array]:
        return self._inner.ground_truth()

    def jacobian(self, sample: Array) -> Array:
        return self._inner.function().jacobian(sample)

    def smoothness(self) -> str:
        return "analytic"

    def estimated_evaluation_cost(self) -> float:
        return 3.5e-05

    def global_minimum(self) -> Optional[float]:
        return self._inner.ground_truth().global_minimum

    def global_minimizers(self) -> Optional[Array]:
        return self._inner.ground_truth().global_minimizers


def rosenbrock_2d(
    bkd: Backend[Array],
) -> RosenbrockBenchmark[Array]:
    """Create the standard 2D Rosenbrock benchmark.

    Standard 2D Rosenbrock benchmark for unconstrained optimization.
    No prior - deterministic optimization problem.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    RosenbrockBenchmark
        The 2D Rosenbrock benchmark instance.

    References
    ----------
    Rosenbrock, H.H. (1960). "An automatic method for finding the greatest or
    least value of a function."
    """
    inner = Benchmark(
        _name="rosenbrock_2d",
        _function=RosenbrockFunction(bkd, nvars=2),
        _domain=BoxDomain(
            _bounds=bkd.array([[-5.0, 10.0], [-5.0, 10.0]]),
            _bkd=bkd,
        ),
        _ground_truth=OptimizationGroundTruth(
            global_minimum=0.0,
            global_minimizers=bkd.array([[1.0], [1.0]]),
        ),
        _description="2D Rosenbrock function - banana-shaped valley",
        _reference="Rosenbrock, H.H. (1960)",
    )

    return RosenbrockBenchmark(inner)


def rosenbrock_10d(
    bkd: Backend[Array],
) -> RosenbrockBenchmark[Array]:
    """Create the 10D Rosenbrock benchmark.

    10D Rosenbrock benchmark for high-dimensional optimization.
    No prior - deterministic optimization problem.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    RosenbrockBenchmark
        The 10D Rosenbrock benchmark instance.

    References
    ----------
    Rosenbrock, H.H. (1960). "An automatic method for finding the greatest or
    least value of a function."
    """
    nvars = 10
    inner = Benchmark(
        _name="rosenbrock_10d",
        _function=RosenbrockFunction(bkd, nvars=nvars),
        _domain=BoxDomain(
            _bounds=bkd.array([[-5.0, 10.0]] * nvars),
            _bkd=bkd,
        ),
        _ground_truth=OptimizationGroundTruth(
            global_minimum=0.0,
            global_minimizers=bkd.ones((nvars, 1)),
        ),
        _description="10D Rosenbrock function - high-dimensional optimization",
        _reference="Rosenbrock, H.H. (1960)",
    )

    return RosenbrockBenchmark(inner)


@BenchmarkRegistry.register(
    "rosenbrock_2d",
    category="analytic",
    description="Standard 2D Rosenbrock function for optimization",
)
def _rosenbrock_2d_factory(bkd: Backend[Array]) -> RosenbrockBenchmark[Array]:
    return rosenbrock_2d(bkd)


@BenchmarkRegistry.register(
    "rosenbrock_10d",
    category="analytic",
    description="High-dimensional 10D Rosenbrock function",
)
def _rosenbrock_10d_factory(bkd: Backend[Array]) -> RosenbrockBenchmark[Array]:
    return rosenbrock_10d(bkd)
