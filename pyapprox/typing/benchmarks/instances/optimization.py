"""Fixed optimization benchmark instances.

These are pre-configured benchmark instances with standard parameters
and known ground truth values.
"""

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.benchmarks.benchmark import Benchmark, BoxDomain
from pyapprox.typing.benchmarks.ground_truth import OptimizationGroundTruth
from pyapprox.typing.benchmarks.registry import BenchmarkRegistry
from pyapprox.typing.benchmarks.functions.algebraic.rosenbrock import (
    RosenbrockFunction,
)


def rosenbrock_2d(
    bkd: Backend[Array],
) -> Benchmark[Array, OptimizationGroundTruth]:
    """Create the standard 2D Rosenbrock benchmark.

    Standard 2D Rosenbrock benchmark for unconstrained optimization.
    No prior - deterministic optimization problem.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    Benchmark[Array, OptimizationGroundTruth]
        The 2D Rosenbrock benchmark instance.

    References
    ----------
    Rosenbrock, H.H. (1960). "An automatic method for finding the greatest or
    least value of a function."
    """
    return Benchmark(
        _name="rosenbrock_2d",
        _function=RosenbrockFunction(bkd, nvars=2),
        _domain=BoxDomain(
            _bounds=bkd.array([[-5.0, 10.0], [-5.0, 10.0]]),
            _bkd=bkd,
        ),
        _ground_truth=OptimizationGroundTruth(
            global_minimum=0.0,
            global_minimizers=np.array([[1.0], [1.0]]),
        ),
        _description="2D Rosenbrock function - banana-shaped valley",
        _reference="Rosenbrock, H.H. (1960)",
    )


def rosenbrock_10d(
    bkd: Backend[Array],
) -> Benchmark[Array, OptimizationGroundTruth]:
    """Create the 10D Rosenbrock benchmark.

    10D Rosenbrock benchmark for high-dimensional optimization.
    No prior - deterministic optimization problem.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    Benchmark[Array, OptimizationGroundTruth]
        The 10D Rosenbrock benchmark instance.

    References
    ----------
    Rosenbrock, H.H. (1960). "An automatic method for finding the greatest or
    least value of a function."
    """
    nvars = 10
    return Benchmark(
        _name="rosenbrock_10d",
        _function=RosenbrockFunction(bkd, nvars=nvars),
        _domain=BoxDomain(
            _bounds=bkd.array([[-5.0, 10.0]] * nvars),
            _bkd=bkd,
        ),
        _ground_truth=OptimizationGroundTruth(
            global_minimum=0.0,
            global_minimizers=np.ones((nvars, 1)),
        ),
        _description="10D Rosenbrock function - high-dimensional optimization",
        _reference="Rosenbrock, H.H. (1960)",
    )


@BenchmarkRegistry.register(
    "rosenbrock_2d",
    category="optimization",
    description="Standard 2D Rosenbrock function for optimization",
)
def _rosenbrock_2d_factory(bkd: Backend[Array]) -> Benchmark[
    Array, OptimizationGroundTruth
]:
    return rosenbrock_2d(bkd)


@BenchmarkRegistry.register(
    "rosenbrock_10d",
    category="optimization",
    description="High-dimensional 10D Rosenbrock function",
)
def _rosenbrock_10d_factory(bkd: Backend[Array]) -> Benchmark[
    Array, OptimizationGroundTruth
]:
    return rosenbrock_10d(bkd)
