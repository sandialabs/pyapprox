"""Branin benchmark instance.

Standard 2D Branin (Branin-Hoo) function for optimization with three
known global minima.
"""

from pyapprox.benchmarks.benchmark import Benchmark, BoxDomain
from pyapprox.benchmarks.functions.algebraic.branin import (
    BRANIN_GLOBAL_MINIMUM,
    BRANIN_MINIMIZERS,
    BraninFunction,
)
from pyapprox.benchmarks.ground_truth import OptimizationGroundTruth
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.util.backends.protocols import Array, Backend


class BraninBenchmark:
    """Branin benchmark wrapper.

    Satisfies: HasForwardModel, HasJacobian, HasGlobalMinimum,
    HasSmoothness, HasEstimatedEvaluationCost.
    """

    def __init__(self, inner):
        self._inner = inner

    def name(self):
        return self._inner.name()

    def function(self):
        return self._inner.function()

    def domain(self):
        return self._inner.domain()

    def ground_truth(self):
        return self._inner.ground_truth()

    def jacobian(self, sample):
        return self._inner.function().jacobian(sample)

    def smoothness(self):
        return "analytic"

    def estimated_evaluation_cost(self):
        return 1.7e-05

    def global_minimum(self):
        return self._inner.ground_truth().global_minimum

    def global_minimizers(self):
        return self._inner.ground_truth().global_minimizers


def branin_2d(
    bkd: Backend[Array],
) -> BraninBenchmark:
    """Create the standard 2D Branin benchmark.

    The Branin (Branin-Hoo) function is a classic benchmark for optimization
    algorithms, featuring three global minima in a 2D search space.

    Standard domain: x1 in [-5, 10], x2 in [0, 15]
    Global minimum: f(x*) ~ 0.397887
    Three global minimizers: (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    BraninBenchmark
        The 2D Branin benchmark instance.

    References
    ----------
    Branin, F.H. (1972). "Widely convergent method of finding multiple
    solutions of simultaneous nonlinear equations."
    """
    # Convert minimizers to array format (nvars, n_minimizers)
    minimizers = bkd.array(
        [
            [m[0] for m in BRANIN_MINIMIZERS],
            [m[1] for m in BRANIN_MINIMIZERS],
        ]
    )

    inner = Benchmark(
        _name="branin_2d",
        _function=BraninFunction(bkd),
        _domain=BoxDomain(
            _bounds=bkd.array([[-5.0, 10.0], [0.0, 15.0]]),
            _bkd=bkd,
        ),
        _ground_truth=OptimizationGroundTruth(
            global_minimum=BRANIN_GLOBAL_MINIMUM,
            global_minimizers=minimizers,
        ),
        _description="2D Branin function - three equivalent global minima",
        _reference="Branin, F.H. (1972)",
    )

    return BraninBenchmark(inner)


# TODO: I dont think analytic is a category. Rather this should be optimization
# analytic is the type of function underlying the benchmark but benchmark
# category should be about what benchmark is used to test. Should we have two
# categories, benchmark category and model category, e.g. analytic, ode, pde?
@BenchmarkRegistry.register(
    "branin_2d",
    category="analytic",
    description="Standard 2D Branin function with three global minima",
)
def _branin_2d_factory(bkd: Backend[Array]) -> BraninBenchmark:
    return branin_2d(bkd)
