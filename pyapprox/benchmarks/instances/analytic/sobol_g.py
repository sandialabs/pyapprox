"""Sobol G-function benchmark instances.

Standard Sobol G-function benchmarks for sensitivity analysis with known
analytical Sobol indices.
"""

from pyapprox.benchmarks.benchmark import BenchmarkWithPrior, BoxDomain
from pyapprox.benchmarks.functions.algebraic.sobol_g import (
    SobolGFunction,
    SobolGSensitivityIndices,
)
from pyapprox.benchmarks.ground_truth import SensitivityGroundTruth
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend


class SobolGBenchmark:
    """Sobol G-function benchmark wrapper.

    Satisfies: HasForwardModel, HasPrior, HasJacobian,
    HasReferenceMean, HasReferenceVariance, HasMainEffects,
    HasTotalEffects, HasSmoothness, HasEstimatedEvaluationCost.
    """

    def __init__(self, inner):
        self._inner = inner

    def name(self):
        return self._inner.name()

    def function(self):
        return self._inner.function()

    def domain(self):
        return self._inner.domain()

    def prior(self):
        return self._inner.prior()

    def ground_truth(self):
        return self._inner.ground_truth()

    def jacobian(self, sample):
        return self._inner.function().jacobian(sample)

    def smoothness(self):
        return "analytic"

    def estimated_evaluation_cost(self):
        return 6.5e-05

    def reference_mean(self):
        return self._inner.ground_truth().mean

    def reference_variance(self):
        return self._inner.ground_truth().variance

    def main_effects(self):
        return self._inner.ground_truth().main_effects

    def total_effects(self):
        return self._inner.ground_truth().total_effects


def sobol_g_6d(
    bkd: Backend[Array],
) -> SobolGBenchmark:
    """Create the standard 6D Sobol G-function benchmark.

    Standard configuration with a = [0, 1, 4.5, 9, 99, 99].
    This gives decreasing importance from x1 to x6.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    SobolGBenchmark
        The Sobol G benchmark instance.

    References
    ----------
    Sobol, I.M. (1993). "Sensitivity analysis for non-linear mathematical
    models."
    """
    a = [0.0, 1.0, 4.5, 9.0, 99.0, 99.0]
    nvars = len(a)

    # Analytical Sobol indices
    indices = SobolGSensitivityIndices(bkd, a)
    main_effects = indices.main_effects()
    total_effects = indices.total_effects()
    variance = float(indices.variance()[0])

    # Build Sobol indices dict for first-order
    sobol_dict = {(i,): float(main_effects[i, 0]) for i in range(nvars)}

    # Standard uniform prior on [0, 1]^6
    prior = IndependentJoint(
        [UniformMarginal(0.0, 1.0, bkd) for _ in range(nvars)],
        bkd,
    )

    inner = BenchmarkWithPrior(
        _name="sobol_g_6d",
        _function=SobolGFunction(bkd, a=a),
        _domain=BoxDomain(
            _bounds=bkd.array([[0.0, 1.0]] * nvars),
            _bkd=bkd,
        ),
        _ground_truth=SensitivityGroundTruth(
            mean=1.0,  # Product of means, each g_i has mean 1
            variance=variance,
            main_effects=main_effects,
            total_effects=total_effects,
            sobol_indices=sobol_dict,
        ),
        _prior=prior,
        _description="Sobol G-function - 6D with standard importance parameters",
        _reference="Sobol, I.M. (1993)",
    )

    return SobolGBenchmark(inner)


def sobol_g_4d(
    bkd: Backend[Array],
) -> SobolGBenchmark:
    """Create a 4D Sobol G-function benchmark.

    Configuration with a = [0, 0, 0, 0] giving equal importance.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    SobolGBenchmark
        The Sobol G benchmark instance.
    """
    a = [0.0, 0.0, 0.0, 0.0]
    nvars = len(a)

    # Analytical Sobol indices
    indices = SobolGSensitivityIndices(bkd, a)
    main_effects = indices.main_effects()
    total_effects = indices.total_effects()
    variance = float(indices.variance()[0])

    # Build Sobol indices dict for first-order
    sobol_dict = {(i,): float(main_effects[i, 0]) for i in range(nvars)}

    # Standard uniform prior on [0, 1]^4
    prior = IndependentJoint(
        [UniformMarginal(0.0, 1.0, bkd) for _ in range(nvars)],
        bkd,
    )

    inner = BenchmarkWithPrior(
        _name="sobol_g_4d",
        _function=SobolGFunction(bkd, a=a),
        _domain=BoxDomain(
            _bounds=bkd.array([[0.0, 1.0]] * nvars),
            _bkd=bkd,
        ),
        _ground_truth=SensitivityGroundTruth(
            mean=1.0,
            variance=variance,
            main_effects=main_effects,
            total_effects=total_effects,
            sobol_indices=sobol_dict,
        ),
        _prior=prior,
        _description="Sobol G-function - 4D with equal importance",
        _reference="Sobol, I.M. (1993)",
    )

    return SobolGBenchmark(inner)


@BenchmarkRegistry.register(
    "sobol_g_6d",
    category="analytic",
    description="Standard 6D Sobol G-function for sensitivity analysis",
)
def _sobol_g_6d_factory(bkd: Backend[Array]) -> SobolGBenchmark:
    return sobol_g_6d(bkd)


@BenchmarkRegistry.register(
    "sobol_g_4d",
    category="analytic",
    description="4D Sobol G-function with equal importance",
)
def _sobol_g_4d_factory(bkd: Backend[Array]) -> SobolGBenchmark:
    return sobol_g_4d(bkd)
