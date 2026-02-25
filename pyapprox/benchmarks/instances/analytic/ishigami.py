"""Ishigami benchmark instance.

Standard 3D Ishigami function for sensitivity analysis with known
analytical Sobol indices.
"""

import math

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.benchmarks.benchmark import BenchmarkWithPrior, BoxDomain
from pyapprox.benchmarks.ground_truth import SensitivityGroundTruth
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.benchmarks.functions.algebraic.ishigami import (
    IshigamiFunction,
    IshigamiSensitivityIndices,
)
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.probability.joint.independent import IndependentJoint


class IshigamiBenchmark:
    """Ishigami benchmark wrapper.

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
        return 2.6e-05

    def reference_mean(self):
        return self._inner.ground_truth().mean

    def reference_variance(self):
        return self._inner.ground_truth().variance

    def main_effects(self):
        return self._inner.ground_truth().main_effects

    def total_effects(self):
        return self._inner.ground_truth().total_effects


def ishigami_3d(
    bkd: Backend[Array],
) -> IshigamiBenchmark:
    """Create the standard Ishigami benchmark.

    Standard Ishigami benchmark with a=7, b=0.1 on [-pi, pi]^3.

    Returns fixed benchmark instance - NOT configurable.
    Includes uniform prior U[-pi, pi]^3 for sensitivity analysis.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    IshigamiBenchmark
        The Ishigami benchmark instance.

    References
    ----------
    Ishigami, T. and Homma, T. (1990). "An importance quantification technique
    in uncertainty analysis for computer models."
    """
    a, b = 7.0, 0.1
    pi = math.pi

    # Analytical Sobol indices
    indices = IshigamiSensitivityIndices(bkd, a=a, b=b)
    mean_val = float(indices.mean()[0])
    variance = float(indices.variance()[0])
    main_effects = indices.main_effects()
    total_effects = indices.total_effects()
    sobol_all = indices.sobol_indices()

    # Build Sobol indices dict
    # Order from IshigamiSensitivityIndices: S_1, S_2, S_3, S_12, S_13, S_23, S_123
    sobol_dict = {
        (0,): float(sobol_all[0, 0]),
        (1,): float(sobol_all[1, 0]),
        (2,): float(sobol_all[2, 0]),
        (0, 2): float(sobol_all[4, 0]),  # S_13 is at index 4
    }

    # Standard uniform prior for sensitivity analysis
    prior = IndependentJoint(
        [
            UniformMarginal(-pi, pi, bkd),
            UniformMarginal(-pi, pi, bkd),
            UniformMarginal(-pi, pi, bkd),
        ],
        bkd,
    )

    inner = BenchmarkWithPrior(
        _name="ishigami_3d",
        _function=IshigamiFunction(bkd, a=a, b=b),
        _domain=BoxDomain(
            _bounds=bkd.array([[-pi, pi], [-pi, pi], [-pi, pi]]),
            _bkd=bkd,
        ),
        _ground_truth=SensitivityGroundTruth(
            mean=mean_val,
            variance=variance,
            main_effects=main_effects,
            total_effects=total_effects,
            sobol_indices=sobol_dict,
        ),
        _prior=prior,
        _description="Ishigami function - standard sensitivity analysis benchmark",
        _reference="Ishigami, T. and Homma, T. (1990)",
    )

    return IshigamiBenchmark(inner)


@BenchmarkRegistry.register(
    "ishigami_3d",
    category="analytic",
    description="Standard 3D Ishigami function for sensitivity analysis",
)
def _ishigami_3d_factory(bkd: Backend[Array]) -> IshigamiBenchmark:
    return ishigami_3d(bkd)
