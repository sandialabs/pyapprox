"""Tunable ensemble multifidelity benchmark instance.

Pre-configured benchmark instance wrapping TunableModelEnsemble
for testing multifidelity estimators with tunable correlation structure.
"""

from typing import Generic, List

from pyapprox.benchmarks.benchmark import BoxDomain
from pyapprox.benchmarks.functions.multifidelity.tunable_ensemble import (
    TunableModelEnsemble,
)
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend


class TunableEnsembleBenchmark(Generic[Array]):
    """Tunable ensemble benchmark wrapper.

    Wraps a TunableModelEnsemble with 3 models, 2D input U[-1,1]^2,
    and tunable correlation structure.

    Satisfies: HasEnsembleModels, HasModelCosts, HasPrior,
    HasEnsembleMeans, HasEnsembleCovariance, HasSmoothness,
    HasEstimatedEvaluationCost.
    """

    def __init__(
        self,
        inner_ensemble: TunableModelEnsemble[Array],
        domain: BoxDomain[Array],
        prior,
        name: str,
        estimated_cost: float,
    ) -> None:
        self._ensemble = inner_ensemble
        self._domain = domain
        self._prior = prior
        self._name = name
        self._estimated_cost = estimated_cost

    def name(self) -> str:
        return self._name

    def models(self) -> List[FunctionProtocol[Array]]:
        return list(self._ensemble.models())

    def nmodels(self) -> int:
        return self._ensemble.nmodels()

    def domain(self):
        return self._domain

    def prior(self):
        return self._prior

    def costs(self) -> Array:
        return self._ensemble.costs()

    def smoothness(self) -> str:
        return "analytic"

    def estimated_evaluation_cost(self) -> float:
        return self._estimated_cost

    def ensemble_means(self) -> Array:
        return self._ensemble.means()  # (nmodels, 1)

    def ensemble_covariance(self) -> Array:
        return self._ensemble.covariance()  # (nmodels, nmodels)


def tunable_ensemble_3model(
    bkd: Backend[Array],
    theta1: float = 1.0,
) -> TunableEnsembleBenchmark[Array]:
    """Create 3-model tunable ensemble benchmark.

    Models with tunable correlation structure. theta1 controls the
    correlation between the high-fidelity and medium-fidelity model.
    Must satisfy pi/6 < theta1 < pi/2.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    theta1 : float
        Angle controlling medium-fidelity correlation. Default 1.0.

    Returns
    -------
    TunableEnsembleBenchmark
        The benchmark instance.
    """
    ensemble = TunableModelEnsemble(theta1=theta1, bkd=bkd)
    domain = BoxDomain(_bounds=bkd.array([[-1.0, 1.0], [-1.0, 1.0]]), _bkd=bkd)
    prior = IndependentJoint(
        [UniformMarginal(-1.0, 1.0, bkd), UniformMarginal(-1.0, 1.0, bkd)],
        bkd,
    )

    return TunableEnsembleBenchmark(
        inner_ensemble=ensemble,
        domain=domain,
        prior=prior,
        name="tunable_ensemble_3model",
        estimated_cost=2.3e-05,
    )

# TODO: these benchmarks do not use ground truth pattern
@BenchmarkRegistry.register(
    "tunable_ensemble_3model",
    category="multifidelity",
    description="3-model tunable correlation ensemble",
)
def _tunable_ensemble_3model_factory(
    bkd: Backend[Array],
) -> TunableEnsembleBenchmark[Array]:
    return tunable_ensemble_3model(bkd)
