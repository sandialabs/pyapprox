"""Multi-fidelity Forrester ensemble benchmark instance.

Pre-configured benchmark with two Forrester function variants:
high-fidelity f_h(x) = (6x-2)^2 sin(12x-4) and low-fidelity
f_l(x) = A*f_h(x) + B*(x-0.5) + C. Domain: x in [0, 1].
"""

from typing import Generic, List

from pyapprox.benchmarks.benchmark import BoxDomain
from pyapprox.benchmarks.functions.multifidelity.forrester_ensemble import (
    ForresterEnsemble,
)
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend


class ForresterEnsembleBenchmark(Generic[Array]):
    """Multi-fidelity Forrester ensemble benchmark wrapper.

    Satisfies: HasEnsembleModels, HasModelCosts, HasPrior,
    HasSmoothness, HasEstimatedEvaluationCost.

    Parameters
    ----------
    inner_ensemble : ForresterEnsemble[Array]
        The ensemble of Forrester model functions.
    domain : BoxDomain
        Input domain.
    prior : IndependentJoint
        Prior distribution over the input domain.
    name : str
        Benchmark name.
    estimated_cost : float
        Estimated evaluation cost in seconds.
    """

    def __init__(
        self,
        inner_ensemble: ForresterEnsemble[Array],
        domain: BoxDomain[Array],
        prior: IndependentJoint[Array],
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

    def domain(self) -> BoxDomain[Array]:
        return self._domain

    def prior(self) -> IndependentJoint[Array]:
        return self._prior

    def costs(self) -> Array:
        return self._ensemble.costs()

    def smoothness(self) -> str:
        return "analytic"

    def estimated_evaluation_cost(self) -> float:
        return self._estimated_cost


def forrester_ensemble_2model(
    bkd: Backend[Array],
) -> ForresterEnsembleBenchmark[Array]:
    """Create 2-model multi-fidelity Forrester benchmark.

    Model 0 (high fidelity): f(x) = (6x - 2)^2 sin(12x - 4)
    Model 1 (low fidelity):  f(x) = 0.5 * f_h(x) + 10 * (x - 0.5) - 5

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    ForresterEnsembleBenchmark
        The benchmark instance.
    """
    ensemble = ForresterEnsemble(bkd)
    domain = BoxDomain(
        _bounds=bkd.array([[0.0, 1.0]]),
        _bkd=bkd,
    )
    prior = IndependentJoint(
        [UniformMarginal(0.0, 1.0, bkd)],
        bkd,
    )
    return ForresterEnsembleBenchmark(
        inner_ensemble=ensemble,
        domain=domain,
        prior=prior,
        name="forrester_ensemble_2model",
        estimated_cost=1.0e-06,
    )


@BenchmarkRegistry.register(
    "forrester_ensemble_2model",
    category="multifidelity",
    description="2-model multi-fidelity Forrester ensemble",
)
def _forrester_ensemble_2model_factory(
    bkd: Backend[Array],
) -> ForresterEnsembleBenchmark[Array]:
    return forrester_ensemble_2model(bkd)


# TODO: these benchmarks do not use ground truth pattern
