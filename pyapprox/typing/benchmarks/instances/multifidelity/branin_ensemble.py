"""Multi-fidelity Branin ensemble benchmark instance.

Pre-configured benchmark with three Branin function variants at
different fidelity levels. Useful for testing multi-fidelity surrogate
methods (MFNets, co-kriging) on a nonlinear 2D problem.
"""

from typing import Generic, List

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.benchmarks.benchmark import BoxDomain
from pyapprox.typing.benchmarks.registry import BenchmarkRegistry
from pyapprox.typing.benchmarks.functions.multifidelity.branin_ensemble import (
    BraninEnsemble,
)
from pyapprox.typing.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.typing.probability.univariate.uniform import UniformMarginal
from pyapprox.typing.probability.joint.independent import IndependentJoint


class BraninEnsembleBenchmark(Generic[Array]):
    """Multi-fidelity Branin ensemble benchmark wrapper.

    Satisfies: HasEnsembleModels, HasModelCosts, HasPrior,
    HasSmoothness, HasEstimatedEvaluationCost.

    Parameters
    ----------
    inner_ensemble : BraninEnsemble[Array]
        The ensemble of Branin model functions.
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
        inner_ensemble: BraninEnsemble[Array],
        domain: BoxDomain,
        prior: IndependentJoint,
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

    def domain(self) -> BoxDomain:
        return self._domain

    def prior(self) -> IndependentJoint:
        return self._prior

    def costs(self) -> Array:
        return self._ensemble.costs()

    def smoothness(self) -> str:
        return "analytic"

    def estimated_evaluation_cost(self) -> float:
        return self._estimated_cost


def branin_ensemble_3model(
    bkd: Backend[Array],
) -> BraninEnsembleBenchmark[Array]:
    """Create 3-model multi-fidelity Branin benchmark.

    Model 0 (high fidelity): Standard Branin on [-5, 10] x [0, 15].
    Model 1 (medium fidelity): Perturbed parameters + shift.
    Model 2 (low fidelity): Larger perturbations + larger shift.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    BraninEnsembleBenchmark
        The benchmark instance.
    """
    ensemble = BraninEnsemble(bkd)
    domain = BoxDomain(
        _bounds=bkd.array([[-5.0, 10.0], [0.0, 15.0]]),
        _bkd=bkd,
    )
    prior = IndependentJoint(
        [
            UniformMarginal(-5.0, 10.0, bkd),
            UniformMarginal(0.0, 15.0, bkd),
        ],
        bkd,
    )
    return BraninEnsembleBenchmark(
        inner_ensemble=ensemble,
        domain=domain,
        prior=prior,
        name="branin_ensemble_3model",
        estimated_cost=2.0e-06,
    )


@BenchmarkRegistry.register(
    "branin_ensemble_3model",
    category="multifidelity",
    description="3-model multi-fidelity Branin ensemble",
)
def _branin_ensemble_3model_factory(
    bkd: Backend[Array],
) -> BraninEnsembleBenchmark[Array]:
    return branin_ensemble_3model(bkd)
