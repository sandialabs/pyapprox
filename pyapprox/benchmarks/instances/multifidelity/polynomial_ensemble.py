"""Polynomial ensemble multifidelity benchmark instances.

Pre-configured benchmark instances for testing multifidelity
estimators (MLMC, MFMC, ACV) with known analytical statistics.
"""

from typing import Generic, List

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.benchmarks.benchmark import BoxDomain
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.benchmarks.functions.multifidelity.polynomial_ensemble import (
    PolynomialEnsemble,
)
from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.probability.joint.independent import IndependentJoint


class PolynomialEnsembleBenchmark(Generic[Array]):
    """Polynomial ensemble multifidelity benchmark wrapper.

    Satisfies: HasEnsembleModels, HasModelCosts, HasPrior,
    HasEnsembleMeans, HasEnsembleCovariance, HasSmoothness,
    HasEstimatedEvaluationCost.
    """

    def __init__(
        self,
        inner_ensemble: PolynomialEnsemble,
        domain: BoxDomain,
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
        bkd = self._ensemble.bkd()
        means_1d = self._ensemble.means()  # (nmodels,)
        return bkd.reshape(means_1d, (-1, 1))  # (nmodels, 1)

    def ensemble_covariance(self) -> Array:
        return self._ensemble.covariance_matrix()  # (nmodels, nmodels)


def polynomial_ensemble_5model(
    bkd: Backend[Array],
) -> PolynomialEnsembleBenchmark[Array]:
    """Create 5-model polynomial ensemble benchmark.

    Models: f_k(x) = x^(5-k) for k = 0, 1, 2, 3, 4
    Input: x ~ U[0, 1]

    This is a standard benchmark for testing multifidelity estimators
    with known analytical correlations and costs.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    PolynomialEnsembleBenchmark
        The benchmark instance.
    """
    ensemble = PolynomialEnsemble(bkd, nmodels=5)
    domain = BoxDomain(_bounds=bkd.array([[0.0, 1.0]]), _bkd=bkd)
    prior = IndependentJoint([UniformMarginal(0.0, 1.0, bkd)], bkd)

    return PolynomialEnsembleBenchmark(
        inner_ensemble=ensemble,
        domain=domain,
        prior=prior,
        name="polynomial_ensemble_5model",
        estimated_cost=1.6e-06,
    )


def polynomial_ensemble_3model(
    bkd: Backend[Array],
) -> PolynomialEnsembleBenchmark[Array]:
    """Create 3-model polynomial ensemble benchmark.

    Models: f_k(x) = x^(3-k) for k = 0, 1, 2
    Input: x ~ U[0, 1]

    Smaller ensemble for simpler testing scenarios.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    PolynomialEnsembleBenchmark
        The benchmark instance.
    """
    ensemble = PolynomialEnsemble(bkd, nmodels=3)
    domain = BoxDomain(_bounds=bkd.array([[0.0, 1.0]]), _bkd=bkd)
    prior = IndependentJoint([UniformMarginal(0.0, 1.0, bkd)], bkd)

    return PolynomialEnsembleBenchmark(
        inner_ensemble=ensemble,
        domain=domain,
        prior=prior,
        name="polynomial_ensemble_3model",
        estimated_cost=1.6e-06,
    )


@BenchmarkRegistry.register(
    "polynomial_ensemble_5model",
    category="multifidelity",
    description="5-model polynomial ensemble",
)
def _polynomial_ensemble_5model_factory(
    bkd: Backend[Array],
) -> PolynomialEnsembleBenchmark[Array]:
    return polynomial_ensemble_5model(bkd)


@BenchmarkRegistry.register(
    "polynomial_ensemble_3model",
    category="multifidelity",
    description="3-model polynomial ensemble",
)
def _polynomial_ensemble_3model_factory(
    bkd: Backend[Array],
) -> PolynomialEnsembleBenchmark[Array]:
    return polynomial_ensemble_3model(bkd)
