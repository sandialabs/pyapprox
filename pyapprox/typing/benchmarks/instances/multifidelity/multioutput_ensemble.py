"""Multi-output ensemble multifidelity benchmark instances.

Pre-configured benchmark instances wrapping MultiOutputModelEnsemble
and PSDMultiOutputModelEnsemble for testing multi-output multifidelity
estimators with known analytical or numerical statistics.
"""

from typing import Generic, List

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.benchmarks.benchmark import BoxDomain
from pyapprox.typing.benchmarks.registry import BenchmarkRegistry
from pyapprox.typing.benchmarks.functions.multifidelity.multioutput_ensemble import (
    MultiOutputModelEnsemble,
    PSDMultiOutputModelEnsemble,
)
from pyapprox.typing.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.typing.probability.univariate.uniform import UniformMarginal
from pyapprox.typing.probability.joint.independent import IndependentJoint


class MultiOutputEnsembleBenchmark(Generic[Array]):
    """Multi-output ensemble benchmark wrapper.

    Wraps a MultiOutputModelEnsemble (or PSD variant) and exposes
    protocol-compliant methods.

    Satisfies: HasEnsembleModels, HasModelCosts, HasPrior,
    HasEnsembleMeans, HasEnsembleCovariance, HasSmoothness,
    HasEstimatedEvaluationCost.
    """

    def __init__(
        self,
        inner_ensemble,
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
        return self._ensemble.means()  # (nmodels, nqoi)

    def ensemble_covariance(self) -> Array:
        return self._ensemble.covariance_matrix()  # (nmodels*nqoi, nmodels*nqoi)


def multioutput_ensemble_3x3(
    bkd: Backend[Array],
) -> MultiOutputEnsembleBenchmark[Array]:
    """Create 3-model, 3-QoI multi-output ensemble benchmark.

    Wraps MultiOutputModelEnsemble with analytical covariance matrix.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    MultiOutputEnsembleBenchmark
        The benchmark instance.
    """
    ensemble = MultiOutputModelEnsemble(bkd)
    domain = BoxDomain(_bounds=bkd.array([[0.0, 1.0]]), _bkd=bkd)
    prior = IndependentJoint([UniformMarginal(0.0, 1.0, bkd)], bkd)

    return MultiOutputEnsembleBenchmark(
        inner_ensemble=ensemble,
        domain=domain,
        prior=prior,
        name="multioutput_ensemble_3x3",
        estimated_cost=1.8e-05,
    )


def psd_multioutput_ensemble_3x3(
    bkd: Backend[Array],
) -> MultiOutputEnsembleBenchmark[Array]:
    """Create PSD variant 3-model, 3-QoI multi-output ensemble benchmark.

    Wraps PSDMultiOutputModelEnsemble which uses numerical quadrature
    for means and covariance (no closed-form formulas due to
    perturbation terms).

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    MultiOutputEnsembleBenchmark
        The benchmark instance.
    """
    ensemble = PSDMultiOutputModelEnsemble(bkd)
    domain = BoxDomain(_bounds=bkd.array([[0.0, 1.0]]), _bkd=bkd)
    prior = IndependentJoint([UniformMarginal(0.0, 1.0, bkd)], bkd)

    return MultiOutputEnsembleBenchmark(
        inner_ensemble=ensemble,
        domain=domain,
        prior=prior,
        name="psd_multioutput_ensemble_3x3",
        estimated_cost=2.6e-05,
    )


@BenchmarkRegistry.register(
    "multioutput_ensemble_3x3",
    category="multifidelity",
    description="3-model, 3-QoI multi-output ensemble with analytical covariance",
)
def _multioutput_ensemble_3x3_factory(
    bkd: Backend[Array],
) -> MultiOutputEnsembleBenchmark[Array]:
    return multioutput_ensemble_3x3(bkd)


@BenchmarkRegistry.register(
    "psd_multioutput_ensemble_3x3",
    category="multifidelity",
    description="3-model, 3-QoI PSD multi-output ensemble with numerical covariance",
)
def _psd_multioutput_ensemble_3x3_factory(
    bkd: Backend[Array],
) -> MultiOutputEnsembleBenchmark[Array]:
    return psd_multioutput_ensemble_3x3(bkd)
