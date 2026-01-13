"""Fixed multifidelity benchmark instances.

Pre-configured benchmark instances for testing multifidelity
estimators (MLMC, MFMC, ACV) with known analytical statistics.
"""

from typing import TypeVar, Generic
from dataclasses import dataclass

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.benchmarks.benchmark import BoxDomain
from pyapprox.typing.benchmarks.ground_truth import MultifidelityGroundTruth
from pyapprox.typing.benchmarks.registry import BenchmarkRegistry
from pyapprox.typing.benchmarks.functions.multifidelity.polynomial_ensemble import (
    PolynomialEnsemble,
)


GT = TypeVar("GT")


@dataclass
class MultifidelityBenchmark(Generic[Array, GT]):
    """Benchmark for multifidelity methods.

    Unlike standard benchmarks, this contains an ensemble of models
    rather than a single function.

    Attributes
    ----------
    _name : str
        Benchmark identifier.
    _ensemble : PolynomialEnsemble[Array]
        The model ensemble.
    _domain : BoxDomain[Array]
        Input domain.
    _ground_truth : GT
        Ground truth values.
    _description : str
        Human-readable description.
    _reference : str
        Literature reference.
    """

    _name: str
    _ensemble: PolynomialEnsemble[Array]
    _domain: BoxDomain[Array]
    _ground_truth: GT
    _description: str = ""
    _reference: str = ""

    def name(self) -> str:
        """Return benchmark name."""
        return self._name

    def ensemble(self) -> PolynomialEnsemble[Array]:
        """Return the model ensemble."""
        return self._ensemble

    def domain(self) -> BoxDomain[Array]:
        """Return input domain."""
        return self._domain

    def ground_truth(self) -> GT:
        """Return ground truth."""
        return self._ground_truth

    def description(self) -> str:
        """Return description."""
        return self._description

    def reference(self) -> str:
        """Return literature reference."""
        return self._reference


def polynomial_ensemble_5model(
    bkd: Backend[Array],
) -> MultifidelityBenchmark[Array, MultifidelityGroundTruth]:
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
    MultifidelityBenchmark[Array, MultifidelityGroundTruth]
        The benchmark instance.
    """
    nmodels = 5
    ensemble = PolynomialEnsemble(bkd, nmodels=nmodels)

    # Extract analytical statistics (keep as backend arrays)
    means = ensemble.means()
    variances = ensemble.variances()
    costs = ensemble.costs()
    correlations = ensemble.correlation_matrix()

    ground_truth = MultifidelityGroundTruth(
        high_fidelity_mean=float(bkd.to_numpy(means)[0]),
        high_fidelity_variance=float(bkd.to_numpy(variances)[0]),
        model_correlations=correlations,
        model_costs=costs,
    )

    domain = BoxDomain(
        _bounds=bkd.array([[0.0, 1.0]]),
        _bkd=bkd,
    )

    return MultifidelityBenchmark(
        _name="polynomial_ensemble_5model",
        _ensemble=ensemble,
        _domain=domain,
        _ground_truth=ground_truth,
        _description="5-model polynomial ensemble for multifidelity testing",
        _reference="Gorodetsky et al. (2020), JCP",
    )


def polynomial_ensemble_3model(
    bkd: Backend[Array],
) -> MultifidelityBenchmark[Array, MultifidelityGroundTruth]:
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
    MultifidelityBenchmark[Array, MultifidelityGroundTruth]
        The benchmark instance.
    """
    nmodels = 3
    ensemble = PolynomialEnsemble(bkd, nmodels=nmodels)

    # Extract analytical statistics (keep as backend arrays)
    means = ensemble.means()
    variances = ensemble.variances()
    costs = ensemble.costs()
    correlations = ensemble.correlation_matrix()

    ground_truth = MultifidelityGroundTruth(
        high_fidelity_mean=float(bkd.to_numpy(means)[0]),
        high_fidelity_variance=float(bkd.to_numpy(variances)[0]),
        model_correlations=correlations,
        model_costs=costs,
    )

    domain = BoxDomain(
        _bounds=bkd.array([[0.0, 1.0]]),
        _bkd=bkd,
    )

    return MultifidelityBenchmark(
        _name="polynomial_ensemble_3model",
        _ensemble=ensemble,
        _domain=domain,
        _ground_truth=ground_truth,
        _description="3-model polynomial ensemble for multifidelity testing",
        _reference="Gorodetsky et al. (2020), JCP",
    )


# Register benchmarks
@BenchmarkRegistry.register(
    "polynomial_ensemble_5model",
    category="multifidelity",
    description="5-model polynomial ensemble",
)
def _polynomial_ensemble_5model_factory(
    bkd: Backend[Array],
) -> MultifidelityBenchmark[Array, MultifidelityGroundTruth]:
    return polynomial_ensemble_5model(bkd)


@BenchmarkRegistry.register(
    "polynomial_ensemble_3model",
    category="multifidelity",
    description="3-model polynomial ensemble",
)
def _polynomial_ensemble_3model_factory(
    bkd: Backend[Array],
) -> MultifidelityBenchmark[Array, MultifidelityGroundTruth]:
    return polynomial_ensemble_3model(bkd)
