"""Benchmark protocols for the PyApprox typing module.

This module defines minimal protocols specific to benchmarks. Function protocols
are reused from pyapprox.interface.functions.protocols.

Protocols are organized in two groups:

**Capability protocols** — what the benchmark can do:
    HasForwardModel, HasPrior, HasJacobian, HasEnsembleModels,
    HasModelCosts, HasResidual, HasSmoothness, HasEstimatedEvaluationCost

**Reference data protocols** — what known answers exist:
    HasReferenceMean, HasReferenceVariance, HasMainEffects,
    HasTotalEffects, HasGlobalMinimum, HasReferenceIntegral,
    HasEnsembleMeans, HasEnsembleCovariance
"""

from typing import Any, Generic, List, Protocol, Sequence, runtime_checkable

from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class DomainProtocol(Protocol, Generic[Array]):
    """Input domain specification."""

    def bounds(self) -> Array:
        """Return bounds of shape (nvars, 2)."""
        ...

    def nvars(self) -> int:
        """Return number of variables."""
        ...

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        ...


@runtime_checkable
class GroundTruthProtocol(Protocol):
    """Ground truth for a benchmark - what is known/computable."""

    def available(self) -> Sequence[str]:
        """Return list of available ground truth properties."""
        ...

    def get(self, name: str) -> Any:
        """Get a ground truth value by name."""
        ...


@runtime_checkable
class BenchmarkProtocol(Protocol, Generic[Array]):
    """Fixed benchmark instance with ground truth."""

    def name(self) -> str:
        """Return the benchmark name."""
        ...

    def function(self) -> FunctionProtocol[Array]:
        """Return the benchmark function."""
        ...

    def ground_truth(self) -> GroundTruthProtocol:
        """Return the ground truth object."""
        ...

    def domain(self) -> DomainProtocol[Array]:
        """Return the domain specification."""
        ...


@runtime_checkable
class BenchmarkWithPriorProtocol(Protocol, Generic[Array]):
    """Benchmark with probability distribution (for UQ, sensitivity)."""

    def name(self) -> str:
        """Return the benchmark name."""
        ...

    def function(self) -> FunctionProtocol[Array]:
        """Return the benchmark function."""
        ...

    def ground_truth(self) -> GroundTruthProtocol:
        """Return the ground truth object."""
        ...

    def domain(self) -> DomainProtocol[Array]:
        """Return the domain specification."""
        ...

    def prior(self) -> Any:
        """Return the prior distribution."""
        ...


@runtime_checkable
class ConstraintProtocol(Protocol, Generic[Array]):
    """Single constraint function."""

    def __call__(self, samples: Array) -> Array:
        """Evaluate constraint at samples."""
        ...

    def constraint_type(self) -> str:
        """Return constraint type: 'eq' or 'ineq'."""
        ...


@runtime_checkable
class ConstrainedBenchmarkProtocol(Protocol, Generic[Array]):
    """Benchmark with constraints (extends base via composition)."""

    def name(self) -> str:
        """Return the benchmark name."""
        ...

    def function(self) -> FunctionProtocol[Array]:
        """Return the benchmark function."""
        ...

    def ground_truth(self) -> GroundTruthProtocol:
        """Return the ground truth object."""
        ...

    def domain(self) -> DomainProtocol[Array]:
        """Return the domain specification."""
        ...

    def constraints(self) -> Sequence[ConstraintProtocol[Array]]:
        """Return the list of constraints."""
        ...


# ---------------------------------------------------------------------------
# Capability protocols — what the benchmark can do
# ---------------------------------------------------------------------------


@runtime_checkable
class HasForwardModel(Protocol, Generic[Array]):
    """Benchmark exposes a forward model and domain."""

    def function(self) -> FunctionProtocol[Array]:
        """Return the forward model."""
        ...

    def domain(self) -> DomainProtocol[Array]:
        """Return the input domain."""
        ...


@runtime_checkable
class HasPrior(Protocol, Generic[Array]):
    """Benchmark exposes a prior distribution."""

    def prior(self) -> Any:
        """Return the prior distribution."""
        ...


@runtime_checkable
class HasJacobian(Protocol, Generic[Array]):
    """Benchmark exposes a Jacobian evaluation.

    Passthrough to ``function().jacobian()``.  Matches the signature of
    ``FunctionWithJacobianProtocol``: sample shape ``(nvars, 1)``,
    returns ``(nqoi, nvars)``.
    """

    def jacobian(self, sample: Array) -> Array:
        """Return the Jacobian at *sample*."""
        ...


@runtime_checkable
class HasEnsembleModels(Protocol, Generic[Array]):
    """Benchmark provides an ensemble of models at different fidelities.

    Each model in the list satisfies ``FunctionProtocol[Array]``.
    """

    def models(self) -> List[FunctionProtocol[Array]]:
        """Return list of model objects, each satisfying FunctionProtocol."""
        ...

    def nmodels(self) -> int:
        """Return number of models in the ensemble."""
        ...


@runtime_checkable
class HasModelCosts(Protocol, Generic[Array]):
    """Benchmark provides per-model evaluation costs."""

    def costs(self) -> Array:
        """Return per-model costs of shape (nmodels,)."""
        ...


@runtime_checkable
class HasResidual(Protocol, Generic[Array]):
    """Benchmark provides a residual function (ODE / implicit)."""

    def residual(self) -> Any:
        """Return the residual."""
        ...


@runtime_checkable
class HasSmoothness(Protocol):
    """Benchmark reports the smoothness of its forward model.

    Returns ``"analytic"`` or ``"C_inf"``.
    """

    def smoothness(self) -> str:
        """Return smoothness descriptor."""
        ...


@runtime_checkable
class HasEstimatedEvaluationCost(Protocol):
    """Approximate wall-clock seconds per forward-model evaluation."""

    def estimated_evaluation_cost(self) -> float:
        """Return estimated cost in seconds."""
        ...


# ---------------------------------------------------------------------------
# Reference data protocols — what known answers exist
# ---------------------------------------------------------------------------


@runtime_checkable
class HasReferenceMean(Protocol):
    """Benchmark has a known reference mean."""

    def reference_mean(self) -> float:
        """Return the reference mean."""
        ...


@runtime_checkable
class HasReferenceVariance(Protocol):
    """Benchmark has a known reference variance."""

    def reference_variance(self) -> float:
        """Return the reference variance."""
        ...


@runtime_checkable
class HasMainEffects(Protocol):
    """Benchmark has known main (first-order) Sobol indices."""

    def main_effects(self) -> Any:
        """Return main effects."""
        ...


@runtime_checkable
class HasTotalEffects(Protocol):
    """Benchmark has known total Sobol indices."""

    def total_effects(self) -> Any:
        """Return total effects."""
        ...


@runtime_checkable
class HasGlobalMinimum(Protocol):
    """Benchmark has a known global minimum."""

    def global_minimum(self) -> float:
        """Return the global minimum value."""
        ...

    def global_minimizers(self) -> Any:
        """Return the global minimizer(s)."""
        ...


@runtime_checkable
class HasReferenceIntegral(Protocol):
    """Benchmark has a known reference integral."""

    def reference_integral(self) -> float:
        """Return the reference integral value."""
        ...


# ---------------------------------------------------------------------------
# Ensemble reference data protocols — per-model statistics
# ---------------------------------------------------------------------------


@runtime_checkable
class HasEnsembleMeans(Protocol, Generic[Array]):
    """Benchmark has known per-model means.

    Returns array of shape ``(nmodels, nqoi)``.
    """

    def ensemble_means(self) -> Array:
        """Return per-model means."""
        ...


@runtime_checkable
class HasEnsembleCovariance(Protocol, Generic[Array]):
    """Benchmark has known cross-model covariance.

    Returns array of shape ``(nmodels, nmodels)`` for scalar QoI,
    or block structure for vector QoI.
    """

    def ensemble_covariance(self) -> Array:
        """Return cross-model covariance matrix."""
        ...


# ---------------------------------------------------------------------------
# OED capability protocols — what OED tasks the benchmark supports
# ---------------------------------------------------------------------------


@runtime_checkable
class HasObservationModel(Protocol, Generic[Array]):
    """Benchmark provides an observation model for OED."""

    def observation_model(self) -> Any:
        """Return the observation model."""
        ...


@runtime_checkable
class HasPredictionModel(Protocol, Generic[Array]):
    """Benchmark provides a prediction model for OED."""

    def prediction_model(self) -> Any:
        """Return the prediction model."""
        ...


@runtime_checkable
class HasExactEIG(Protocol, Generic[Array]):
    """Benchmark provides analytical expected information gain."""

    def exact_eig(self, weights: Array) -> float:
        """Return analytical EIG for given design weights."""
        ...
