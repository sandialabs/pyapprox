"""Protocol definitions for statistical estimators."""

from typing import (
    Protocol,
    runtime_checkable,
    Generic,
    List,
    Tuple,
    Callable,
)

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class StatisticProtocol(Protocol, Generic[Array]):
    """Protocol for multi-output statistics.

    Statistics compute quantities like mean, variance, or both from
    model evaluations across multiple models and QoIs.
    """

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        ...

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        ...

    def nstats(self) -> int:
        """Return the number of statistics computed."""
        ...

    def sample_estimate(self, values: Array) -> Array:
        """Compute the statistic from sample values."""
        ...

    def high_fidelity_estimator_covariance(self, nhf_samples: int) -> Array:
        """Return the covariance of the high-fidelity estimator."""
        ...

    def compute_pilot_quantities(
        self, pilot_values: List[Array]
    ) -> Tuple[Array, ...]:
        """Compute quantities from pilot samples."""
        ...

    def set_pilot_quantities(self, *args) -> None:
        """Set the pilot quantities."""
        ...

    def min_nsamples(self) -> int:
        """Return the minimum number of samples to compute the statistic."""
        ...


@runtime_checkable
class EstimatorProtocol(Protocol, Generic[Array]):
    """Protocol for Monte Carlo estimators.

    Estimators compute statistics from model evaluations with
    optimal sample allocation.
    """

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        ...

    def allocate_samples(self, target_cost: float) -> None:
        """Find optimal sample allocation for given budget."""
        ...

    def generate_samples_per_model(
        self, rvs: Callable[[int], Array]
    ) -> List[Array]:
        """Generate samples for each model."""
        ...

    def __call__(self, values: List[Array]) -> Array:
        """Compute the estimator from model values."""
        ...

    def optimized_covariance(self) -> Array:
        """Return the estimator covariance at optimal allocation."""
        ...

    def bootstrap(
        self, values: List[Array], nbootstraps: int
    ) -> Tuple[Array, Array]:
        """Estimate variance using bootstrapping."""
        ...


@runtime_checkable
class CVEstimatorProtocol(EstimatorProtocol[Array], Protocol):
    """Protocol for control variate estimators.

    Extends EstimatorProtocol with control variate specific methods.
    """

    def insert_pilot_values(
        self, values_per_model: List[Array]
    ) -> List[Array]:
        """Insert pilot values into model values."""
        ...


@runtime_checkable
class ACVEstimatorProtocol(CVEstimatorProtocol[Array], Protocol):
    """Protocol for approximate control variate estimators.

    Extends CVEstimatorProtocol with ACV specific methods.
    """

    def combine_acv_values(self, acv_values: List[Array]) -> List[Array]:
        """Combine ACV subset values into unique values per model."""
        ...

    def combine_acv_samples(self, acv_samples: List[Array]) -> List[Array]:
        """Combine ACV subset samples into unique samples per model."""
        ...
