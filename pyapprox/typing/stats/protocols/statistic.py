"""Statistic protocol hierarchy for multifidelity estimation.

Defines a 3-level protocol hierarchy:
- Level 1: StatisticProtocol - Base for all statistics
- Level 2: StatisticWithCovarianceProtocol - Adds covariance computation
- Level 3: StatisticWithDiscrepancyProtocol - Adds control variate support
"""

from typing import Protocol, Generic, List, Tuple, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class StatisticProtocol(Protocol, Generic[Array]):
    """Level 1: Base protocol for all statistics.

    A statistic computes quantities of interest (QoI) from model samples.
    Examples: mean, variance, quantiles.

    Type Parameters
    ---------------
    Array : TypeVar
        Array type (e.g., numpy.ndarray, torch.Tensor).
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nqoi(self) -> int:
        """Return number of quantities of interest."""
        ...

    def nstats(self) -> int:
        """Return total number of scalar statistics.

        For mean of nqoi QoIs, nstats = nqoi.
        For variance of nqoi QoIs, nstats = nqoi.
        For mean+variance of nqoi QoIs, nstats = 2*nqoi.
        """
        ...

    def sample_estimate(self, values: Array) -> Array:
        """Compute sample estimate of the statistic.

        Parameters
        ----------
        values : Array
            Model output samples. Shape: (nsamples, nqoi)

        Returns
        -------
        Array
            Estimated statistic values. Shape: (nstats,)
        """
        ...

    def min_nsamples(self) -> int:
        """Return minimum number of samples needed.

        For mean: 1
        For variance: 2
        """
        ...


@runtime_checkable
class StatisticWithCovarianceProtocol(StatisticProtocol[Array], Protocol):
    """Level 2: Statistic with covariance estimation capability.

    Extends StatisticProtocol with methods for computing the covariance
    of the statistic estimator, which is needed for optimal sample allocation.
    """

    def nmodels(self) -> int:
        """Return number of models in the covariance structure.

        This is inferred from the pilot covariance matrix shape.

        Returns
        -------
        int
            Number of models.
        """
        ...

    def high_fidelity_estimator_covariance(self, nhf_samples: int) -> Array:
        """Compute covariance of the high-fidelity sample estimator.

        Parameters
        ----------
        nhf_samples : int
            Number of high-fidelity samples.

        Returns
        -------
        Array
            Covariance matrix. Shape: (nstats, nstats)
        """
        ...

    def compute_pilot_quantities(
        self, pilot_values: List[Array]
    ) -> Tuple[Array, ...]:
        """Compute pilot quantities from pilot samples.

        Parameters
        ----------
        pilot_values : List[Array]
            List of model outputs from pilot samples.
            pilot_values[m] has shape (npilot, nqoi) for model m.

        Returns
        -------
        Tuple[Array, ...]
            Pilot quantities (e.g., covariance matrix, means).
        """
        ...

    def set_pilot_quantities(self, *args) -> None:
        """Set pilot quantities directly.

        Parameters
        ----------
        *args
            Pilot quantities (statistic-specific).
        """
        ...

    def cov(self) -> Array:
        """Return the pilot covariance matrix.

        Returns
        -------
        Array
            Covariance matrix. Shape depends on statistic type.
        """
        ...


@runtime_checkable
class StatisticWithDiscrepancyProtocol(
    StatisticWithCovarianceProtocol[Array], Protocol
):
    """Level 3: Statistic with control variate discrepancy support.

    Extends StatisticWithCovarianceProtocol with methods for computing
    the covariance structure needed for control variate estimators.

    Control variates use discrepancies Delta_m = Q_m - Q_{m+1} to reduce
    variance. This protocol provides the covariance matrices needed to
    optimize the control variate weights and sample allocation.
    """

    def get_cv_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        """Get covariance matrices for standard CV estimator.

        For a CV estimator with a single low-fidelity model:
            Q_CV = Q_0 + eta * (mu_1 - Q_1)

        Parameters
        ----------
        npartition_samples : Array
            Number of samples in each partition. Shape: (npartitions,)

        Returns
        -------
        CF : Array
            Covariance between high-fidelity estimator and control.
            Shape: (nstats, nstats)
        cf : Array
            Variance of control variate estimator.
            Shape: (nstats, nstats)
        """
        ...

    def get_acv_discrepancy_covariances(
        self, allocation_mat: Array, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        """Get covariance matrices for ACV estimator.

        For an ACV estimator:
            Q_ACV = Q_0 + sum_m eta_m * (mu_m - Q_m)

        where mu_m and Q_m are computed on different sample sets
        determined by the allocation matrix.

        Parameters
        ----------
        allocation_mat : Array
            Allocation matrix A. A[i,j] = 1 if model i is evaluated
            on partition j. Shape: (nmodels, npartitions)
        npartition_samples : Array
            Number of samples in each partition. Shape: (npartitions,)

        Returns
        -------
        CF : Array
            Covariance between high-fidelity estimator and controls.
            Shape: (nstats, nstats * (nmodels-1))
        cf : Array
            Covariance of control variate estimators.
            Shape: (nstats * (nmodels-1), nstats * (nmodels-1))
        """
        ...

    def get_npartition_samples(
        self, allocation_mat: Array, nsamples_per_model: Array
    ) -> Array:
        """Compute samples per partition from samples per model.

        Parameters
        ----------
        allocation_mat : Array
            Allocation matrix. Shape: (nmodels, npartitions)
        nsamples_per_model : Array
            Number of samples for each model. Shape: (nmodels,)

        Returns
        -------
        Array
            Number of samples in each partition. Shape: (npartitions,)
        """
        ...
