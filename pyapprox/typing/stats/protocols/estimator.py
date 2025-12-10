"""Estimator protocol hierarchy for multifidelity estimation.

Defines a 3-level protocol hierarchy:
- Level 1: EstimatorProtocol - Base for all estimators
- Level 2: ControlVariateEstimatorProtocol - Control variate estimators
- Level 3: ParametricEstimatorProtocol - Configurable recursion structure
"""

from typing import Protocol, Generic, List, Callable, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class EstimatorProtocol(Protocol, Generic[Array]):
    """Level 1: Base protocol for all multifidelity estimators.

    An estimator computes a statistic using samples from one or more models.

    Type Parameters
    ---------------
    Array : TypeVar
        Array type (e.g., numpy.ndarray, torch.Tensor).
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nmodels(self) -> int:
        """Return number of models."""
        ...

    def costs(self) -> Array:
        """Return computational cost per sample for each model.

        Returns
        -------
        Array
            Costs. Shape: (nmodels,)
        """
        ...

    def allocate_samples(self, target_cost: float) -> None:
        """Allocate samples to minimize variance for given cost.

        Parameters
        ----------
        target_cost : float
            Total computational budget.
        """
        ...

    def nsamples_per_model(self) -> Array:
        """Return allocated samples per model.

        Returns
        -------
        Array
            Number of samples. Shape: (nmodels,)
        """
        ...

    def generate_samples_per_model(
        self, rvs: Callable[[int], Array]
    ) -> List[Array]:
        """Generate samples for each model.

        Parameters
        ----------
        rvs : Callable[[int], Array]
            Random variable sampler. Takes nsamples, returns samples.
            Output shape: (nsamples, nvars)

        Returns
        -------
        List[Array]
            Samples for each model. samples[m] has shape (nsamples_m, nvars)
        """
        ...

    def __call__(self, values: List[Array]) -> Array:
        """Compute the estimate from model evaluations.

        Parameters
        ----------
        values : List[Array]
            Model outputs. values[m] has shape (nsamples_m, nqoi)

        Returns
        -------
        Array
            Estimated statistic. Shape: (nstats,)
        """
        ...

    def optimized_covariance(self) -> Array:
        """Return covariance of the optimized estimator.

        Returns
        -------
        Array
            Covariance matrix. Shape: (nstats, nstats)
        """
        ...


@runtime_checkable
class ControlVariateEstimatorProtocol(EstimatorProtocol[Array], Protocol):
    """Level 2: Control variate estimator protocol.

    Extends EstimatorProtocol for estimators that use low-fidelity models
    as control variates to reduce variance.

    The general form is:
        Q_CV = Q_0 + sum_m eta_m * (mu_m - Q_m)

    where Q_0 is the high-fidelity estimate, Q_m are low-fidelity estimates,
    mu_m are known or estimated means, and eta_m are optimal weights.
    """

    def npartitions(self) -> int:
        """Return number of sample partitions.

        Partitions define groups of samples that are shared across models.
        """
        ...

    def get_allocation_matrix(self) -> Array:
        """Return the allocation matrix.

        The allocation matrix A defines which models are evaluated on
        which sample partitions. A[i,j] = 1 if model i is evaluated
        on partition j.

        Returns
        -------
        Array
            Allocation matrix. Shape: (nmodels, npartitions)
        """
        ...

    def weights(self) -> Array:
        """Return optimal control variate weights.

        Returns
        -------
        Array
            Weights eta. Shape: (nstats, nmodels-1)
        """
        ...

    def npartition_samples(self) -> Array:
        """Return number of samples in each partition.

        Returns
        -------
        Array
            Partition sizes. Shape: (npartitions,)
        """
        ...


@runtime_checkable
class ParametricEstimatorProtocol(
    ControlVariateEstimatorProtocol[Array], Protocol
):
    """Level 3: Parametric estimator with configurable recursion.

    Extends ControlVariateEstimatorProtocol for estimators whose
    allocation matrix structure is determined by a recursion index.

    The recursion index defines how models are coupled:
    - recursion_index[m] = k means model m is coupled with model k
    - This determines which discrepancies Delta_m = Q_m - Q_k are used

    Examples:
    - MFMC: recursion_index = [0, 0, 0, ...] (all coupled with HF)
    - MLMC: recursion_index = [0, 1, 2, ...] (successive coupling)
    - ACV: recursion_index optimized for minimum variance
    """

    def recursion_index(self) -> Array:
        """Return the recursion index.

        Returns
        -------
        Array
            Recursion index. Shape: (nmodels-1,)
            recursion_index[m] is the index of the model that
            low-fidelity model m+1 is coupled with.
        """
        ...

    def set_recursion_index(self, index: Array) -> None:
        """Set the recursion index.

        Parameters
        ----------
        index : Array
            New recursion index. Shape: (nmodels-1,)
        """
        ...

    def allocation_matrix_from_recursion(self, index: Array) -> Array:
        """Compute allocation matrix from recursion index.

        Parameters
        ----------
        index : Array
            Recursion index. Shape: (nmodels-1,)

        Returns
        -------
        Array
            Allocation matrix. Shape: (nmodels, npartitions)
        """
        ...
