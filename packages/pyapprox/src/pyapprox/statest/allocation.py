"""Allocation result types and allocators for MC/CV estimators."""

from dataclasses import dataclass
from typing import Generic, Protocol, runtime_checkable

from pyapprox.statest.cv_estimator import CVEstimator, FittedCVEstimator
from pyapprox.statest.mc_estimator import FittedMCEstimator, MCEstimator
from pyapprox.statest.statistics import (
    MultiOutputMeanAndVariance,
    MultiOutputVariance,
)
from pyapprox.util.backends.protocols import Array, Array_co


@runtime_checkable
class AllocationResultProtocol(Protocol, Generic[Array_co]):
    """Minimum interface all allocation results must satisfy."""

    @property
    def nsamples_per_model(self) -> Array_co:
        """Number of samples per model. Shape (nmodels,)."""
        ...

    @property
    def actual_cost(self) -> float:
        """Actual computational cost after rounding."""
        ...

    @property
    def success(self) -> bool:
        """Whether allocation succeeded."""
        ...

    @property
    def message(self) -> str:
        """Status message or error description."""
        ...


@dataclass(frozen=True)
class CVAllocationResult(Generic[Array]):
    """Allocation result for CV estimators.

    Attributes
    ----------
    nsamples_per_model : Array
        Number of samples per model. Shape (nmodels,).
    actual_cost : float
        Actual computational cost.
    objective_value : Array
        Objective value. Shape (1,) to preserve autograd.
    success : bool
        Whether allocation succeeded.
    message : str
        Status message.
    """

    nsamples_per_model: Array
    actual_cost: float
    objective_value: Array
    success: bool
    message: str = ""


class MCAllocator(Generic[Array]):
    """Allocator for MC estimators: floor(budget / cost)."""

    def __init__(self, template: MCEstimator[Array]) -> None:
        if not isinstance(template, MCEstimator):
            raise TypeError(
                f"MCAllocator requires MCEstimator, got {type(template).__name__}"
            )
        self._template = template
        self._bkd = template._bkd

    def allocate(self, target_cost: float) -> FittedMCEstimator[Array]:
        """Allocate samples and return a FittedMCEstimator.

        Parameters
        ----------
        target_cost : float
            Total computational budget.

        Returns
        -------
        FittedMCEstimator
            The fitted estimator with frozen allocation.
        """
        bkd = self._bkd
        nsamples = bkd.to_int(bkd.floor(target_cost / self._template._costs[0]))
        nsamples_per_model = bkd.asarray([nsamples], dtype=int)
        actual_cost = bkd.to_float(self._template._costs[0] * nsamples)
        return FittedMCEstimator(self._template, nsamples_per_model, actual_cost)


class CVAllocator(Generic[Array]):
    """Allocator for CV estimators: all models get the same sample count."""

    def __init__(self, template: CVEstimator[Array]) -> None:
        if not isinstance(template, CVEstimator):
            raise TypeError(
                f"CVAllocator requires CVEstimator, got {type(template).__name__}"
            )
        self._template = template
        self._bkd = template._bkd

    def allocate(self, target_cost: float) -> FittedCVEstimator[Array]:
        """Allocate samples and return a FittedCVEstimator.

        Parameters
        ----------
        target_cost : float
            Total computational budget.

        Returns
        -------
        FittedCVEstimator
            The fitted estimator with frozen allocation.
        """
        bkd = self._bkd
        template = self._template

        nsamples_float = target_cost / bkd.sum(template._costs)
        nsamples = bkd.to_int(bkd.floor(nsamples_float))

        variance_stats = (MultiOutputVariance, MultiOutputMeanAndVariance)
        if isinstance(template._stat, variance_stats):
            min_nhf_samples = 2
        else:
            min_nhf_samples = 1
        if nsamples < min_nhf_samples:
            raise ValueError(
                "target_cost is too small. Not enough samples of each model"
                " can be taken {0} < {1}".format(nsamples_float, min_nhf_samples)
            )

        nsamples_per_model = bkd.full((template._nmodels,), nsamples, dtype=int)
        actual_cost = bkd.to_float(bkd.sum(template._costs * nsamples_per_model))
        return FittedCVEstimator(template, nsamples_per_model, actual_cost)
