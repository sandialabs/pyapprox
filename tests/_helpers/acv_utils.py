"""ACV test utilities."""
from __future__ import annotations

from typing import Any, Union

from pyapprox.statest.acv.base import ACVEstimator, FittedACVEstimator
from pyapprox.statest.acv.allocation import default_allocator_factory
from pyapprox.statest.cv_estimator import CVEstimator, FittedCVEstimator
from pyapprox.statest.mc_estimator import MCEstimator, FittedMCEstimator
from pyapprox.statest.allocation import MCAllocator, CVAllocator


def allocate_with_allocator(
    template: Any, target_cost: float,
) -> Union[FittedMCEstimator, FittedCVEstimator, FittedACVEstimator]:
    """Allocate samples and return a fitted estimator.

    Routes to the correct allocator based on estimator type:
    - ACVEstimator -> default_allocator_factory -> FittedACVEstimator
    - CVEstimator -> CVAllocator -> FittedCVEstimator
    - MCEstimator -> MCAllocator -> FittedMCEstimator

    Parameters
    ----------
    template : Estimator
        The template estimator.
    target_cost : float
        The total computational budget.

    Returns
    -------
    FittedMCEstimator or FittedCVEstimator or FittedACVEstimator
        The fitted estimator with frozen allocation.
    """
    if isinstance(template, ACVEstimator):
        allocator = default_allocator_factory(template)
        result = allocator.allocate(target_cost)
        if not result.success:
            raise RuntimeError(f"Allocation failed: {result.message}")
        return FittedACVEstimator(template, result)
    elif isinstance(template, CVEstimator):
        return CVAllocator(template).allocate(target_cost)
    else:
        return MCAllocator(template).allocate(target_cost)
