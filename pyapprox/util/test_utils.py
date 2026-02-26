"""Shared test utilities."""

import unittest
from typing import Iterable, cast

import pytest

# Convenience aliases — apply the marker with "*" (all backends).
# Existing code using @slow_test continues to work unchanged.
slow_test = pytest.mark.slow_on("*")
slower_test = pytest.mark.slower_on("*")
slowest_test = pytest.mark.slowest_on("*")


# Kept during incremental migration — will be removed in Phase 13.
def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    """Exclude base classes from unittest discovery."""
    suite = unittest.TestSuite()
    for group in cast(Iterable[Iterable[unittest.TestCase]], tests):
        for test in group:
            if test.__class__.__dict__.get("__test__", True):
                suite.addTest(test)
    return suite


def allocate_with_allocator(est, target_cost: float):
    """Allocate samples using the new allocator API.

    This function replaces `est.allocate_samples(target_cost)` for all
    estimators. For ACV estimators, it uses the new allocator/estimator
    split pattern. For MC/CV estimators (which don't have allocators),
    it falls back to the direct `allocate_samples()` method.

    Parameters
    ----------
    est : Estimator
        The estimator to allocate samples for.
    target_cost : float
        The total computational budget.

    Returns
    -------
    AllocationResult or None
        The allocation result for ACV estimators, None for MC/CV.

    Raises
    ------
    RuntimeError
        If allocation fails.
    """
    from pyapprox.statest.acv.base import ACVEstimator

    if isinstance(est, ACVEstimator):
        from pyapprox.statest.acv.allocation import default_allocator_factory

        allocator = default_allocator_factory(est)
        result = allocator.allocate(target_cost)
        if not result.success:
            raise RuntimeError(f"Allocation failed: {result.message}")
        est.set_allocation(result)
        return result
    else:
        est.allocate_samples(target_cost)
        return None
