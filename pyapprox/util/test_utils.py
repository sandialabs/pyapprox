"""Shared test utilities for the typing module."""

import os
import unittest
from typing import Callable, Iterable, TypeVar, Union, cast

# Type variable for decorator that can decorate functions or classes
_T = TypeVar("_T", bound=Union[Callable[..., object], type])

# Environment variables for slow test tiers
_RUN_SLOW = os.environ.get("PYAPPROX_RUN_SLOW", "").lower() in ("1", "true", "yes")
_RUN_SLOWER = os.environ.get("PYAPPROX_RUN_SLOWER", "").lower() in (
    "1",
    "true",
    "yes",
)
_RUN_SLOWEST = os.environ.get("PYAPPROX_RUN_SLOWEST", "").lower() in (
    "1",
    "true",
    "yes",
)


def slow_test(func_or_class: _T) -> _T:
    """Mark a test as slow (>5 seconds).

    Skipped unless PYAPPROX_RUN_SLOW=1 or a higher tier is set.
    Works with both pytest and unittest.
    """
    skip_condition = not (_RUN_SLOW or _RUN_SLOWER or _RUN_SLOWEST)
    reason = "Slow test (>5s). Set PYAPPROX_RUN_SLOW=1 to run."

    try:
        import pytest

        func_or_class = pytest.mark.slow(func_or_class)
    except ImportError:
        pass

    if skip_condition:
        return cast(_T, unittest.skip(reason)(func_or_class))
    return func_or_class


def slower_test(func_or_class: _T) -> _T:
    """Mark a test as slower (>30 seconds).

    Skipped unless PYAPPROX_RUN_SLOWER=1 or PYAPPROX_RUN_SLOWEST=1.
    Works with both pytest and unittest.
    """
    skip_condition = not (_RUN_SLOWER or _RUN_SLOWEST)
    reason = "Slower test (>30s). Set PYAPPROX_RUN_SLOWER=1 to run."

    try:
        import pytest

        func_or_class = pytest.mark.slower(func_or_class)
    except ImportError:
        pass

    if skip_condition:
        return cast(_T, unittest.skip(reason)(func_or_class))
    return func_or_class


def slowest_test(func_or_class: _T) -> _T:
    """Mark a test as slowest (>60 seconds).

    Skipped unless PYAPPROX_RUN_SLOWEST=1.
    Works with both pytest and unittest.
    """
    skip_condition = not _RUN_SLOWEST
    reason = "Slowest test (>60s). Set PYAPPROX_RUN_SLOWEST=1 to run."

    try:
        import pytest

        func_or_class = pytest.mark.slowest(func_or_class)
    except ImportError:
        pass

    if skip_condition:
        return cast(_T, unittest.skip(reason)(func_or_class))
    return func_or_class


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
    # Check if this is an ACV estimator (has _recursion_index attribute)
    # This distinguishes ACVEstimator from CVEstimator/MCEstimator
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
        # MC/CV estimators - use direct method
        est.allocate_samples(target_cost)
        return None


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    """
    Exclude base classes from unittest discovery.

    Base classes should set __test__ = False to be excluded.
    This function works with both pytest (which respects __test__ natively)
    and unittest (via this load_tests protocol).

    Note: We check __dict__ directly to avoid inheritance issues. If we used
    getattr(), derived classes would inherit __test__ = False from their base.

    Usage:
        # In test file, import and re-export:
        from pyapprox.util.test_utils import load_tests

        class MyBaseTest(unittest.TestCase):
            __test__ = False
            ...

        class MyConcreteTest(MyBaseTest):
            pass  # No __test__ in __dict__, so defaults to True
    """
    suite = unittest.TestSuite()
    for group in cast(Iterable[Iterable[unittest.TestCase]], tests):
        for test in group:
            # Check __dict__ directly to avoid inheritance issues
            if test.__class__.__dict__.get('__test__', True):
                suite.addTest(test)
    return suite
