"""Parametrized forward-UQ tests over registered benchmarks.

Uses ``verify_forward_uq_mean`` and ``verify_forward_uq_variance`` from
``harnesses`` to check benchmarks satisfying ``HasReferenceMean`` /
``HasReferenceVariance`` against Monte Carlo estimates.
"""

import unittest
from typing import Any, Generic

from numpy.typing import NDArray

import pyapprox.benchmarks.instances  # noqa: F401
from pyapprox.benchmarks.protocols import (
    HasPrior,
    HasReferenceMean,
    HasReferenceVariance,
)
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.benchmarks.tests.harnesses import (
    MCMeanEstimator,
    MCVarianceEstimator,
    verify_forward_uq_mean,
    verify_forward_uq_variance,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array
from pyapprox.util.test_utils import load_tests, slow_test  # noqa: F401

# Benchmarks with HasReferenceMean + HasPrior
_MEAN_BENCHMARK_NAMES = [
    "ishigami_3d",
    "sobol_g_6d",
    "sobol_g_4d",
]

# Benchmarks with HasReferenceVariance + HasPrior
_VARIANCE_BENCHMARK_NAMES = [
    "ishigami_3d",
    "sobol_g_6d",
    "sobol_g_4d",
]


# ---------------------------------------------------------------------------
# CI-level tests — moderate sample count, relaxed tolerance
# ---------------------------------------------------------------------------


class TestForwardUQMean(Generic[Array], unittest.TestCase):
    """MC mean verification for benchmarks with HasReferenceMean."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _run_mean_check(self, name: str) -> None:
        bm = BenchmarkRegistry.get(name, self._bkd)
        self.assertIsInstance(bm, HasReferenceMean)
        self.assertIsInstance(bm, HasPrior)
        verify_forward_uq_mean(
            bm,
            self._bkd,
            strategy=MCMeanEstimator(n_samples=50000, seed=42),
            rtol=5e-2,
        )


def _make_mean_test(name: str):
    def test_method(self):
        self._run_mean_check(name)

    test_method.__name__ = f"test_mean_{name}"
    test_method.__qualname__ = f"TestForwardUQMean.test_mean_{name}"
    return test_method


for _name in _MEAN_BENCHMARK_NAMES:
    setattr(TestForwardUQMean, f"test_mean_{_name}", _make_mean_test(_name))


class TestForwardUQMeanNumpy(TestForwardUQMean[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# ---------------------------------------------------------------------------
# CI-level variance tests
# ---------------------------------------------------------------------------


class TestForwardUQVariance(Generic[Array], unittest.TestCase):
    """MC variance verification for benchmarks with HasReferenceVariance."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _run_variance_check(self, name: str) -> None:
        bm = BenchmarkRegistry.get(name, self._bkd)
        self.assertIsInstance(bm, HasReferenceVariance)
        self.assertIsInstance(bm, HasPrior)
        verify_forward_uq_variance(
            bm,
            self._bkd,
            strategy=MCVarianceEstimator(n_samples=50000, seed=42),
            rtol=5e-2,
        )


def _make_variance_test(name: str):
    def test_method(self):
        self._run_variance_check(name)

    test_method.__name__ = f"test_variance_{name}"
    test_method.__qualname__ = f"TestForwardUQVariance.test_variance_{name}"
    return test_method


for _name in _VARIANCE_BENCHMARK_NAMES:
    setattr(TestForwardUQVariance, f"test_variance_{_name}", _make_variance_test(_name))


class TestForwardUQVarianceNumpy(TestForwardUQVariance[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# ---------------------------------------------------------------------------
# Nightly tests — high sample count, tight tolerance
# ---------------------------------------------------------------------------


@slow_test
class TestForwardUQMeanNightly(Generic[Array], unittest.TestCase):
    """Nightly MC mean verification with tighter tolerances."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _run_mean_check(self, name: str) -> None:
        bm = BenchmarkRegistry.get(name, self._bkd)
        verify_forward_uq_mean(
            bm,
            self._bkd,
            strategy=MCMeanEstimator(n_samples=500000, seed=42),
            rtol=5e-3,
        )


def _make_nightly_mean_test(name: str):
    def test_method(self):
        self._run_mean_check(name)

    test_method.__name__ = f"test_mean_nightly_{name}"
    test_method.__qualname__ = f"TestForwardUQMeanNightly.test_mean_nightly_{name}"
    return test_method


for _name in _MEAN_BENCHMARK_NAMES:
    setattr(
        TestForwardUQMeanNightly,
        f"test_mean_nightly_{_name}",
        _make_nightly_mean_test(_name),
    )


class TestForwardUQMeanNightlyNumpy(TestForwardUQMeanNightly[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


@slow_test
class TestForwardUQVarianceNightly(Generic[Array], unittest.TestCase):
    """Nightly MC variance verification with tighter tolerances."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _run_variance_check(self, name: str) -> None:
        bm = BenchmarkRegistry.get(name, self._bkd)
        verify_forward_uq_variance(
            bm,
            self._bkd,
            strategy=MCVarianceEstimator(n_samples=500000, seed=42),
            rtol=5e-3,
        )


def _make_nightly_variance_test(name: str):
    def test_method(self):
        self._run_variance_check(name)

    test_method.__name__ = f"test_variance_nightly_{name}"
    test_method.__qualname__ = (
        f"TestForwardUQVarianceNightly.test_variance_nightly_{name}"
    )
    return test_method


for _name in _VARIANCE_BENCHMARK_NAMES:
    setattr(
        TestForwardUQVarianceNightly,
        f"test_variance_nightly_{_name}",
        _make_nightly_variance_test(_name),
    )


class TestForwardUQVarianceNightlyNumpy(TestForwardUQVarianceNightly[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# ---------------------------------------------------------------------------
# Dynamic coverage test
# ---------------------------------------------------------------------------


class TestForwardUQCoverage(unittest.TestCase):
    """Verify all HasReferenceMean/HasReferenceVariance benchmarks covered."""

    def test_all_mean_benchmarks_covered(self):
        bkd = NumpyBkd()
        registered = set(
            BenchmarkRegistry.names_satisfying(HasReferenceMean, HasPrior, bkd=bkd)
        )
        covered = set(_MEAN_BENCHMARK_NAMES)
        missing = registered - covered
        self.assertEqual(
            missing,
            set(),
            f"Benchmarks with HasReferenceMean+HasPrior not covered: {missing}",
        )

    def test_all_variance_benchmarks_covered(self):
        bkd = NumpyBkd()
        registered = set(
            BenchmarkRegistry.names_satisfying(HasReferenceVariance, HasPrior, bkd=bkd)
        )
        covered = set(_VARIANCE_BENCHMARK_NAMES)
        missing = registered - covered
        self.assertEqual(
            missing,
            set(),
            f"Benchmarks with HasReferenceVariance+HasPrior not covered: {missing}",
        )
