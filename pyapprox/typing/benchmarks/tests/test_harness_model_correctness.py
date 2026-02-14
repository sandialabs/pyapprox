"""Parametrized Jacobian correctness tests over registered benchmarks.

Uses ``verify_jacobian_fd`` from ``harnesses`` to check every benchmark
that satisfies ``HasJacobian`` via finite-difference convergence.

PDE benchmarks (expensive) are gated behind ``@slow_test`` /
``@slower_test`` decorators.
"""

from typing import Any, Generic

import unittest

import numpy as np
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.test_utils import load_tests, slow_test, slower_test  # noqa: F401
from pyapprox.typing.benchmarks.registry import BenchmarkRegistry
from pyapprox.typing.benchmarks.protocols import HasJacobian
import pyapprox.typing.benchmarks.instances  # noqa: F401

from pyapprox.typing.benchmarks.tests.harnesses import verify_jacobian_fd


# ---------------------------------------------------------------------------
# Analytic benchmarks — fast
# ---------------------------------------------------------------------------

_ANALYTIC_JACOBIAN_NAMES = [
    "ishigami_3d",
    "sobol_g_6d",
    "sobol_g_4d",
    "rosenbrock_2d",
    "rosenbrock_10d",
    "branin_2d",
    "genz_oscillatory_2d",
    "genz_product_peak_2d",
    "genz_corner_peak_2d",
    "genz_gaussian_peak_2d",
    "genz_oscillatory_5d",
    "genz_gaussian_peak_5d",
]

# PDE benchmarks — progressively more expensive
_PDE_FAST_JACOBIAN_NAMES = [
    "elastic_bar_1d_linear",
    "elastic_bar_1d_hyperelastic",
]

_PDE_SLOW_JACOBIAN_NAMES = [
    "pressurized_cylinder_2d_linear",
    "pressurized_cylinder_2d_hyperelastic",
]


class TestAnalyticJacobians(Generic[Array], unittest.TestCase):
    """Jacobian FD convergence for all analytic benchmarks."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _run_jacobian_check(self, name: str) -> None:
        bm = BenchmarkRegistry.get(name, self._bkd)
        self.assertIsInstance(bm, HasJacobian)
        verify_jacobian_fd(bm, self._bkd, n_tests=3, rtol=1e-4)


def _make_analytic_test(name: str):
    def test_method(self):
        self._run_jacobian_check(name)
    test_method.__name__ = f"test_jacobian_{name}"
    test_method.__qualname__ = f"TestAnalyticJacobians.test_jacobian_{name}"
    return test_method


for _name in _ANALYTIC_JACOBIAN_NAMES:
    setattr(TestAnalyticJacobians, f"test_jacobian_{_name}", _make_analytic_test(_name))


class TestAnalyticJacobiansNumpy(TestAnalyticJacobians[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# ---------------------------------------------------------------------------
# PDE benchmarks — moderate cost (elastic bar: ~1ms)
# ---------------------------------------------------------------------------


@slow_test
class TestPDEFastJacobians(Generic[Array], unittest.TestCase):
    """Jacobian FD convergence for fast PDE benchmarks."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _run_jacobian_check(self, name: str) -> None:
        bm = BenchmarkRegistry.get(name, self._bkd)
        self.assertIsInstance(bm, HasJacobian)
        verify_jacobian_fd(bm, self._bkd, n_tests=2, rtol=1e-4)


def _make_pde_fast_test(name: str):
    def test_method(self):
        self._run_jacobian_check(name)
    test_method.__name__ = f"test_jacobian_{name}"
    test_method.__qualname__ = f"TestPDEFastJacobians.test_jacobian_{name}"
    return test_method


for _name in _PDE_FAST_JACOBIAN_NAMES:
    setattr(TestPDEFastJacobians, f"test_jacobian_{_name}", _make_pde_fast_test(_name))


class TestPDEFastJacobiansNumpy(TestPDEFastJacobians[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# ---------------------------------------------------------------------------
# PDE benchmarks — expensive (pressurized cylinder: ~0.4s)
# ---------------------------------------------------------------------------


@slower_test
class TestPDESlowJacobians(Generic[Array], unittest.TestCase):
    """Jacobian FD convergence for expensive PDE benchmarks."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _run_jacobian_check(self, name: str) -> None:
        bm = BenchmarkRegistry.get(name, self._bkd)
        self.assertIsInstance(bm, HasJacobian)
        verify_jacobian_fd(bm, self._bkd, n_tests=1, rtol=1e-3)


def _make_pde_slow_test(name: str):
    def test_method(self):
        self._run_jacobian_check(name)
    test_method.__name__ = f"test_jacobian_{name}"
    test_method.__qualname__ = f"TestPDESlowJacobians.test_jacobian_{name}"
    return test_method


for _name in _PDE_SLOW_JACOBIAN_NAMES:
    setattr(TestPDESlowJacobians, f"test_jacobian_{_name}", _make_pde_slow_test(_name))


class TestPDESlowJacobiansNumpy(TestPDESlowJacobians[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# ---------------------------------------------------------------------------
# Dynamic discovery test — verifies exhaustiveness
# ---------------------------------------------------------------------------


class TestJacobianCoverage(unittest.TestCase):
    """Verify that all HasJacobian benchmarks are covered by harness tests."""

    def test_all_jacobian_benchmarks_covered(self):
        bkd = NumpyBkd()
        registered = set(
            BenchmarkRegistry.names_satisfying(HasJacobian, bkd=bkd)
        )
        covered = set(
            _ANALYTIC_JACOBIAN_NAMES
            + _PDE_FAST_JACOBIAN_NAMES
            + _PDE_SLOW_JACOBIAN_NAMES
        )
        missing = registered - covered
        self.assertEqual(
            missing,
            set(),
            f"Benchmarks with HasJacobian not covered by harness: {missing}",
        )
