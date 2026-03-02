"""Parametrized Jacobian correctness tests over registered benchmarks.

Uses ``verify_jacobian_fd`` from ``harnesses`` to check every benchmark
that satisfies ``HasJacobian`` via finite-difference convergence.

PDE benchmarks (expensive) are gated behind ``@slow_test`` /
``@slower_test`` decorators.
"""

import pyapprox.benchmarks.instances  # noqa: F401
from pyapprox.benchmarks.protocols import HasJacobian
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.benchmarks.tests.harnesses import verify_jacobian_fd
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.test_utils import (
    slow_test,
    slower_test,
    slowest_test,
)

# ---------------------------------------------------------------------------
# Analytic benchmarks --- fast
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

# PDE benchmarks --- progressively more expensive
_PDE_FAST_JACOBIAN_NAMES = [
    "elastic_bar_1d_linear",
    "elastic_bar_1d_hyperelastic",
]

_PDE_SLOW_JACOBIAN_NAMES = [
    "pressurized_cylinder_2d_linear",
    "pressurized_cylinder_2d_hyperelastic",
]


class TestAnalyticJacobians:
    """Jacobian FD convergence for all analytic benchmarks."""

    def _run_jacobian_check(self, bkd, name: str) -> None:
        bm = BenchmarkRegistry.get(name, bkd)
        assert isinstance(bm, HasJacobian)
        verify_jacobian_fd(bm, bkd, n_tests=3, rtol=1e-4)


def _make_analytic_test(name: str):
    def test_method(self, bkd):
        self._run_jacobian_check(bkd, name)

    test_method.__name__ = f"test_jacobian_{name}"
    test_method.__qualname__ = f"TestAnalyticJacobians.test_jacobian_{name}"
    return test_method


for _name in _ANALYTIC_JACOBIAN_NAMES:
    setattr(TestAnalyticJacobians, f"test_jacobian_{_name}", _make_analytic_test(_name))


# ---------------------------------------------------------------------------
# PDE benchmarks --- moderate cost (elastic bar: ~1ms)
# ---------------------------------------------------------------------------


@slow_test
class TestPDEFastJacobians:
    """Jacobian FD convergence for fast PDE benchmarks."""

    def _run_jacobian_check(self, bkd, name: str) -> None:
        bm = BenchmarkRegistry.get(name, bkd)
        assert isinstance(bm, HasJacobian)
        verify_jacobian_fd(bm, bkd, n_tests=2, rtol=1e-4)


def _make_pde_fast_test(name: str):
    def test_method(self, bkd):
        self._run_jacobian_check(bkd, name)

    test_method.__name__ = f"test_jacobian_{name}"
    test_method.__qualname__ = f"TestPDEFastJacobians.test_jacobian_{name}"
    return test_method


for _name in _PDE_FAST_JACOBIAN_NAMES:
    setattr(TestPDEFastJacobians, f"test_jacobian_{_name}", _make_pde_fast_test(_name))


# ---------------------------------------------------------------------------
# PDE benchmarks --- expensive (pressurized cylinder: ~0.4s)
# ---------------------------------------------------------------------------


@slower_test
class TestPDESlowJacobians:
    """Jacobian FD convergence for expensive PDE benchmarks."""

    def _run_jacobian_check(self, bkd, name: str) -> None:
        bm = BenchmarkRegistry.get(name, bkd)
        assert isinstance(bm, HasJacobian)
        verify_jacobian_fd(bm, bkd, n_tests=1, rtol=1e-3)


def _make_pde_slow_test(name: str):
    def test_method(self, bkd):
        self._run_jacobian_check(bkd, name)

    test_method.__name__ = f"test_jacobian_{name}"
    test_method.__qualname__ = f"TestPDESlowJacobians.test_jacobian_{name}"
    return test_method


for _name in _PDE_SLOW_JACOBIAN_NAMES:
    setattr(TestPDESlowJacobians, f"test_jacobian_{_name}", _make_pde_slow_test(_name))


# ---------------------------------------------------------------------------
# Dynamic discovery test --- verifies exhaustiveness
# ---------------------------------------------------------------------------


@slowest_test
class TestJacobianCoverage:
    """Verify that all HasJacobian benchmarks are covered by harness tests."""

    def test_all_jacobian_benchmarks_covered(self):
        bkd = NumpyBkd()
        registered = set(BenchmarkRegistry.names_satisfying(HasJacobian, bkd=bkd))
        covered = set(
            _ANALYTIC_JACOBIAN_NAMES
            + _PDE_FAST_JACOBIAN_NAMES
            + _PDE_SLOW_JACOBIAN_NAMES
        )
        missing = registered - covered
        assert (
            missing == set()
        ), f"Benchmarks with HasJacobian not covered by harness: {missing}"
