"""Tests for the duplicate-OpenMP warning on the SDP allocator.

Every test here fabricates the dangerous arrangement rather than
creating it. Two OpenMP runtimes in one process abort the interpreter
outright on macOS, so a test that genuinely installed them would take
the test session down with it, with no traceback to report. The check
under test is a pure function of the filesystem that never imports the
packages it inspects, which is exactly what makes it safe to drive from
fabricated paths.
"""

import sys
import warnings
from typing import List

import pytest
from pyapprox.statest.groupacv import mlblue_optimizer as omod


def _runtime_warnings(solver_name: str) -> List[warnings.WarningMessage]:
    """Return the RuntimeWarnings the guard emits for ``solver_name``.

    ``pytest.warns(None)`` was the natural way to assert that nothing is
    warned, but it is deprecated, so capture explicitly instead. Filters
    are reset to "always" because a warning already emitted once in the
    session would otherwise be suppressed, turning a real regression
    into a passing test.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        omod._warn_on_duplicate_openmp(solver_name)
    return [
        w for w in record if issubclass(w.category, RuntimeWarning)
    ]


def _fake_runtimes(cvxopt_paths, torch_paths):
    """Stand in for _vendored_openmp_runtimes over the two packages."""

    def _lookup(module_name: str):
        if module_name == "cvxopt":
            return list(cvxopt_paths)
        if module_name == "torch":
            return list(torch_paths)
        return []

    return _lookup


class TestVendoredOpenmpDiscovery:
    """The discovery helper must not import what it inspects."""

    def test_absent_package_reports_no_runtimes(self) -> None:
        """A package that is not installed contributes nothing."""
        assert omod._vendored_openmp_runtimes("not_a_real_package_xyz") == []

    def test_discovery_does_not_import_the_package(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Importing is the operation that aborts, so it must not happen.

        The helper exists to be called before a package is loaded. If it
        imported the package to find its files, it would trigger the
        failure it is meant to predict.
        """
        monkeypatch.delitem(sys.modules, "cvxopt", raising=False)
        omod._vendored_openmp_runtimes("cvxopt")
        assert "cvxopt" not in sys.modules


class TestDuplicateOpenmpWarning:
    """The warning fires only for the arrangement that actually aborts."""

    def test_warns_when_both_packages_vendor_a_runtime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both vendoring a runtime is the case that kills the process."""
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(
            omod,
            "_vendored_openmp_runtimes",
            _fake_runtimes(["/x/cvxopt/.dylibs/libomp.dylib"],
                           ["/x/torch/lib/libomp.dylib"]),
        )
        with pytest.warns(RuntimeWarning, match="OpenMP"):
            omod._warn_on_duplicate_openmp("CVXOPT")

    def test_warning_names_both_paths_and_the_fix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A warning without the paths or the remedy is not actionable.

        The abort leaves no traceback, so this text is the only thing
        the user gets to work from.
        """
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(
            omod,
            "_vendored_openmp_runtimes",
            _fake_runtimes(["/x/cvxopt/.dylibs/libomp.dylib"],
                           ["/x/torch/lib/libomp.dylib"]),
        )
        with pytest.warns(RuntimeWarning) as record:
            omod._warn_on_duplicate_openmp("CVXOPT")
        message = str(record[0].message)
        assert "/x/cvxopt/.dylibs/libomp.dylib" in message
        assert "/x/torch/lib/libomp.dylib" in message
        assert "conda-forge" in message
        assert "CLARABEL" in message

    def test_silent_when_only_torch_vendors_a_runtime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One runtime is the normal, working case.

        This is what a conda-forge cvxopt looks like, and it is the
        arrangement the recommended fix produces. Warning here would
        train users to ignore the warning.
        """
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(
            omod,
            "_vendored_openmp_runtimes",
            _fake_runtimes([], ["/x/torch/lib/libomp.dylib"]),
        )
        assert _runtime_warnings("CVXOPT") == []

    def test_silent_without_torch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No collision is possible when only one package is present."""
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(
            omod,
            "_vendored_openmp_runtimes",
            _fake_runtimes(["/x/cvxopt/.dylibs/libomp.dylib"], []),
        )
        assert _runtime_warnings("CVXOPT") == []

    def test_silent_for_clarabel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLARABEL needs no OpenMP runtime, so it cannot collide."""
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(
            omod,
            "_vendored_openmp_runtimes",
            _fake_runtimes(["/x/cvxopt/.dylibs/libomp.dylib"],
                           ["/x/torch/lib/libomp.dylib"]),
        )
        assert _runtime_warnings("CLARABEL") == []

    @pytest.mark.parametrize("platform", ["linux", "win32"])
    def test_silent_off_macos(
        self, monkeypatch: pytest.MonkeyPatch, platform: str
    ) -> None:
        """The abort is a macOS behavior; elsewhere pip cvxopt is fine.

        Warning on every platform would make the message wrong for most
        users, and a warning that is usually wrong gets muted along with
        the cases where it is right.
        """
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setattr(
            omod,
            "_vendored_openmp_runtimes",
            _fake_runtimes(["/x/cvxopt/.dylibs/libomp.so"],
                           ["/x/torch/lib/libomp.so"]),
        )
        assert _runtime_warnings("CVXOPT") == []


class TestGuardRunsAtConstruction:
    """The check must run before a solve, not during one."""

    def test_constructor_consults_the_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checking at solve time would be too late.

        The abort happens when the solver backend loads, so the warning
        has to reach the user before that. Pinning the call site keeps a
        later refactor from moving it past the point of no return.
        """
        pytest.importorskip("cvxpy")
        from pyapprox.statest.groupacv import MLBLUEEstimator
        from pyapprox.statest.statistics import MultiOutputMean
        from pyapprox.util.backends.numpy import NumpyBkd

        seen = []
        monkeypatch.setattr(
            omod, "_warn_on_duplicate_openmp", lambda name: seen.append(name)
        )

        bkd = NumpyBkd()
        cov = bkd.array([[1.0, 0.9, 0.8], [0.9, 1.0, 0.7], [0.8, 0.7, 1.0]])
        costs = bkd.array([4.0, 2.0, 1.0])
        stat = MultiOutputMean(1, bkd)
        stat.set_pilot_quantities(cov)
        est = MLBLUEEstimator(stat, costs)

        omod.MLBLUESPDAllocationOptimizer(est, solver_name="CVXOPT")
        assert seen == ["CVXOPT"]
