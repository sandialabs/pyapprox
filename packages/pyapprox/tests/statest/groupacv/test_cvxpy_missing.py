"""Tests verifying graceful behaviour when cvxpy is not installed."""

import importlib
import sys

import pytest


@pytest.fixture()
def _hide_cvxpy(monkeypatch: pytest.MonkeyPatch):
    """Temporarily make ``import cvxpy`` fail, then reload optimizer module."""
    monkeypatch.setitem(sys.modules, "cvxpy", None)

    # Clear the package_available cache so import_optional_dependency
    # re-evaluates cvxpy availability.
    from pyapprox.util.optional_deps import _package_cache

    _cached = _package_cache.pop("cvxpy", None)

    import pyapprox.statest.groupacv.mlblue_optimizer as omod

    importlib.reload(omod)

    yield omod

    # Restore cache and reload so subsequent tests see the real cvxpy
    if _cached is not None:
        _package_cache["cvxpy"] = _cached
    importlib.reload(omod)


class TestCvxpyMissing:

    @pytest.mark.usefixtures("_hide_cvxpy")
    def test_module_import_succeeds_without_cvxpy(self) -> None:
        """Importing the optimizer module should never fail, even without cvxpy."""

    def test_instantiation_raises_importerror_without_cvxpy(
        self, _hide_cvxpy,
    ) -> None:
        """MLBLUESPDAllocationOptimizer() should raise ImportError."""
        omod = _hide_cvxpy
        with pytest.raises(ImportError, match="cvxpy"):
            # estimator=None is fine — import_optional_dependency is called
            # before any estimator attribute access.
            omod.MLBLUESPDAllocationOptimizer(estimator=None)
