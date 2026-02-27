"""Tests verifying graceful behaviour when skfem is not installed."""

import importlib
import sys

import pytest

from pyapprox.util.optional_deps import package_available


@pytest.fixture()
def _hide_skfem(monkeypatch: pytest.MonkeyPatch):
    """Temporarily make ``import skfem`` fail."""
    monkeypatch.setitem(sys.modules, "skfem", None)
    monkeypatch.setitem(sys.modules, "skfem.element", None)
    monkeypatch.setitem(sys.modules, "skfem.models", None)
    monkeypatch.setitem(sys.modules, "skfem.models.poisson", None)
    monkeypatch.setitem(sys.modules, "skfem.models.elasticity", None)
    monkeypatch.setitem(sys.modules, "skfem.helpers", None)
    monkeypatch.setitem(sys.modules, "skfem.mesh", None)
    monkeypatch.setitem(sys.modules, "skfem.utils", None)
    monkeypatch.setitem(sys.modules, "skfem.models.general", None)

    # Clear the package_available cache so import_optional_dependency
    # re-evaluates skfem availability.
    from pyapprox.util.optional_deps import _package_cache

    _cached = _package_cache.pop("skfem", None)

    yield

    # Restore cache
    if _cached is not None:
        _package_cache["skfem"] = _cached


@pytest.mark.skipif(not package_available("skfem"), reason="skfem not installed")
class TestSkfemMissing:

    def test_structured_mesh_raises_without_skfem(
        self, _hide_skfem,
    ) -> None:
        """Reloading StructuredMesh1D module raises ImportError matching 'skfem'."""
        import pyapprox.pde.galerkin.mesh.structured as mesh_mod

        with pytest.raises(ImportError, match="skfem"):
            importlib.reload(mesh_mod)

        # Restore the module so later tests are not affected
        # (monkeypatch teardown will restore sys.modules, then reload succeeds)

    def test_lagrange_basis_raises_without_skfem(
        self, _hide_skfem,
    ) -> None:
        """Reloading LagrangeBasis module raises ImportError matching 'skfem'."""
        import pyapprox.pde.galerkin.basis.lagrange as basis_mod

        with pytest.raises(ImportError, match="skfem"):
            importlib.reload(basis_mod)
