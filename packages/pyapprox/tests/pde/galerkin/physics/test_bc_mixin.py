"""Tests for GalerkinBCMixin."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Generic, List, Optional

import numpy as np
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import (
    DirichletBC,
    RobinBC,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics.bc_mixin import GalerkinBCMixin
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend


class _ConcreteMixinUser(GalerkinBCMixin[Array], Generic[Array]):
    """Minimal class using GalerkinBCMixin for testing."""

    def __init__(
        self,
        bkd: Backend[Array],
        boundary_conditions: Optional[List[Any]] = None,
        nstates: int = 11,
    ) -> None:
        self._bkd = bkd
        self._boundary_conditions = boundary_conditions or []
        self._nstates = nstates

    def nstates(self) -> int:
        return self._nstates


def _make_basis(bkd: Backend[Any], nx: int = 10) -> LagrangeBasis[Any]:
    """Create a simple 1D Lagrange basis for testing."""
    mesh = StructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=bkd)
    return LagrangeBasis(mesh, degree=1)


class TestGalerkinBCMixin:
    def _make_user(
        self,
        bkd: Backend[Any],
        boundary_conditions: Optional[List[Any]] = None,
        nstates: int = 11,
    ) -> _ConcreteMixinUser[Any]:
        # default nstates matches _make_basis(nx=10) -> 11 DOFs
        return _ConcreteMixinUser(bkd, boundary_conditions, nstates)

    # --- dirichlet_dof_info ---

    def test_dirichlet_dof_info_no_bcs(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        user = self._make_user(bkd)
        dofs, vals = user.dirichlet_dof_info(0.0)
        assert len(bkd.to_numpy(dofs)) == 0
        assert len(bkd.to_numpy(vals)) == 0

    def test_dirichlet_dof_info_with_dirichlet(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        basis = _make_basis(bkd)
        dirichlet = DirichletBC(
            basis=basis,
            boundary_name="left",
            value_func=5.0,
            bkd=bkd,
        )
        user = self._make_user(bkd, [dirichlet])
        dofs, vals = user.dirichlet_dof_info(0.0)
        dofs_np = bkd.to_numpy(dofs)
        vals_np = bkd.to_numpy(vals)
        assert len(dofs_np) > 0
        # Left boundary of 1D mesh is DOF 0
        assert 0 in dofs_np
        bkd.assert_allclose(
            bkd.asarray(vals_np),
            bkd.asarray(np.full_like(vals_np, 5.0)),
        )

    def test_dirichlet_dof_info_skips_robin(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        basis = _make_basis(bkd)
        robin = RobinBC(
            basis=basis,
            boundary_name="left",
            alpha=1.0,
            value_func=5.0,
            bkd=bkd,
        )
        user = self._make_user(bkd, [robin])
        dofs, vals = user.dirichlet_dof_info(0.0)
        # Robin should be skipped — no Dirichlet DOFs
        assert len(bkd.to_numpy(dofs)) == 0

    def test_dirichlet_dof_info_robin_then_dirichlet(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        basis = _make_basis(bkd)
        robin = RobinBC(
            basis=basis,
            boundary_name="left",
            alpha=1.0,
            value_func=0.0,
            bkd=bkd,
        )
        dirichlet = DirichletBC(
            basis=basis,
            boundary_name="right",
            value_func=1.0,
            bkd=bkd,
        )
        user = self._make_user(bkd, [robin, dirichlet])
        dofs, vals = user.dirichlet_dof_info(0.0)
        dofs_np = bkd.to_numpy(dofs)
        # Only right boundary DOF, not left (Robin)
        n = basis.ndofs()
        assert n - 1 in dofs_np
        assert 0 not in dofs_np

