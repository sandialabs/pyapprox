"""Tests for split_by_role: every BC has exactly one role."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any

from pyapprox.pde.boundary import split_by_role
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import (
    DirectDirichletBC,
    DirichletBC,
    NeumannBC,
    RobinBC,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.util.backends.protocols import Array, Backend


class _NoRole:
    """A constraint the framework has no role for (e.g. multipoint)."""

    def __repr__(self) -> str:
        return "_NoRole()"


class _BothRoles(RobinBC[Any]):
    """A Robin term that also claims to be an essential constraint."""

    def constrained_dofs(self) -> Any:
        return self.bkd().zeros((0,))

    def constrained_values(self, time: float) -> Any:
        return self.bkd().zeros((0,))

    def is_time_invariant(self) -> bool:
        return True

    def constrained_values_derivative(self, order: int) -> Any:
        return None


def _basis(bkd: Backend[Array]) -> LagrangeBasis[Array]:
    return LagrangeBasis(StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd), 1)


class TestSplitByRole:
    def test_split_keeps_order_within_each_role(
        self, bkd: Backend[Array]
    ) -> None:
        basis = _basis(bkd)
        dirichlet = DirichletBC(basis, "left", 0.0, bkd)
        robin = RobinBC(basis, "right", alpha=1.0, value_func=0.0, bkd=bkd)
        neumann = NeumannBC(basis, "right", flux_func=1.0, bkd=bkd)
        direct = DirectDirichletBC([2], [1.0], bkd)
        roles = split_by_role([robin, dirichlet, neumann, direct])
        assert roles.terms() == [robin, neumann]
        assert roles.essentials() == [dirichlet, direct]

    def test_empty(self) -> None:
        roles = split_by_role([])
        assert roles.terms() == [] and roles.essentials() == []

    def test_no_role_raises_naming_the_bc(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        with pytest.raises(TypeError, match=r"1 \(_NoRole\(\)\).*neither"):
            split_by_role(
                [DirectDirichletBC([0], [0.0], numpy_bkd), _NoRole()]
            )

    def test_both_roles_raises(self, bkd: Backend[Array]) -> None:
        bc = _BothRoles(_basis(bkd), "left", alpha=1.0, value_func=0.0, bkd=bkd)
        with pytest.raises(TypeError, match="both"):
            split_by_role([bc])

    def test_physics_construction_rejects_unknown_role(
        self, bkd: Backend[Array]
    ) -> None:
        """Fails when the physics is built, not at the first solve."""
        with pytest.raises(TypeError, match="neither"):
            AdvectionDiffusionReaction(
                basis=_basis(bkd),
                diffusivity=1.0,
                bkd=bkd,
                boundary_conditions=[_NoRole()],
            )
