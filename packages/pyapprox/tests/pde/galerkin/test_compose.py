"""Tests for composing a Galerkin physics with its boundary conditions."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, List

import numpy as np
from pyapprox.pde.boundary import BoundaryConditionRole
from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import DirichletBC, RobinBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.protocols.physics import GalerkinPhysicsProtocol
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


def _forcing(x: Any) -> Any:
    return np.sin(np.pi * x[0])


def _setup(
    bkd: Backend[Array],
) -> tuple[
    LagrangeBasis[Array], List[BoundaryConditionRole[Array]], Array
]:
    mesh = StructuredMesh1D(nx=8, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=2)
    bcs: List[BoundaryConditionRole[Array]] = [
        DirichletBC(basis, "left", 0.5, bkd),
        RobinBC(basis, "right", 2.0, 1.0, bkd),
    ]
    state = bkd.asarray(np.cos(np.linspace(0.0, 1.0, basis.ndofs())))
    return basis, bcs, state


def _physics(
    basis: LagrangeBasis[Array], bkd: Backend[Array]
) -> AdvectionDiffusionReaction[Array]:
    return AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=0.3,
        bkd=bkd,
        forcing=TimeIndependent(_forcing),
    )


def _dense(matrix: Any, bkd: Backend[Array]) -> Array:
    if hasattr(matrix, "toarray"):
        return bkd.asarray(matrix.toarray())
    return bkd.asarray(matrix)


class TestComposeGalerkinSystem:
    def test_parts_follow_the_roles(self, bkd: Backend[Array]) -> None:
        """F is the interior plus the Robin term, the constraints are the
        Dirichlet DOFs and values, and the mass is the physics'."""
        basis, bcs, state = _setup(bkd)
        dirichlet, robin = bcs
        assert isinstance(dirichlet, DirichletBC)
        assert isinstance(robin, RobinBC)
        physics = _physics(basis, bkd)
        system = compose_galerkin_system(physics, bcs)
        assert isinstance(system, GalerkinTransientSystemProtocol)
        operator = system.spatial_operator()
        bkd.assert_allclose(
            operator.spatial_residual(state, 0.0),
            robin.apply_to_residual(
                physics.interior_residual(state, 0.0), state, 0.0
            ),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            _dense(operator.spatial_jacobian(state, 0.0), bkd),
            _dense(
                robin.apply_to_jacobian(
                    physics.interior_jacobian(state, 0.0), state, 0.0
                ),
                bkd,
            ),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            system.constraint_set().dofs(), dirichlet.boundary_dofs()
        )
        bkd.assert_allclose(
            system.constraint_set().values(0.0),
            dirichlet.boundary_values(0.0),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            _dense(system.mass_matrix(), bkd),
            _dense(physics.mass_matrix(), bkd),
            rtol=1e-12,
        )

    def test_without_bcs_is_the_interior(self, bkd: Backend[Array]) -> None:
        basis, _, state = _setup(bkd)
        physics = _physics(basis, bkd)
        system = compose_galerkin_system(physics)
        assert system.constraint_set().ndofs() == 0
        bkd.assert_allclose(
            system.spatial_operator().spatial_residual(state, 0.0),
            physics.interior_residual(state, 0.0),
            rtol=1e-12,
        )

    def test_system_owns_physics_and_bcs(self, bkd: Backend[Array]) -> None:
        """Parameterizations bind by membership, so the system must own
        the physics and its natural terms."""
        basis, bcs, _ = _setup(bkd)
        physics = _physics(basis, bkd)
        system = compose_galerkin_system(physics, bcs)
        assert system.owns(physics)
        assert system.owns(bcs[1])

    def test_rejects_non_physics(self, numpy_bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="GalerkinPhysicsProtocol"):
            compose_galerkin_system(object())  # type: ignore[arg-type]

    def test_rejects_bc_with_no_role(self, numpy_bkd: Backend[Array]) -> None:
        basis, _, _ = _setup(numpy_bkd)
        physics = _physics(basis, numpy_bkd)
        assert isinstance(physics, GalerkinPhysicsProtocol)
        with pytest.raises(TypeError, match="satisfies neither"):
            compose_galerkin_system(
                physics, [object()]  # type: ignore[list-item]
            )
