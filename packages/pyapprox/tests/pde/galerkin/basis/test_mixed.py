"""MixedBasis: field offsets, global boundary DOFs, per-component values.

A field view must answer in the mixed system's global DOF indices, and a
vector field's components must come from its own basis, not from the
global index: with the vector field placed after a field of odd size,
``global_dof % ncomponents`` picks the wrong component.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.pde.galerkin.basis import (
    LagrangeBasis,
    MixedBasis,
    VectorLagrangeBasis,
)
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.protocols.basis import (
    BoundaryDofsProtocol,
    ComponentDofsBasisProtocol,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend


def _mesh(bkd: Backend[Array]) -> StructuredMesh2D[Array]:
    return StructuredMesh2D(2, 2, [[0.0, 1.0], [0.0, 1.0]], bkd)


def _velocity_rows(coords: np.ndarray) -> np.ndarray:
    """One row per component: (1 + y, 2 x)."""
    return np.stack([1.0 + coords[1], 2.0 * coords[0]])


class TestMixedBasis:
    def test_offsets_and_coordinates(self, bkd: Backend[Array]) -> None:
        mesh = _mesh(bkd)
        vel = VectorLagrangeBasis(mesh, degree=2)
        pres = LagrangeBasis(mesh, degree=1)
        mixed = MixedBasis([("velocity", vel), ("pressure", pres)])
        assert mixed.field_names() == ["velocity", "pressure"]
        assert mixed.offset("velocity") == 0
        assert mixed.offset("pressure") == vel.ndofs()
        assert mixed.ndofs() == vel.ndofs() + pres.ndofs()
        bkd.assert_allclose(
            mixed.dof_coordinates(),
            bkd.hstack([vel.dof_coordinates(), pres.dof_coordinates()]),
        )

    def test_field_views_answer_in_global_dofs(
        self, bkd: Backend[Array]
    ) -> None:
        mesh = _mesh(bkd)
        vel = VectorLagrangeBasis(mesh, degree=2)
        pres = LagrangeBasis(mesh, degree=1)
        mixed = MixedBasis([("velocity", vel), ("pressure", pres)])
        pressure = mixed.field("pressure")
        velocity = mixed.field("velocity")
        assert isinstance(pressure, BoundaryDofsProtocol)
        assert not isinstance(pressure, ComponentDofsBasisProtocol)
        assert isinstance(velocity, ComponentDofsBasisProtocol)
        bkd.assert_allclose(
            pressure.get_dofs("left"), pres.get_dofs("left") + vel.ndofs()
        )
        bkd.assert_allclose(
            velocity.get_dofs("left", components=(1,)),
            vel.get_dofs("left", components=(1,)),
        )

    def test_components_from_the_field_at_any_offset(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Velocity after a scalar field of odd size: each DOF takes its
        own component row of a (2, npts) boundary function."""
        bkd = numpy_bkd
        mesh = _mesh(bkd)
        pres = LagrangeBasis(mesh, degree=1)
        assert pres.ndofs() % 2 == 1
        vel = VectorLagrangeBasis(mesh, degree=2)
        mixed = MixedBasis([("pressure", pres), ("velocity", vel)])
        bc = DirichletBC(mixed.field("velocity"), "left", _velocity_rows, bkd)
        dofs = bkd.to_numpy(bc.constrained_dofs())
        local = dofs - pres.ndofs()
        coords = bkd.to_numpy(mixed.dof_coordinates())[:, dofs]
        expected = np.where(
            local % 2 == 0, 1.0 + coords[1], 2.0 * coords[0]
        )
        bkd.assert_allclose(
            bc.constrained_values(0.0), bkd.asarray(expected), rtol=1e-14
        )
        # The global-index rule would have swapped the components.
        assert np.any((dofs % 2) != (local % 2))

    def test_invalid_fields_raise(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        mesh = _mesh(bkd)
        pres = LagrangeBasis(mesh, degree=1)
        with pytest.raises(ValueError, match="at least one"):
            MixedBasis([])
        with pytest.raises(ValueError, match="unique"):
            MixedBasis([("p", pres), ("p", pres)])
        with pytest.raises(ValueError, match="share one mesh"):
            MixedBasis([("p", pres), ("q", LagrangeBasis(_mesh(bkd)))])
        with pytest.raises(KeyError, match="no field"):
            MixedBasis([("p", pres)]).field("velocity")
