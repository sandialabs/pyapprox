"""Unit tests for Stokes and Navier-Stokes physics: state layout, the
mass matrix's block structure, and field boundary conditions.

The physics solves, on Taylor-Hood elements (P2 velocity, P1 pressure):
    -viscosity * Lap(u) + grad(p) = f_vel     (momentum)
    -div(u) = f_pres                            (continuity)

Steady and transient manufactured-solution checks are in
``tests/pde/models/galerkin/test_verification_stokes.py``.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from pyapprox.pde.galerkin.basis import LagrangeBasis, MixedBasis
from pyapprox.pde.galerkin.basis.mixed import MixedFieldBasisProtocol
from pyapprox.pde.galerkin.basis.vector_lagrange import (
    VectorLagrangeBasis,
)
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.physics.stokes import StokesPhysics
from pyapprox.util.backends.numpy import NumpyBkd

# These tests are numpy-only (numpy_bkd fixture), so generics are
# instantiated with the concrete numpy array type rather than erased.
# NDArray[Any] is the instantiation NumpyBkd itself is declared with
# (Backend[NDArray[Any]]); a floating-dtype parameter fails the
# ArrayProtocol bound.
_NumpyArray = NDArray[Any]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_dense(mat: Union[spmatrix, _NumpyArray]) -> _NumpyArray:
    """Convert sparse matrix to dense numpy array if needed."""
    if isinstance(mat, spmatrix):
        return np.asarray(mat.toarray())
    return np.asarray(mat)


def _stokes(mesh: Any, bkd: NumpyBkd, **kwargs: Any) -> StokesPhysics[_NumpyArray]:
    """Taylor-Hood Stokes on ``mesh``: P2 velocity, P1 pressure."""
    fields: List[Tuple[str, MixedFieldBasisProtocol[_NumpyArray]]] = [
        ("velocity", VectorLagrangeBasis(mesh, degree=2)),
        ("pressure", LagrangeBasis(mesh, degree=1)),
    ]
    return StokesPhysics(MixedBasis(fields), bkd, **kwargs)


# ---------------------------------------------------------------------------
# Unit Tests (dual-backend)
# ---------------------------------------------------------------------------


def _mesh_2d(bkd: NumpyBkd) -> StructuredMesh2D[_NumpyArray]:
    return StructuredMesh2D(nx=3, ny=3, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd)


class TestStokesBase:
    """Base test class for Stokes physics."""

    def test_nstates(self, numpy_bkd: NumpyBkd) -> None:
        """nstates is vel_ndofs + pres_ndofs, the mixed basis size."""
        bkd = numpy_bkd
        for mesh in (
            StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd),
            _mesh_2d(bkd),
        ):
            physics = _stokes(mesh, bkd)
            assert (
                physics.nstates()
                == physics.vel_ndofs() + physics.pres_ndofs()
                == physics.basis().ndofs()
            )

    def test_requires_velocity_then_pressure(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        mesh = _mesh_2d(bkd)
        swapped: List[Tuple[str, MixedFieldBasisProtocol[_NumpyArray]]] = [
            ("pressure", LagrangeBasis(mesh, degree=1)),
            ("velocity", VectorLagrangeBasis(mesh, degree=2)),
        ]
        with pytest.raises(ValueError, match="velocity"):
            StokesPhysics(MixedBasis(swapped), bkd)

    def test_mass_matrix_block_structure(self, numpy_bkd: NumpyBkd) -> None:
        """Test mass matrix has [M_vel, 0; 0, 0] block structure."""
        bkd = numpy_bkd
        physics = _stokes(StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd), bkd)
        M = _to_dense(physics.mass_matrix())
        n_vel = physics.vel_ndofs()
        assert M.shape == (physics.nstates(), physics.nstates())

        # Pressure block should be zero
        bkd.assert_allclose(bkd.asarray(M[n_vel:, :]), bkd.zeros(M[n_vel:, :].shape))
        bkd.assert_allclose(bkd.asarray(M[:, n_vel:]), bkd.zeros(M[:, n_vel:].shape))

        # Velocity block should be symmetric positive definite
        M_vel = M[:n_vel, :n_vel]
        bkd.assert_allclose(bkd.asarray(M_vel), bkd.asarray(M_vel.T), atol=1e-14)
        assert np.all(np.linalg.eigvalsh(M_vel) > 0)

    def test_ndim(self, numpy_bkd: NumpyBkd) -> None:
        """Test ndim returns spatial dimension."""
        bkd = numpy_bkd
        mesh1d = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=bkd)
        assert _stokes(mesh1d, bkd).ndim() == 1
        assert _stokes(_mesh_2d(bkd), bkd).ndim() == 2

    def test_field_bcs_constrain_global_dofs(self, numpy_bkd: NumpyBkd) -> None:
        """A pressure BC constrains pressure DOFs (offset by the velocity
        DOFs) and a velocity BC velocity DOFs, with each value taken
        from its own component row."""
        bkd = numpy_bkd
        physics = _stokes(_mesh_2d(bkd), bkd)
        basis = physics.basis()
        n_vel = physics.vel_ndofs()

        def inflow(coords: _NumpyArray) -> _NumpyArray:
            return np.stack([1.0 + coords[1], -2.0 + 0.0 * coords[0]])

        vel_bc = DirichletBC(basis.field("velocity"), "left", inflow, bkd)
        pres_bc = DirichletBC(basis.field("pressure"), "right", 3.0, bkd)
        vel_dofs = bkd.to_numpy(vel_bc.constrained_dofs())
        pres_dofs = bkd.to_numpy(pres_bc.constrained_dofs())
        assert np.all(vel_dofs < n_vel) and np.all(pres_dofs >= n_vel)

        coords = bkd.to_numpy(basis.dof_coordinates())
        values = bkd.to_numpy(vel_bc.constrained_values(0.0))
        expected = np.where(vel_dofs % 2 == 0, 1.0 + coords[1, vel_dofs], -2.0)
        bkd.assert_allclose(bkd.asarray(values), bkd.asarray(expected))
        bkd.assert_allclose(
            pres_bc.constrained_values(0.0),
            bkd.full((pres_dofs.shape[0],), 3.0),
        )
