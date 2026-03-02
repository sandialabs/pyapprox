"""Tests that verify matrices are actually sparse after the refactor.

These tests prevent silent regression back to dense storage by asserting
that physics assembly methods return scipy sparse matrices.
"""


import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from scipy.sparse import issparse

from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.basis.vector_lagrange import (
    VectorLagrangeBasis,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.physics import (
    Helmholtz,
    LinearAdvectionDiffusionReaction,
)
from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity,
)
from pyapprox.pde.galerkin.physics.euler_bernoulli import (
    EulerBernoulliBeamFEM,
)
from pyapprox.pde.galerkin.physics.stokes import StokesPhysics
from pyapprox.pde.sparse_utils import (
    solve_maybe_sparse,
    sparse_or_dense_solve,
)
from pyapprox.util.backends.numpy import NumpyBkd


class TestSparsePathADR:
    """Verify ADR physics returns sparse matrices."""

    def setup_method(self) -> None:
        bkd = NumpyBkd()
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        self._physics = LinearAdvectionDiffusionReaction(
            basis=basis,
            diffusivity=0.01,
            bkd=bkd,
        )

    def test_mass_matrix_is_sparse(self, numpy_bkd) -> None:
        _bkd = numpy_bkd
        assert issparse(self._physics.mass_matrix())

    def test_spatial_jacobian_is_sparse(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        u = bkd.asarray(np.zeros(self._physics.nstates()))
        assert issparse(self._physics.spatial_jacobian(u, 0.0))

    def test_jacobian_is_sparse(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        u = bkd.asarray(np.zeros(self._physics.nstates()))
        assert issparse(self._physics.jacobian(u, 0.0))


class TestSparsePathHelmholtz:
    """Verify Helmholtz physics returns sparse matrices."""

    def test_jacobian_is_sparse(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        bkd = NumpyBkd()
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = Helmholtz(basis=basis, wavenumber=1.0, bkd=bkd)
        u = bkd.asarray(np.zeros(physics.nstates()))
        assert issparse(physics.jacobian(u, 0.0))


class TestSparsePathElasticity:
    """Verify CompositeLinearElasticity returns sparse matrices."""

    def setup_method(self) -> None:
        bkd = NumpyBkd()
        mesh = StructuredMesh2D(
            nx=3,
            ny=3,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        self._physics = CompositeLinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )

    def test_mass_matrix_is_sparse(self, numpy_bkd) -> None:
        _bkd = numpy_bkd
        assert issparse(self._physics.mass_matrix())

    def test_stiffness_matrix_is_sparse(self, numpy_bkd) -> None:
        _bkd = numpy_bkd
        assert issparse(self._physics.stiffness_matrix())

    def test_jacobian_is_sparse(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        u = bkd.asarray(np.zeros(self._physics.nstates()))
        assert issparse(self._physics.jacobian(u, 0.0))


class TestSparsePathStokes:
    """Verify Stokes physics returns sparse matrices."""

    def setup_method(self) -> None:
        bkd = NumpyBkd()
        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=bkd)
        vel_basis = VectorLagrangeBasis(mesh, degree=2)
        pres_basis = LagrangeBasis(mesh, degree=1)
        self._physics = StokesPhysics(
            vel_basis=vel_basis,
            pres_basis=pres_basis,
            bkd=bkd,
        )

    def test_mass_matrix_is_sparse(self, numpy_bkd) -> None:
        _bkd = numpy_bkd
        assert issparse(self._physics.mass_matrix())

    def test_vel_mass_matrix_is_sparse(self, numpy_bkd) -> None:
        _bkd = numpy_bkd
        assert issparse(self._physics.vel_mass_matrix())

    def test_jacobian_is_sparse(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        u = bkd.asarray(np.zeros(self._physics.nstates()))
        assert issparse(self._physics.jacobian(u, 0.0))


class TestSparsePathEulerBernoulli:
    """Verify Euler-Bernoulli beam returns sparse matrices."""

    def setup_method(self) -> None:
        bkd = NumpyBkd()
        self._beam = EulerBernoulliBeamFEM(
            nx=5,
            length=1.0,
            EI=1.0,
            load_func=lambda x: np.ones_like(x),
            bkd=bkd,
        )

    def test_stiffness_matrix_is_sparse(self, numpy_bkd) -> None:
        _bkd = numpy_bkd
        assert issparse(self._beam.stiffness_matrix())

    def test_mass_matrix_is_sparse(self, numpy_bkd) -> None:
        _bkd = numpy_bkd
        assert issparse(self._beam.mass_matrix())

    def test_jacobian_is_sparse(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        u = bkd.asarray(np.zeros(self._beam.nstates()))
        assert issparse(self._beam.jacobian(u))


class TestSolveDispatch:
    """Verify solve_maybe_sparse dispatches correctly."""

    def test_sparse_dispatch(self, numpy_bkd) -> None:
        _bkd = numpy_bkd
        from scipy.sparse import csr_matrix

        A = csr_matrix(np.eye(3))
        b = np.array([1.0, 2.0, 3.0])
        x = sparse_or_dense_solve(A, b)
        np.testing.assert_allclose(x, b, atol=1e-14)

    def test_dense_dispatch(self, numpy_bkd) -> None:
        _bkd = numpy_bkd
        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        x = sparse_or_dense_solve(A, b)
        np.testing.assert_allclose(x, b, atol=1e-14)

    def test_solve_maybe_sparse_with_bkd(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        from scipy.sparse import csr_matrix

        bkd = NumpyBkd()
        A = csr_matrix(np.eye(3))
        b = bkd.asarray(np.array([1.0, 2.0, 3.0]))
        x = solve_maybe_sparse(bkd, A, b)
        bkd.assert_allclose(x, b, atol=1e-14)
