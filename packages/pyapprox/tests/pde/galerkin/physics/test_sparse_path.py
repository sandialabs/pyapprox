"""Tests that verify matrices are actually sparse after the refactor.

These tests prevent silent regression back to dense storage by asserting
that physics assembly methods return scipy sparse matrices.
"""


import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import List, Tuple

import numpy as np
from scipy.sparse import issparse

from pyapprox.pde.galerkin.basis import (
    CubicHermiteBasis,
    LagrangeBasis,
    MixedBasis,
)
from pyapprox.pde.galerkin.basis.mixed import MixedFieldBasisProtocol
from pyapprox.pde.galerkin.basis.vector_lagrange import (
    VectorLagrangeBasis,
)
from pyapprox.pde.galerkin.compose import compose_galerkin_system
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
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import LinearizationPoint
from pyapprox.util.linalg.sparse_dispatch import (
    solve_maybe_sparse,
    sparse_or_dense_solve,
)
from tests._helpers.adjoint_checks import NumpyArray


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
        self._system = compose_galerkin_system(self._physics)

    def test_mass_matrix_is_sparse(self, numpy_bkd: NumpyBkd) -> None:
        _bkd = numpy_bkd
        assert issparse(self._physics.mass_matrix())

    def test_spatial_jacobian_is_sparse(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        u = bkd.asarray(np.zeros(self._physics.nstates()))
        operator = self._system.spatial_operator()
        assert issparse(operator.spatial_jacobian(u, 0.0))

    def test_steady_jacobian_is_sparse(self, numpy_bkd: NumpyBkd) -> None:
        """The pinned steady residual's Jacobian is stored sparse."""
        residual = galerkin_steady_residual(skfem_discretization(self._system))
        jacobian = residual.derivatives().jacobian
        assert jacobian is not None
        point = LinearizationPoint(residual.parameters_version(), None)
        u = numpy_bkd.zeros((self._physics.nstates(), 1))
        assert issparse(jacobian(u, point).format().matrix())


class TestSparsePathHelmholtz:
    """Verify Helmholtz physics returns sparse matrices."""

    def test_jacobian_is_sparse(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        bkd = NumpyBkd()
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = Helmholtz(basis=basis, wavenumber=1.0, bkd=bkd)
        system = compose_galerkin_system(physics)
        u = bkd.asarray(np.zeros(physics.nstates()))
        assert issparse(system.spatial_operator().spatial_jacobian(u, 0.0))


class TestSparsePathElasticity:
    """Verify CompositeLinearElasticity returns sparse matrices."""

    def setup_method(self) -> None:
        bkd = NumpyBkd()
        mesh = StructuredMesh2D(
            nx=3,
            ny=3,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        self._physics = CompositeLinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )
        self._system = compose_galerkin_system(self._physics)

    def test_mass_matrix_is_sparse(self, numpy_bkd: NumpyBkd) -> None:
        _bkd = numpy_bkd
        assert issparse(self._physics.mass_matrix())

    def test_stiffness_matrix_is_sparse(self, numpy_bkd: NumpyBkd) -> None:
        _bkd = numpy_bkd
        assert issparse(self._physics.stiffness_matrix())

    def test_jacobian_is_sparse(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        u = bkd.asarray(np.zeros(self._physics.nstates()))
        operator = self._system.spatial_operator()
        assert issparse(operator.spatial_jacobian(u, 0.0))


class TestSparsePathStokes:
    """Verify Stokes physics returns sparse matrices."""

    def setup_method(self) -> None:
        bkd = NumpyBkd()
        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=bkd)
        fields: List[Tuple[str, MixedFieldBasisProtocol[NumpyArray]]] = [
            ("velocity", VectorLagrangeBasis(mesh, degree=2)),
            ("pressure", LagrangeBasis(mesh, degree=1)),
        ]
        basis = MixedBasis(fields)
        self._physics = StokesPhysics(basis, bkd)

    def test_mass_matrix_is_sparse(self, numpy_bkd: NumpyBkd) -> None:
        _bkd = numpy_bkd
        assert issparse(self._physics.mass_matrix())

    def test_vel_mass_matrix_is_sparse(self, numpy_bkd: NumpyBkd) -> None:
        _bkd = numpy_bkd
        assert issparse(self._physics.vel_mass_matrix())

    def test_jacobian_is_sparse(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        u = bkd.asarray(np.zeros(self._physics.nstates()))
        assert issparse(self._physics.interior_jacobian(u, 0.0))


class TestSparsePathEulerBernoulli:
    """Verify Euler-Bernoulli beam returns sparse matrices."""

    def setup_method(self) -> None:
        bkd = NumpyBkd()
        self._beam = EulerBernoulliBeamFEM(
            CubicHermiteBasis(StructuredMesh1D(5, (0.0, 1.0), bkd)),
            1.0,
            lambda x: np.ones_like(x),
            bkd,
        )

    def test_stiffness_matrix_is_sparse(self, numpy_bkd: NumpyBkd) -> None:
        assert issparse(self._beam.stiffness_matrix())

    def test_mass_matrix_is_sparse(self, numpy_bkd: NumpyBkd) -> None:
        assert issparse(self._beam.mass_matrix())

    def test_jacobian_is_sparse(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        u = bkd.asarray(np.zeros(self._beam.nstates()))
        assert issparse(self._beam.interior_jacobian(u))


class TestSolveDispatch:
    """Verify solve_maybe_sparse dispatches correctly."""

    def test_sparse_dispatch(self, numpy_bkd: NumpyBkd) -> None:
        _bkd = numpy_bkd
        from scipy.sparse import csr_matrix

        A = csr_matrix(np.eye(3))
        b = np.array([1.0, 2.0, 3.0])
        x = sparse_or_dense_solve(A, b)
        np.testing.assert_allclose(x, b, atol=1e-14)

    def test_dense_dispatch(self, numpy_bkd: NumpyBkd) -> None:
        _bkd = numpy_bkd
        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        x = sparse_or_dense_solve(A, b)
        np.testing.assert_allclose(x, b, atol=1e-14)

    def test_solve_maybe_sparse_with_bkd(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        from scipy.sparse import csr_matrix

        bkd = NumpyBkd()
        A = csr_matrix(np.eye(3))
        b = bkd.asarray(np.array([1.0, 2.0, 3.0]))
        x = solve_maybe_sparse(bkd, A, b)
        bkd.assert_allclose(x, b, atol=1e-14)
