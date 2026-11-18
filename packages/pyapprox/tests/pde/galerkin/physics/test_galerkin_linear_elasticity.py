"""Tests for LinearElasticity physics."""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)


import numpy as np
from scipy.sparse import issparse

from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
    StructuredMesh3D,
)
from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity,
)


def _to_dense(mat, bkd):
    """Convert a matrix (possibly sparse) to a dense numpy array."""
    if issparse(mat):
        return mat.toarray()
    return bkd.to_numpy(mat)


# Alias for backward compatibility in tests
LinearElasticity = CompositeLinearElasticity


class TestLinearElasticityBase:
    """Base test class for LinearElasticity."""
    def test_1d_stiffness_symmetric(self, numpy_bkd) -> None:
        """Test stiffness matrix is symmetric in 1D."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(
            nx=5,
            bounds=(0.0, 1.0),
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )
        K = physics.stiffness_matrix()
        K_np = _to_dense(K, bkd)
        bkd.assert_allclose(K_np, K_np.T, rtol=0.0, atol=1.5e-6)


    def test_1d_rigid_body_motion(self, numpy_bkd) -> None:
        """Test that constant displacement gives zero stiffness action in 1D."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(
            nx=5,
            bounds=(0.0, 1.0),
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )
        u = bkd.asarray(np.ones(physics.nstates()))
        K = physics.stiffness_matrix()
        Ku = _to_dense(K, bkd) @ bkd.to_numpy(u)
        assert np.linalg.norm(Ku) < 1e-10


    def test_2d_stiffness_symmetric(self, numpy_bkd) -> None:
        """Test stiffness matrix is symmetric in 2D."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )

        K = physics.stiffness_matrix()
        K_np = _to_dense(K, bkd)

        bkd.assert_allclose(K_np, K_np.T, rtol=0.0, atol=1.5e-6)

    def test_2d_mass_matrix_symmetric(self, numpy_bkd) -> None:
        """Test mass matrix is symmetric in 2D."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )

        M = physics.mass_matrix()
        M_np = _to_dense(M, bkd)

        bkd.assert_allclose(M_np, M_np.T, rtol=0.0, atol=1.5e-6)


    def test_3d_stiffness_symmetric(self, numpy_bkd) -> None:
        """Test stiffness matrix is symmetric in 3D."""
        bkd = numpy_bkd
        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )

        K = physics.stiffness_matrix()
        K_np = _to_dense(K, bkd)

        bkd.assert_allclose(K_np, K_np.T, rtol=0.0, atol=1.5e-6)

    def test_3d_mass_matrix_symmetric(self, numpy_bkd) -> None:
        """Test mass matrix is symmetric in 3D."""
        bkd = numpy_bkd
        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )

        M = physics.mass_matrix()
        M_np = _to_dense(M, bkd)

        bkd.assert_allclose(M_np, M_np.T, rtol=0.0, atol=1.5e-6)


    def test_lame_parameters(self, numpy_bkd) -> None:
        """Test Lame parameter computation."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=3,
            ny=3,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)

        E = 100.0
        nu = 0.25

        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=bkd,
        )

        # Expected Lame parameters
        # lambda = E * nu / ((1 + nu) * (1 - 2*nu))
        # mu = E / (2 * (1 + nu))
        expected_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
        expected_mu = E / (2 * (1 + nu))

        assert abs(physics.lame_lambda() - expected_lambda) < 10**(-10)
        assert abs(physics.lame_mu() - expected_mu) < 10**(-10)

    def test_poisson_ratio_validation(self, numpy_bkd) -> None:
        """Test that invalid Poisson ratios raise errors."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=3,
            ny=3,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)

        # nu = 0.5 is invalid (incompressible limit)
        with pytest.raises(ValueError):
            LinearElasticity.from_uniform(
                basis=basis,
                youngs_modulus=1.0,
                poisson_ratio=0.5,
                bkd=bkd,
            )

        # nu = -1 is invalid
        with pytest.raises(ValueError):
            LinearElasticity.from_uniform(
                basis=basis,
                youngs_modulus=1.0,
                poisson_ratio=-1.0,
                bkd=bkd,
            )

    def test_stiffness_action_consistency(self, numpy_bkd) -> None:
        """Test that K*u gives expected result for constant strain field.

        For u = [ax, ay] (uniform scaling), K*u should represent the
        boundary traction integrals since -div(sigma) = 0 in the interior.
        """
        bkd = numpy_bkd
        E = 1.0  # Young's modulus
        nu = 0.3  # Poisson's ratio

        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=bkd,
        )

        # Linear displacement u = [x, y] -> constant strain
        def u_linear(x):
            return x

        u = physics.initial_condition(u_linear)
        K = physics.stiffness_matrix()
        K_np = _to_dense(K, bkd)
        u_np = bkd.to_numpy(u)

        # K*u should be consistent (non-zero due to boundary tractions)
        Ku = K_np @ u_np
        assert Ku.shape == (physics.nstates(),)
        # For uniform strain, the response should be consistent
        assert np.linalg.norm(Ku) > 0

    def test_stiffness_action_consistency_3d(self, numpy_bkd) -> None:
        """Test that K*u gives expected result for constant strain field in 3D."""
        bkd = numpy_bkd
        E = 1.0  # Young's modulus
        nu = 0.25  # Poisson's ratio

        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=bkd,
        )

        # Linear displacement u = [x, y, z] -> constant strain
        def u_linear(x):
            return x

        u = physics.initial_condition(u_linear)
        K = physics.stiffness_matrix()
        K_np = _to_dense(K, bkd)
        u_np = bkd.to_numpy(u)

        # K*u should be consistent (non-zero due to boundary tractions)
        Ku = K_np @ u_np
        assert Ku.shape == (physics.nstates(),)
        # For uniform strain, the response should be consistent
        assert np.linalg.norm(Ku) > 0

    def test_rigid_body_motion_2d(self, numpy_bkd) -> None:
        """Test that rigid body motion gives zero strain energy.

        Pure translation u = [c, c] should give zero K*u.
        """
        bkd = numpy_bkd
        E = 1.0
        nu = 0.3

        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=bkd,
        )

        # Constant translation
        def u_translation(x):
            result = np.zeros_like(x)
            result[0, :] = 1.0  # Translate in x
            result[1, :] = 2.0  # Translate in y
            return result

        u = physics.initial_condition(u_translation)
        K = physics.stiffness_matrix()
        K_np = _to_dense(K, bkd)
        u_np = bkd.to_numpy(u)

        # K*u should be zero for rigid body motion
        Ku = K_np @ u_np
        assert np.linalg.norm(Ku) < 1e-10

    def test_rigid_body_motion_3d(self, numpy_bkd) -> None:
        """Test that rigid body motion gives zero strain energy in 3D."""
        bkd = numpy_bkd
        E = 1.0
        nu = 0.25

        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=bkd,
        )

        # Constant translation
        def u_translation(x):
            result = np.zeros_like(x)
            result[0, :] = 1.0
            result[1, :] = 2.0
            result[2, :] = 3.0
            return result

        u = physics.initial_condition(u_translation)
        K = physics.stiffness_matrix()
        K_np = _to_dense(K, bkd)
        u_np = bkd.to_numpy(u)

        # K*u should be zero for rigid body motion
        Ku = K_np @ u_np
        assert np.linalg.norm(Ku) < 1e-10


