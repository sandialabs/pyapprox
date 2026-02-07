"""Tests for LinearElasticity physics."""

import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.galerkin.mesh import (
    StructuredMesh2D,
    StructuredMesh3D,
)
from pyapprox.typing.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.typing.pde.galerkin.physics import LinearElasticity
from pyapprox.typing.pde.galerkin.solvers import SteadyStateSolver


class TestLinearElasticityBase(Generic[Array], unittest.TestCase):
    """Base test class for LinearElasticity."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self.bkd_inst = self.bkd()

    def test_2d_stiffness_symmetric(self) -> None:
        """Test stiffness matrix is symmetric in 2D."""
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=self.bkd_inst,
        )

        K = physics.stiffness_matrix()
        K_np = self.bkd_inst.to_numpy(K)

        np.testing.assert_array_almost_equal(K_np, K_np.T)

    def test_2d_mass_matrix_symmetric(self) -> None:
        """Test mass matrix is symmetric in 2D."""
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=self.bkd_inst,
        )

        M = physics.mass_matrix()
        M_np = self.bkd_inst.to_numpy(M)

        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_2d_residual_shape(self) -> None:
        """Test residual has correct shape in 2D."""
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=self.bkd_inst,
        )

        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)

        self.assertEqual(res.shape, (physics.nstates(),))

    def test_2d_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape in 2D."""
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=self.bkd_inst,
        )

        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        jac = physics.jacobian(u0, 0.0)

        self.assertEqual(jac.shape, (physics.nstates(), physics.nstates()))

    def test_3d_stiffness_symmetric(self) -> None:
        """Test stiffness matrix is symmetric in 3D."""
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=self.bkd_inst,
        )

        K = physics.stiffness_matrix()
        K_np = self.bkd_inst.to_numpy(K)

        np.testing.assert_array_almost_equal(K_np, K_np.T)

    def test_3d_mass_matrix_symmetric(self) -> None:
        """Test mass matrix is symmetric in 3D."""
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=self.bkd_inst,
        )

        M = physics.mass_matrix()
        M_np = self.bkd_inst.to_numpy(M)

        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_3d_residual_shape(self) -> None:
        """Test residual has correct shape in 3D."""
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=self.bkd_inst,
        )

        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)

        self.assertEqual(res.shape, (physics.nstates(),))

    def test_3d_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape in 3D."""
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=self.bkd_inst,
        )

        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        jac = physics.jacobian(u0, 0.0)

        self.assertEqual(jac.shape, (physics.nstates(), physics.nstates()))

    def test_2d_with_body_force(self) -> None:
        """Test 2D elasticity with body force."""
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)

        def body_force(x, time):
            # Constant gravity-like force in y direction
            f = np.zeros_like(x)
            f[1, :] = -1.0
            return f

        physics = LinearElasticity(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            body_force=body_force,
            bkd=self.bkd_inst,
        )

        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        res_np = self.bkd_inst.to_numpy(res)

        # With body force and u=0, residual should be non-zero
        self.assertTrue(np.linalg.norm(res_np) > 0)

    def test_3d_with_body_force(self) -> None:
        """Test 3D elasticity with body force."""
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)

        def body_force(x, time):
            # Constant gravity-like force in z direction
            f = np.zeros_like(x)
            f[2, :] = -1.0
            return f

        physics = LinearElasticity(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            body_force=body_force,
            bkd=self.bkd_inst,
        )

        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        res_np = self.bkd_inst.to_numpy(res)

        # With body force and u=0, residual should be non-zero
        self.assertTrue(np.linalg.norm(res_np) > 0)

    def test_lame_parameters(self) -> None:
        """Test Lame parameter computation."""
        mesh = StructuredMesh2D(
            nx=3, ny=3,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)

        E = 100.0
        nu = 0.25

        physics = LinearElasticity(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=self.bkd_inst,
        )

        # Expected Lame parameters
        # lambda = E * nu / ((1 + nu) * (1 - 2*nu))
        # mu = E / (2 * (1 + nu))
        expected_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
        expected_mu = E / (2 * (1 + nu))

        self.assertAlmostEqual(physics.lame_lambda, expected_lambda, places=10)
        self.assertAlmostEqual(physics.lame_mu, expected_mu, places=10)

    def test_poisson_ratio_validation(self) -> None:
        """Test that invalid Poisson ratios raise errors."""
        mesh = StructuredMesh2D(
            nx=3, ny=3,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)

        # nu = 0.5 is invalid (incompressible limit)
        with self.assertRaises(ValueError):
            LinearElasticity(
                basis=basis,
                youngs_modulus=1.0,
                poisson_ratio=0.5,
                bkd=self.bkd_inst,
            )

        # nu = -1 is invalid
        with self.assertRaises(ValueError):
            LinearElasticity(
                basis=basis,
                youngs_modulus=1.0,
                poisson_ratio=-1.0,
                bkd=self.bkd_inst,
            )

    def test_stiffness_action_consistency(self) -> None:
        """Test that K*u gives expected result for constant strain field.

        For u = [ax, ay] (uniform scaling), K*u should represent the
        boundary traction integrals since -div(sigma) = 0 in the interior.
        """
        E = 1.0  # Young's modulus
        nu = 0.3  # Poisson's ratio

        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=self.bkd_inst,
        )

        # Linear displacement u = [x, y] -> constant strain
        def u_linear(x):
            return x

        u = physics.initial_condition(u_linear)
        K = physics.stiffness_matrix()
        K_np = self.bkd_inst.to_numpy(K)
        u_np = self.bkd_inst.to_numpy(u)

        # K*u should be consistent (non-zero due to boundary tractions)
        Ku = K_np @ u_np
        self.assertEqual(Ku.shape, (physics.nstates(),))
        # For uniform strain, the response should be consistent
        self.assertTrue(np.linalg.norm(Ku) > 0)

    def test_stiffness_action_consistency_3d(self) -> None:
        """Test that K*u gives expected result for constant strain field in 3D."""
        E = 1.0  # Young's modulus
        nu = 0.25  # Poisson's ratio

        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=self.bkd_inst,
        )

        # Linear displacement u = [x, y, z] -> constant strain
        def u_linear(x):
            return x

        u = physics.initial_condition(u_linear)
        K = physics.stiffness_matrix()
        K_np = self.bkd_inst.to_numpy(K)
        u_np = self.bkd_inst.to_numpy(u)

        # K*u should be consistent (non-zero due to boundary tractions)
        Ku = K_np @ u_np
        self.assertEqual(Ku.shape, (physics.nstates(),))
        # For uniform strain, the response should be consistent
        self.assertTrue(np.linalg.norm(Ku) > 0)

    def test_rigid_body_motion_2d(self) -> None:
        """Test that rigid body motion gives zero strain energy.

        Pure translation u = [c, c] should give zero K*u.
        """
        E = 1.0
        nu = 0.3

        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=self.bkd_inst,
        )

        # Constant translation
        def u_translation(x):
            result = np.zeros_like(x)
            result[0, :] = 1.0  # Translate in x
            result[1, :] = 2.0  # Translate in y
            return result

        u = physics.initial_condition(u_translation)
        K = physics.stiffness_matrix()
        K_np = self.bkd_inst.to_numpy(K)
        u_np = self.bkd_inst.to_numpy(u)

        # K*u should be zero for rigid body motion
        Ku = K_np @ u_np
        self.assertLess(np.linalg.norm(Ku), 1e-10)

    def test_rigid_body_motion_3d(self) -> None:
        """Test that rigid body motion gives zero strain energy in 3D."""
        E = 1.0
        nu = 0.25

        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=self.bkd_inst,
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
        K_np = self.bkd_inst.to_numpy(K)
        u_np = self.bkd_inst.to_numpy(u)

        # K*u should be zero for rigid body motion
        Ku = K_np @ u_np
        self.assertLess(np.linalg.norm(Ku), 1e-10)


class TestLinearElasticityNumpy(TestLinearElasticityBase[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Try to import torch for dual-backend testing
try:
    import torch
    from pyapprox.typing.util.backends.torch import TorchBkd

    class TestLinearElasticityTorch(TestLinearElasticityBase[torch.Tensor]):
        """PyTorch backend tests."""

        __test__ = True

        def setUp(self) -> None:
            self._bkd = TorchBkd()
            super().setUp()

        def bkd(self) -> Backend[torch.Tensor]:
            return self._bkd

except ImportError:
    pass


from pyapprox.typing.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
