"""Tests for LinearElasticity physics."""

import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
    StructuredMesh3D,
)
from pyapprox.typing.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.typing.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity,
)
from pyapprox.typing.pde.galerkin.solvers import SteadyStateSolver

def _to_dense(mat, bkd):
    """Convert a matrix (possibly sparse) to a dense numpy array."""
    if issparse(mat):
        return mat.toarray()
    return bkd.to_numpy(mat)


# Alias for backward compatibility in tests
LinearElasticity = CompositeLinearElasticity


class TestLinearElasticityBase(Generic[Array], unittest.TestCase):
    """Base test class for LinearElasticity."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self.bkd_inst = self.bkd()

    def test_1d_stiffness_symmetric(self) -> None:
        """Test stiffness matrix is symmetric in 1D."""
        mesh = StructuredMesh1D(
            nx=5, bounds=(0.0, 1.0), bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=self.bkd_inst,
        )
        K = physics.stiffness_matrix()
        K_np = _to_dense(K, self.bkd_inst)
        np.testing.assert_array_almost_equal(K_np, K_np.T)

    def test_1d_residual_shape(self) -> None:
        """Test residual has correct shape in 1D."""
        mesh = StructuredMesh1D(
            nx=5, bounds=(0.0, 1.0), bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=self.bkd_inst,
        )
        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        self.assertEqual(res.shape, (physics.nstates(),))

    def test_1d_rigid_body_motion(self) -> None:
        """Test that constant displacement gives zero stiffness action in 1D."""
        mesh = StructuredMesh1D(
            nx=5, bounds=(0.0, 1.0), bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=self.bkd_inst,
        )
        u = self.bkd_inst.asarray(np.ones(physics.nstates()))
        K = physics.stiffness_matrix()
        Ku = _to_dense(K, self.bkd_inst) @ self.bkd_inst.to_numpy(u)
        self.assertLess(np.linalg.norm(Ku), 1e-10)

    def test_1d_manufactured_solution(self) -> None:
        """1D solve recovers manufactured solution with non-zero BCs.

        Uses create_elasticity_manufactured_test for MMS forcing.
        u(x) = 0.1*x*(1-x) + 0.2 (degree-2, non-zero at boundaries).
        With degree-2 elements the FEM solution is exact.

        Exercises VectorLagrangeBasis.get_dofs() in 1D — regression test
        for a bug where 1D DOFs were returned empty.
        """
        from skfem.models.elasticity import lame_parameters as _lame
        from pyapprox.typing.pde.galerkin.boundary.implementations import (
            DirichletBC,
        )
        from pyapprox.typing.pde.galerkin.manufactured.adapter import (
            create_elasticity_manufactured_test,
        )

        E, nu = 1.0, 0.3
        lam, mu = _lame(E, nu)

        functions, nvars = create_elasticity_manufactured_test(
            bounds=[0.0, 1.0],
            sol_strs=["0.1*x*(1-x) + 0.2"],
            lambda_str=str(lam),
            mu_str=str(mu),
            bkd=self.bkd_inst,
        )

        mesh = StructuredMesh1D(
            nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=2)

        sol_func = functions["solution"]
        forcing_func = functions["forcing"]

        def body_force(x, time):
            vals = forcing_func(x)  # (npts, 1)
            return vals.T  # (1, npts) = (ndim, npts)

        def dirichlet_value(coords, time=0.0):
            vals = sol_func(coords)  # (nbndry_dofs, 1)
            nbndry_dofs = coords.shape[1]
            result = np.zeros(nbndry_dofs)
            for j in range(nbndry_dofs):
                result[j] = vals[j, j % nvars]
            return result

        bc_list = [
            DirichletBC(basis, "left", dirichlet_value, self.bkd_inst),
            DirichletBC(basis, "right", dirichlet_value, self.bkd_inst),
        ]
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            body_force=body_force,
            boundary_conditions=bc_list,
            bkd=self.bkd_inst,
        )

        solver = SteadyStateSolver(
            physics, tol=1e-12, max_iter=5, line_search=False
        )
        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        result = solver.solve(u0)

        self.assertTrue(
            result.converged,
            f"1D did not converge: {result.residual_norm}",
        )

        u_np = self.bkd_inst.to_numpy(result.solution)
        dof_coords = self.bkd_inst.to_numpy(basis.dof_coordinates())
        sol_vals = functions["solution"](dof_coords)  # (ndofs, 1)
        n_dofs = basis.ndofs()
        exact = np.zeros(n_dofs)
        for i in range(n_dofs):
            exact[i] = sol_vals[i, i % nvars]
        rel_error = np.linalg.norm(u_np - exact) / np.linalg.norm(exact)
        self.assertLess(
            rel_error, 1e-8,
            f"1D manufactured solution rel error: {rel_error:.2e}",
        )

    def test_2d_stiffness_symmetric(self) -> None:
        """Test stiffness matrix is symmetric in 2D."""
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=self.bkd_inst,
        )

        K = physics.stiffness_matrix()
        K_np = _to_dense(K, self.bkd_inst)

        np.testing.assert_array_almost_equal(K_np, K_np.T)

    def test_2d_mass_matrix_symmetric(self) -> None:
        """Test mass matrix is symmetric in 2D."""
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=self.bkd_inst,
        )

        M = physics.mass_matrix()
        M_np = _to_dense(M, self.bkd_inst)

        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_2d_residual_shape(self) -> None:
        """Test residual has correct shape in 2D."""
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
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
        physics = LinearElasticity.from_uniform(
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
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=self.bkd_inst,
        )

        K = physics.stiffness_matrix()
        K_np = _to_dense(K, self.bkd_inst)

        np.testing.assert_array_almost_equal(K_np, K_np.T)

    def test_3d_mass_matrix_symmetric(self) -> None:
        """Test mass matrix is symmetric in 3D."""
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=self.bkd_inst,
        )

        M = physics.mass_matrix()
        M_np = _to_dense(M, self.bkd_inst)

        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_3d_residual_shape(self) -> None:
        """Test residual has correct shape in 3D."""
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
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
        physics = LinearElasticity.from_uniform(
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

        physics = LinearElasticity.from_uniform(
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

        physics = LinearElasticity.from_uniform(
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

        physics = LinearElasticity.from_uniform(
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
            LinearElasticity.from_uniform(
                basis=basis,
                youngs_modulus=1.0,
                poisson_ratio=0.5,
                bkd=self.bkd_inst,
            )

        # nu = -1 is invalid
        with self.assertRaises(ValueError):
            LinearElasticity.from_uniform(
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
        physics = LinearElasticity.from_uniform(
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
        K_np = _to_dense(K, self.bkd_inst)
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
        physics = LinearElasticity.from_uniform(
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
        K_np = _to_dense(K, self.bkd_inst)
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
        physics = LinearElasticity.from_uniform(
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
        K_np = _to_dense(K, self.bkd_inst)
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
        physics = LinearElasticity.from_uniform(
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
        K_np = _to_dense(K, self.bkd_inst)
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

        @unittest.skip(
            "sparse solve not available on CPU with TorchBkd"
        )
        def test_1d_manufactured_solution(self) -> None:
            pass

except ImportError:
    pass


from pyapprox.typing.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
