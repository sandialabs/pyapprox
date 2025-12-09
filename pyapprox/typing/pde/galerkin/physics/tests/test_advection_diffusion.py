"""Tests for LinearAdvectionDiffusionReaction physics."""

import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
    StructuredMesh3D,
)
from pyapprox.typing.pde.galerkin.basis import LagrangeBasis
from pyapprox.typing.pde.galerkin.physics import LinearAdvectionDiffusionReaction
from pyapprox.typing.pde.galerkin.solvers import SteadyStateSolver


class TestLinearADRBase(Generic[Array], unittest.TestCase):
    """Base test class for LinearAdvectionDiffusionReaction."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self.bkd_inst = self.bkd()

    def test_1d_mass_matrix_symmetric(self) -> None:
        """Test mass matrix is symmetric in 1D."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=self.bkd_inst
        )

        M = physics.mass_matrix()
        M_np = self.bkd_inst.to_numpy(M)

        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_1d_mass_matrix_positive_definite(self) -> None:
        """Test mass matrix is positive definite."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=self.bkd_inst
        )

        M = physics.mass_matrix()
        M_np = self.bkd_inst.to_numpy(M)

        eigenvalues = np.linalg.eigvalsh(M_np)
        self.assertTrue(np.all(eigenvalues > 0))

    def test_1d_stiffness_assembly(self) -> None:
        """Test stiffness matrix assembly in 1D."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=1.0, bkd=self.bkd_inst
        )

        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        jac = physics.jacobian(u0, 0.0)
        jac_np = self.bkd_inst.to_numpy(jac)

        # For pure diffusion (no reaction), -jacobian should be the
        # stiffness matrix, which should be symmetric
        K = -jac_np
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)

    def test_1d_residual_shape(self) -> None:
        """Test residual has correct shape."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=self.bkd_inst
        )

        u0 = physics.initial_condition(lambda x: np.sin(np.pi * x[0]))
        res = physics.residual(u0, 0.0)

        self.assertEqual(res.shape, (physics.nstates(),))

    def test_1d_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=self.bkd_inst
        )

        u0 = physics.initial_condition(lambda x: np.sin(np.pi * x[0]))
        jac = physics.jacobian(u0, 0.0)

        self.assertEqual(jac.shape, (physics.nstates(), physics.nstates()))

    def test_2d_physics(self) -> None:
        """Test physics works in 2D."""
        mesh = StructuredMesh2D(
            nx=5, ny=5, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=self.bkd_inst
        )
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=self.bkd_inst
        )

        # Initial condition
        u0 = physics.initial_condition(
            lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        )

        # Check shapes
        self.assertEqual(u0.shape, (physics.nstates(),))

        res = physics.residual(u0, 0.0)
        self.assertEqual(res.shape, (physics.nstates(),))

    def test_with_forcing(self) -> None:
        """Test physics with forcing term."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        def forcing(x):
            return np.ones(x.shape[1])

        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, forcing=forcing, bkd=self.bkd_inst
        )

        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        res_np = self.bkd_inst.to_numpy(res)

        # With forcing and u=0, residual should be non-zero
        self.assertTrue(np.linalg.norm(res_np) > 0)

    def test_3d_physics(self) -> None:
        """Test physics works in 3D."""
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=self.bkd_inst
        )

        # Initial condition
        u0 = physics.initial_condition(
            lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2])
        )

        # Check shapes
        self.assertEqual(u0.shape, (physics.nstates(),))

        res = physics.residual(u0, 0.0)
        self.assertEqual(res.shape, (physics.nstates(),))

        jac = physics.jacobian(u0, 0.0)
        self.assertEqual(jac.shape, (physics.nstates(), physics.nstates()))

    def test_3d_mass_matrix_symmetric(self) -> None:
        """Test mass matrix is symmetric in 3D."""
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=self.bkd_inst
        )

        M = physics.mass_matrix()
        M_np = self.bkd_inst.to_numpy(M)

        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_3d_stiffness_symmetric(self) -> None:
        """Test stiffness matrix is symmetric in 3D (pure diffusion)."""
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = LagrangeBasis(mesh, degree=1)
        physics = LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=1.0, bkd=self.bkd_inst
        )

        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        jac = physics.jacobian(u0, 0.0)
        jac_np = self.bkd_inst.to_numpy(jac)

        # For pure diffusion, -jacobian = stiffness matrix should be symmetric
        K = -jac_np
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)

    def test_3d_steady_state_solve(self) -> None:
        """Test steady-state solve in 3D."""
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = LagrangeBasis(mesh, degree=1)

        # Add reaction term to make problem well-posed
        physics = LinearAdvectionDiffusionReaction(
            basis=basis,
            diffusivity=0.1,
            reaction=1.0,
            forcing=lambda x: np.ones(x.shape[1]),
            bkd=self.bkd_inst,
        )

        solver = SteadyStateSolver(physics, tol=1e-10)
        result = solver.solve_linear()

        self.assertTrue(result.converged)
        self.assertLess(result.residual_norm, 1e-8)

    def test_manufactured_solution_1d(self) -> None:
        """Test convergence using manufactured solution in 1D.

        Use u_exact = cos(pi*x), which satisfies zero Neumann BCs at x=0,1.
        For diffusion-reaction: -D*u'' + r*u = f
        u'' = -pi^2 * cos(pi*x)
        f = D*pi^2*cos(pi*x) + r*cos(pi*x) = (D*pi^2 + r)*cos(pi*x)
        """
        D = 1.0
        r = 1.0

        def u_exact(x):
            return np.cos(np.pi * x[0])

        def forcing(x):
            return (D * np.pi**2 + r) * np.cos(np.pi * x[0])

        errors = []
        mesh_sizes = [10, 20, 40]

        for nx in mesh_sizes:
            mesh = StructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=self.bkd_inst)
            basis = LagrangeBasis(mesh, degree=1)

            physics = LinearAdvectionDiffusionReaction(
                basis=basis,
                diffusivity=D,
                reaction=r,
                forcing=forcing,
                bkd=self.bkd_inst,
            )

            solver = SteadyStateSolver(physics, tol=1e-12)
            result = solver.solve_linear()

            # Compute L2 error at DOF locations
            u_num = self.bkd_inst.to_numpy(result.solution)
            dof_locs = self.bkd_inst.to_numpy(basis.dof_coordinates())
            u_ex = u_exact(dof_locs)
            error = np.sqrt(np.mean((u_num - u_ex)**2))
            errors.append(error)

        # Check convergence rate (should be ~2 for P1 elements)
        errors = np.array(errors)
        rates = np.log(errors[:-1] / errors[1:]) / np.log(2)

        # P1 elements should have convergence rate ~2
        self.assertTrue(np.all(rates > 1.5), f"Rates: {rates}")

    def test_manufactured_solution_2d(self) -> None:
        """Test convergence using manufactured solution in 2D.

        Use u_exact = cos(pi*x)*cos(pi*y), which satisfies zero Neumann BCs.
        Laplacian: -2*pi^2*cos(pi*x)*cos(pi*y)
        f = (2*D*pi^2 + r)*cos(pi*x)*cos(pi*y)
        """
        D = 1.0
        r = 1.0

        def u_exact(x):
            return np.cos(np.pi * x[0]) * np.cos(np.pi * x[1])

        def forcing(x):
            return (2 * D * np.pi**2 + r) * np.cos(np.pi * x[0]) * np.cos(np.pi * x[1])

        errors = []
        mesh_sizes = [5, 10, 20]

        for n in mesh_sizes:
            mesh = StructuredMesh2D(
                nx=n, ny=n,
                bounds=[[0.0, 1.0], [0.0, 1.0]],
                bkd=self.bkd_inst,
            )
            basis = LagrangeBasis(mesh, degree=1)

            physics = LinearAdvectionDiffusionReaction(
                basis=basis,
                diffusivity=D,
                reaction=r,
                forcing=forcing,
                bkd=self.bkd_inst,
            )

            solver = SteadyStateSolver(physics, tol=1e-12)
            result = solver.solve_linear()

            # Compute L2 error at DOF locations
            u_num = self.bkd_inst.to_numpy(result.solution)
            dof_locs = self.bkd_inst.to_numpy(basis.dof_coordinates())
            u_ex = u_exact(dof_locs)
            error = np.sqrt(np.mean((u_num - u_ex)**2))
            errors.append(error)

        # Check convergence rate
        errors = np.array(errors)
        rates = np.log(errors[:-1] / errors[1:]) / np.log(2)

        # P1 elements should have convergence rate ~2
        self.assertTrue(np.all(rates > 1.5), f"Rates: {rates}")

    def test_manufactured_solution_3d(self) -> None:
        """Test convergence using manufactured solution in 3D.

        Use u_exact = cos(pi*x)*cos(pi*y)*cos(pi*z), which satisfies zero Neumann BCs.
        Laplacian: -3*pi^2*cos(pi*x)*cos(pi*y)*cos(pi*z)
        f = (3*D*pi^2 + r)*cos(pi*x)*cos(pi*y)*cos(pi*z)
        """
        D = 1.0
        r = 1.0

        def u_exact(x):
            return np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]) * np.cos(np.pi * x[2])

        def forcing(x):
            return (
                (3 * D * np.pi**2 + r)
                * np.cos(np.pi * x[0])
                * np.cos(np.pi * x[1])
                * np.cos(np.pi * x[2])
            )

        errors = []
        mesh_sizes = [3, 5, 8]

        for n in mesh_sizes:
            mesh = StructuredMesh3D(
                nx=n, ny=n, nz=n,
                bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                bkd=self.bkd_inst,
            )
            basis = LagrangeBasis(mesh, degree=1)

            physics = LinearAdvectionDiffusionReaction(
                basis=basis,
                diffusivity=D,
                reaction=r,
                forcing=forcing,
                bkd=self.bkd_inst,
            )

            solver = SteadyStateSolver(physics, tol=1e-12)
            result = solver.solve_linear()

            # Compute L2 error at DOF locations
            u_num = self.bkd_inst.to_numpy(result.solution)
            dof_locs = self.bkd_inst.to_numpy(basis.dof_coordinates())
            u_ex = u_exact(dof_locs)
            error = np.sqrt(np.mean((u_num - u_ex)**2))
            errors.append(error)

        # Check convergence rate
        errors = np.array(errors)
        rates = np.log(errors[:-1] / errors[1:]) / np.log(2)

        # P1 elements should have convergence rate ~2
        # Use lower threshold for 3D due to coarser meshes
        self.assertTrue(np.all(rates > 1.4), f"Rates: {rates}")


class TestLinearADRNumpy(TestLinearADRBase[NDArray[Any]]):
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

    class TestLinearADRTorch(TestLinearADRBase[torch.Tensor]):
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
