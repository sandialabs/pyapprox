"""Tests for Helmholtz physics."""

import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.typing.pde.galerkin.basis import LagrangeBasis
from pyapprox.typing.pde.galerkin.physics import Helmholtz
from pyapprox.typing.pde.galerkin.solvers import SteadyStateSolver


class TestHelmholtzBase(Generic[Array], unittest.TestCase):
    """Base test class for Helmholtz physics."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self.bkd_inst = self.bkd()

    def test_1d_mass_matrix_symmetric(self) -> None:
        """Test mass matrix is symmetric in 1D."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = Helmholtz(
            basis=basis, wavenumber=2*np.pi, bkd=self.bkd_inst
        )

        M = physics.mass_matrix()
        M_np = self.bkd_inst.to_numpy(M)

        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_1d_stiffness_symmetric(self) -> None:
        """Test stiffness matrix is symmetric in 1D."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = Helmholtz(
            basis=basis, wavenumber=2*np.pi, bkd=self.bkd_inst
        )

        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        jac = physics.jacobian(u0, 0.0)
        jac_np = self.bkd_inst.to_numpy(jac)

        # For Helmholtz, -jacobian = K = K_laplacian + k^2*M, should be symmetric
        K = -jac_np
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)

    def test_1d_residual_shape(self) -> None:
        """Test residual has correct shape."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = Helmholtz(
            basis=basis, wavenumber=2*np.pi, bkd=self.bkd_inst
        )

        u0 = physics.initial_condition(lambda x: np.sin(np.pi * x[0]))
        res = physics.residual(u0, 0.0)

        self.assertEqual(res.shape, (physics.nstates(),))

    def test_1d_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = Helmholtz(
            basis=basis, wavenumber=2*np.pi, bkd=self.bkd_inst
        )

        u0 = physics.initial_condition(lambda x: np.sin(np.pi * x[0]))
        jac = physics.jacobian(u0, 0.0)

        self.assertEqual(jac.shape, (physics.nstates(), physics.nstates()))

    def test_2d_physics(self) -> None:
        """Test Helmholtz works in 2D."""
        mesh = StructuredMesh2D(
            nx=5, ny=5, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=self.bkd_inst
        )
        basis = LagrangeBasis(mesh, degree=1)
        physics = Helmholtz(
            basis=basis, wavenumber=np.pi, bkd=self.bkd_inst
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
        """Test Helmholtz with forcing term."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        def forcing(x):
            return np.ones(x.shape[1])

        physics = Helmholtz(
            basis=basis, wavenumber=2*np.pi, forcing=forcing, bkd=self.bkd_inst
        )

        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        res_np = self.bkd_inst.to_numpy(res)

        # With forcing and u=0, residual should be non-zero
        self.assertTrue(np.linalg.norm(res_np) > 0)

    def test_wavenumber_property(self) -> None:
        """Test wavenumber property."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        k = 3.5
        physics = Helmholtz(basis=basis, wavenumber=k, bkd=self.bkd_inst)

        self.assertEqual(physics.wavenumber, k)

    def test_steady_state_solve(self) -> None:
        """Test solving steady-state Helmholtz with forcing."""
        mesh = StructuredMesh1D(nx=20, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        # Point source in the middle
        def forcing(x):
            return np.exp(-100 * (x[0] - 0.5)**2)

        physics = Helmholtz(
            basis=basis,
            wavenumber=2*np.pi,  # wavelength = 1
            forcing=forcing,
            bkd=self.bkd_inst,
        )

        solver = SteadyStateSolver(physics, tol=1e-10)
        result = solver.solve_linear()

        self.assertTrue(result.converged)
        self.assertLess(result.residual_norm, 1e-8)


class TestHelmholtzNumpy(TestHelmholtzBase[NDArray[Any]]):
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

    class TestHelmholtzTorch(TestHelmholtzBase[torch.Tensor]):
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
