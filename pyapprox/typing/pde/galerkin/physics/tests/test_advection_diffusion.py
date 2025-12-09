"""Tests for LinearAdvectionDiffusionReaction physics."""

import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.typing.pde.galerkin.basis import LagrangeBasis
from pyapprox.typing.pde.galerkin.physics import LinearAdvectionDiffusionReaction


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
