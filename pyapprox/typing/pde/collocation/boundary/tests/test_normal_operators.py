"""Tests for normal operators."""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.pde.collocation.basis import (
    ChebyshevBasis1D,
    ChebyshevBasis2D,
)
from pyapprox.typing.pde.collocation.mesh import (
    TransformedMesh1D,
    TransformedMesh2D,
    create_uniform_mesh_1d,
)
from pyapprox.typing.pde.collocation.boundary.normal_operators import (
    GradientNormalOperator,
    FluxNormalOperator,
    TractionNormalOperator,
    _LegacyNormalOperator,
)
from pyapprox.typing.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)


class TestGradientNormalOperator(Generic[Array], unittest.TestCase):
    """Base test class for GradientNormalOperator."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_1d_left_boundary(self):
        """Test grad(u).n at left boundary for u = x^2."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        left_idx = umesh.boundary_indices(0)
        # Left boundary normal in 1D: [-1]
        normals = bkd.array([[-1.0]])
        D = basis.derivative_matrix()

        normal_op = GradientNormalOperator(bkd, left_idx, normals, [D])

        # u = x^2, du/dx = 2x, at x=-1: du/dx = -2, grad(u).n = (-2)*(-1) = 2
        nodes = basis.nodes()
        state = nodes * nodes
        result = normal_op(state)
        bkd.assert_allclose(result, bkd.array([2.0]), atol=1e-10)

    def test_1d_right_boundary(self):
        """Test grad(u).n at right boundary for u = x^2."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        right_idx = umesh.boundary_indices(1)
        # Right boundary normal in 1D: [+1]
        normals = bkd.array([[1.0]])
        D = basis.derivative_matrix()

        normal_op = GradientNormalOperator(bkd, right_idx, normals, [D])

        # u = x^2, du/dx = 2x, at x=+1: du/dx = 2, grad(u).n = 2*1 = 2
        nodes = basis.nodes()
        state = nodes * nodes
        result = normal_op(state)
        bkd.assert_allclose(result, bkd.array([2.0]), atol=1e-10)

    def test_jacobian_is_constant(self):
        """Test that gradient normal operator Jacobian is state-independent."""
        bkd = self.bkd()
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        left_idx = umesh.boundary_indices(0)
        normals = bkd.array([[-1.0]])
        D = basis.derivative_matrix()
        normal_op = GradientNormalOperator(bkd, left_idx, normals, [D])

        state1 = bkd.ones((npts,))
        state2 = bkd.ones((npts,)) * 5.0
        jac1 = normal_op.jacobian(state1)
        jac2 = normal_op.jacobian(state2)
        bkd.assert_allclose(jac1, jac2, atol=1e-14)

    def test_jacobian_consistency(self):
        """Test that jacobian @ state matches __call__(state)."""
        bkd = self.bkd()
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        left_idx = umesh.boundary_indices(0)
        normals = bkd.array([[-1.0]])
        D = basis.derivative_matrix()
        normal_op = GradientNormalOperator(bkd, left_idx, normals, [D])

        np.random.seed(42)
        state = bkd.asarray(np.random.randn(npts))
        result = normal_op(state)
        jac = normal_op.jacobian(state)
        result_from_jac = jac @ state
        bkd.assert_allclose(result, result_from_jac, atol=1e-12)

    def test_2d_constant_normals(self):
        """Test 2D gradient normal operator with constant normals."""
        bkd = self.bkd()
        npts_x, npts_y = 8, 8
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)

        # Left boundary (x = -1), normal = [-1, 0]
        left_idx = mesh.boundary_indices(0)
        nboundary = left_idx.shape[0]
        normals = bkd.zeros((nboundary, 2))
        normals = bkd.copy(normals)
        for i in range(nboundary):
            normals[i, 0] = -1.0

        Dx = basis.derivative_matrix(1, 0)
        Dy = basis.derivative_matrix(1, 1)
        normal_op = GradientNormalOperator(
            bkd, left_idx, normals, [Dx, Dy]
        )

        # u = x*y, grad = (y, x), grad.n at x=-1 = y*(-1) + x*0 = -y
        physical_pts = mesh.points()
        x_phys = physical_pts[0, :]
        y_phys = physical_pts[1, :]
        state = x_phys * y_phys

        result = normal_op(state)
        # Expected: -y at left boundary points
        expected = -y_phys[left_idx]
        bkd.assert_allclose(result, expected, atol=1e-10)


class TestFluxNormalOperator(Generic[Array], unittest.TestCase):
    """Base test class for FluxNormalOperator."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_flux_with_adr_physics_1d(self):
        """Test FluxNormalOperator with ADR physics as flux provider."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        diffusion = 2.0
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=diffusion)

        left_idx = umesh.boundary_indices(0)
        normals = bkd.array([[-1.0]])

        flux_op = FluxNormalOperator(bkd, left_idx, normals, physics)

        # u = x^2, flux = -D*du/dx = -2*2x = -4x
        # At x=-1: flux = 4, flux.n = 4*(-1) = -4
        nodes = basis.nodes()
        state = nodes * nodes
        result = flux_op(state)
        bkd.assert_allclose(result, bkd.array([-4.0]), atol=1e-10)

    def test_flux_with_velocity(self):
        """Test FluxNormalOperator with ADR physics including velocity."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        diffusion = 1.0
        velocity = [bkd.full((npts,), 3.0)]  # v = 3
        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=diffusion, velocity=velocity
        )

        right_idx = umesh.boundary_indices(1)
        normals = bkd.array([[1.0]])

        flux_op = FluxNormalOperator(bkd, right_idx, normals, physics)

        # u = x, du/dx = 1
        # flux = -D*du/dx + v*u = -1 + 3*x
        # At x=1: flux = -1 + 3 = 2, flux.n = 2*(+1) = 2
        nodes = basis.nodes()
        state = bkd.copy(nodes)
        result = flux_op(state)
        bkd.assert_allclose(result, bkd.array([2.0]), atol=1e-10)

    def test_flux_jacobian_consistency(self):
        """Test flux operator Jacobian via finite differences."""
        bkd = self.bkd()
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        left_idx = umesh.boundary_indices(0)
        normals = bkd.array([[-1.0]])
        flux_op = FluxNormalOperator(bkd, left_idx, normals, physics)

        np.random.seed(42)
        state = bkd.asarray(np.random.randn(npts))
        jac = flux_op.jacobian(state)

        # Finite difference check
        eps = 1e-7
        f0 = flux_op(state)
        for j in range(npts):
            state_pert = bkd.copy(state)
            state_pert[j] = state_pert[j] + eps
            f1 = flux_op(state_pert)
            fd_col = (f1 - f0) / eps
            bkd.assert_allclose(
                bkd.reshape(jac[:, j], fd_col.shape), fd_col, atol=1e-5
            )


class TestLegacyNormalOperator(Generic[Array], unittest.TestCase):
    """Base test class for _LegacyNormalOperator."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_matches_old_behavior(self):
        """Test legacy operator matches old NeumannBC computation."""
        bkd = self.bkd()
        npts = 5
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        D = basis.derivative_matrix()
        D_bndry = bkd.reshape(D[0, :], (1, npts))
        normal_sign = -1.0

        legacy_op = _LegacyNormalOperator(bkd, D_bndry, normal_sign)

        state = bkd.ones((npts,))
        result = legacy_op(state)
        expected = normal_sign * (D_bndry @ state)
        bkd.assert_allclose(result, expected, atol=1e-14)

        jac = legacy_op.jacobian(state)
        expected_jac = normal_sign * D_bndry
        bkd.assert_allclose(jac, expected_jac, atol=1e-14)


class TestTractionNormalOperator(Generic[Array], unittest.TestCase):
    """Base test class for TractionNormalOperator."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_traction_at_boundary(self):
        """Test traction at left boundary for known polynomial displacement."""
        bkd = self.bkd()
        npts_x, npts_y = 8, 8
        npts = npts_x * npts_y
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)

        Dx = basis.derivative_matrix(1, 0)
        Dy = basis.derivative_matrix(1, 1)

        lamda, mu = 1.0, 1.0

        # Left boundary (x = -1), normal = [-1, 0]
        left_idx = mesh.boundary_indices(0)
        nboundary = left_idx.shape[0]
        normals = bkd.zeros((nboundary, 2))
        normals = bkd.copy(normals)
        for i in range(nboundary):
            normals[i, 0] = -1.0

        traction_op_x = TractionNormalOperator(
            bkd, left_idx, normals, [Dx, Dy], lamda, mu, 0, npts
        )
        traction_op_y = TractionNormalOperator(
            bkd, left_idx, normals, [Dx, Dy], lamda, mu, 1, npts
        )

        # u_x = x^2, u_y = x*y
        pts = mesh.points()
        x_phys = pts[0, :]
        y_phys = pts[1, :]
        u = x_phys * x_phys
        v = x_phys * y_phys
        state = bkd.concatenate([u, v])

        # Strain: exx = 2x, exy = 0.5*(0 + y) = 0.5*y, eyy = x
        # At left (x=-1): exx=-2, exy=0.5*y, eyy=-1
        # σ_xx = λ*(exx+eyy) + 2μ*exx = 1*(-3) + 2*(-2) = -7
        # σ_xy = 2μ*exy = 2*0.5*y = y
        # t_x = σ_xx*nx + σ_xy*ny = (-7)*(-1) + y*0 = 7
        # t_y = σ_xy*nx + σ_yy*ny = y*(-1) + 0 = -y
        result_x = traction_op_x(state)
        result_y = traction_op_y(state)

        expected_x = bkd.full((nboundary,), 7.0)
        expected_y = -y_phys[left_idx]
        bkd.assert_allclose(result_x, expected_x, atol=1e-10)
        bkd.assert_allclose(result_y, expected_y, atol=1e-10)

    def test_jacobian_consistency(self):
        """Test traction operator Jacobian via finite differences."""
        bkd = self.bkd()
        npts_x, npts_y = 6, 6
        npts = npts_x * npts_y
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)

        Dx = basis.derivative_matrix(1, 0)
        Dy = basis.derivative_matrix(1, 1)

        left_idx = mesh.boundary_indices(0)
        nboundary = left_idx.shape[0]
        normals = bkd.zeros((nboundary, 2))
        normals = bkd.copy(normals)
        for i in range(nboundary):
            normals[i, 0] = -1.0

        for component in [0, 1]:
            traction_op = TractionNormalOperator(
                bkd, left_idx, normals, [Dx, Dy], 2.0, 0.5, component, npts
            )

            np.random.seed(42)
            state = bkd.asarray(np.random.randn(2 * npts))
            jac = traction_op.jacobian(state)

            # Finite difference check
            eps = 1e-7
            f0 = traction_op(state)
            for j in range(2 * npts):
                state_pert = bkd.copy(state)
                state_pert[j] = state_pert[j] + eps
                f1 = traction_op(state_pert)
                fd_col = (f1 - f0) / eps
                bkd.assert_allclose(
                    bkd.reshape(jac[:, j], fd_col.shape), fd_col, atol=1e-5
                )

    def test_jacobian_is_constant(self):
        """Test that traction Jacobian is state-independent."""
        bkd = self.bkd()
        npts_x, npts_y = 6, 6
        npts = npts_x * npts_y
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)

        Dx = basis.derivative_matrix(1, 0)
        Dy = basis.derivative_matrix(1, 1)

        left_idx = mesh.boundary_indices(0)
        nboundary = left_idx.shape[0]
        normals = bkd.zeros((nboundary, 2))
        normals = bkd.copy(normals)
        for i in range(nboundary):
            normals[i, 0] = -1.0

        traction_op = TractionNormalOperator(
            bkd, left_idx, normals, [Dx, Dy], 1.0, 1.0, 0, npts
        )

        state1 = bkd.ones((2 * npts,))
        state2 = bkd.ones((2 * npts,)) * 5.0
        jac1 = traction_op.jacobian(state1)
        jac2 = traction_op.jacobian(state2)
        bkd.assert_allclose(jac1, jac2, atol=1e-14)


# NumPy backend
class TestGradientNormalOperatorNumpy(TestGradientNormalOperator[NDArray[Any]]):
    __test__ = True
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFluxNormalOperatorNumpy(TestFluxNormalOperator[NDArray[Any]]):
    __test__ = True
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTractionNormalOperatorNumpy(TestTractionNormalOperator[NDArray[Any]]):
    __test__ = True
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLegacyNormalOperatorNumpy(TestLegacyNormalOperator[NDArray[Any]]):
    __test__ = True
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# Torch backend
class TestGradientNormalOperatorTorch(TestGradientNormalOperator[torch.Tensor]):
    __test__ = True
    def bkd(self) -> TorchBkd:
        return TorchBkd()
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


class TestFluxNormalOperatorTorch(TestFluxNormalOperator[torch.Tensor]):
    __test__ = True
    def bkd(self) -> TorchBkd:
        return TorchBkd()
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


class TestTractionNormalOperatorTorch(TestTractionNormalOperator[torch.Tensor]):
    __test__ = True
    def bkd(self) -> TorchBkd:
        return TorchBkd()
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


class TestLegacyNormalOperatorTorch(TestLegacyNormalOperator[torch.Tensor]):
    __test__ = True
    def bkd(self) -> TorchBkd:
        return TorchBkd()
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


if __name__ == "__main__":
    unittest.main()
