"""Tests for physics implementations."""

import unittest
from typing import Generic
import math

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.typing.pde.collocation.mesh import (
    create_uniform_mesh_1d,
    TransformedMesh1D,
)
from pyapprox.typing.pde.collocation.boundary import (
    constant_dirichlet_bc,
    zero_dirichlet_bc,
    DirichletBC,
)
from pyapprox.typing.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
    AdvectionDiffusionReactionWithParam,
    create_steady_diffusion,
    create_advection_diffusion,
)
from pyapprox.typing.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.typing.pde.collocation.manufactured_solutions import (
    ManufacturedAdvectionDiffusionReaction,
)


class TestAdvectionDiffusionReaction(Generic[Array], unittest.TestCase):
    """Base test class for ADR physics."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_diffusion_residual_polynomial(self):
        """Test diffusion residual with polynomial solution.

        For u(x) = x^2 on [-1, 1]:
        laplacian(u) = 2
        With D = 1: residual = D * laplacian(u) = 2
        """
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        # u = x^2
        nodes = basis.nodes()
        u = nodes ** 2

        residual = physics.residual(u, time=0.0)

        # Expected: laplacian(x^2) = 2 (constant)
        expected = bkd.full((npts,), 2.0)
        bkd.assert_allclose(residual, expected, atol=1e-10)

    def test_diffusion_jacobian(self):
        """Test diffusion Jacobian via finite differences."""
        bkd = self.bkd()
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        # Random state
        u = bkd.sin(math.pi * basis.nodes())

        # Compute analytical Jacobian
        jac = physics.jacobian(u, time=0.0)

        # Verify via finite differences
        eps = 1e-7
        jac_fd = bkd.zeros((npts, npts))
        for j in range(npts):
            u_plus = bkd.copy(u)
            u_plus[j] = u_plus[j] + eps
            u_minus = bkd.copy(u)
            u_minus[j] = u_minus[j] - eps
            res_plus = physics.residual(u_plus, 0.0)
            res_minus = physics.residual(u_minus, 0.0)
            for i in range(npts):
                jac_fd[i, j] = (res_plus[i] - res_minus[i]) / (2 * eps)

        bkd.assert_allclose(jac, jac_fd, atol=1e-6)

    def test_advection_residual(self):
        """Test advection residual.

        For u(x) = sin(pi*x), v = 1:
        -v * du/dx = -cos(pi*x) * pi
        """
        bkd = self.bkd()
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        # Velocity = 1
        velocity = [bkd.ones((npts,))]
        physics = AdvectionDiffusionReaction(basis, bkd, velocity=velocity)

        nodes = basis.nodes()
        u = bkd.sin(math.pi * nodes)

        residual = physics.residual(u, time=0.0)

        # Expected: -1 * d/dx[sin(pi*x)] = -pi * cos(pi*x)
        expected = -math.pi * bkd.cos(math.pi * nodes)
        bkd.assert_allclose(residual, expected, atol=1e-8)

    def test_advection_jacobian(self):
        """Test advection Jacobian via finite differences."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        velocity = [bkd.full((npts,), 2.0)]  # v = 2
        physics = AdvectionDiffusionReaction(basis, bkd, velocity=velocity)

        u = bkd.cos(math.pi * basis.nodes())

        jac = physics.jacobian(u, time=0.0)

        # Finite differences
        eps = 1e-7
        jac_fd = bkd.zeros((npts, npts))
        for j in range(npts):
            u_plus = bkd.copy(u)
            u_plus[j] = u_plus[j] + eps
            u_minus = bkd.copy(u)
            u_minus[j] = u_minus[j] - eps
            res_plus = physics.residual(u_plus, 0.0)
            res_minus = physics.residual(u_minus, 0.0)
            for i in range(npts):
                jac_fd[i, j] = (res_plus[i] - res_minus[i]) / (2 * eps)

        bkd.assert_allclose(jac, jac_fd, atol=1e-6)

    def test_reaction_residual(self):
        """Test reaction residual.

        For u and r = 2: residual = 2 * u
        """
        bkd = self.bkd()
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, reaction=2.0)

        u = bkd.linspace(1.0, 5.0, npts)

        residual = physics.residual(u, time=0.0)

        expected = 2.0 * u
        bkd.assert_allclose(residual, expected, atol=1e-14)

    def test_forcing_residual(self):
        """Test forcing term."""
        bkd = self.bkd()
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        forcing = bkd.ones((npts,)) * 3.0
        physics = AdvectionDiffusionReaction(
            basis, bkd, forcing=lambda t: forcing
        )

        u = bkd.zeros((npts,))

        residual = physics.residual(u, time=0.0)

        # With no other terms, residual = forcing
        bkd.assert_allclose(residual, forcing, atol=1e-14)

    def test_combined_adr(self):
        """Test combined ADR physics."""
        bkd = self.bkd()
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        D = 0.1
        v = bkd.full((npts,), 1.0)
        r = -0.5

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=D, velocity=[v], reaction=r
        )

        u = bkd.sin(math.pi * basis.nodes())

        jac = physics.jacobian(u, time=0.0)

        # Verify Jacobian via finite differences
        eps = 1e-7
        jac_fd = bkd.zeros((npts, npts))
        for j in range(npts):
            u_plus = bkd.copy(u)
            u_plus[j] = u_plus[j] + eps
            u_minus = bkd.copy(u)
            u_minus[j] = u_minus[j] - eps
            res_plus = physics.residual(u_plus, 0.0)
            res_minus = physics.residual(u_minus, 0.0)
            for i in range(npts):
                jac_fd[i, j] = (res_plus[i] - res_minus[i]) / (2 * eps)

        bkd.assert_allclose(jac, jac_fd, atol=1e-5)

    def test_boundary_condition_application(self):
        """Test applying boundary conditions."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)

        # Set Dirichlet BCs: u(-1) = 0, u(1) = 1
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = constant_dirichlet_bc(bkd, right_idx, 1.0)
        physics.set_boundary_conditions([bc_left, bc_right])

        # Some state
        u = bkd.linspace(0.0, 1.0, npts)

        residual = physics.residual(u, time=0.0)
        jacobian = physics.jacobian(u, time=0.0)

        # Apply BCs
        residual, jacobian = physics.apply_boundary_conditions(
            residual, jacobian, u, time=0.0
        )

        # Check boundary rows of Jacobian are identity
        bkd.assert_allclose(jacobian[0, :], bkd.eye(npts)[0, :], atol=1e-14)
        bkd.assert_allclose(jacobian[-1, :], bkd.eye(npts)[-1, :], atol=1e-14)


class TestAdvectionDiffusionReactionWithParam(Generic[Array], unittest.TestCase):
    """Base test class for parameterized ADR physics."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_param_jacobian(self):
        """Test parameter Jacobian via finite differences."""
        bkd = self.bkd()
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        # Parameterized diffusion: D = 1 + p0 * phi0 + p1 * phi1
        # where phi0 = 1 (constant), phi1 = x
        nodes = basis.nodes()
        phi0 = bkd.ones((npts,))
        phi1 = nodes

        physics = AdvectionDiffusionReactionWithParam(
            basis, bkd,
            diffusion_base=1.0,
            diffusion_basis_funs=[phi0, phi1],
        )

        # Set parameters
        param = bkd.array([0.5, 0.2])
        physics.set_param(param)

        # State
        u = bkd.sin(math.pi * nodes)

        # Analytical parameter Jacobian
        param_jac = physics.param_jacobian(u, time=0.0)

        # Verify via finite differences
        eps = 1e-7
        param_jac_fd = bkd.zeros((npts, 2))
        for j in range(2):
            param_plus = bkd.copy(param)
            param_plus[j] = param_plus[j] + eps
            param_minus = bkd.copy(param)
            param_minus[j] = param_minus[j] - eps

            physics.set_param(param_plus)
            res_plus = physics.residual(u, 0.0)
            physics.set_param(param_minus)
            res_minus = physics.residual(u, 0.0)

            for i in range(npts):
                param_jac_fd[i, j] = (res_plus[i] - res_minus[i]) / (2 * eps)

        # Reset to original
        physics.set_param(param)

        bkd.assert_allclose(param_jac, param_jac_fd, atol=1e-5)

    def test_nparams(self):
        """Test nparams method."""
        bkd = self.bkd()
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        phi0 = bkd.ones((npts,))
        phi1 = basis.nodes()
        phi2 = basis.nodes() ** 2

        physics = AdvectionDiffusionReactionWithParam(
            basis, bkd,
            diffusion_base=1.0,
            diffusion_basis_funs=[phi0, phi1, phi2],
        )

        self.assertEqual(physics.nparams(), 3)

    def test_initial_param_jacobian_zero(self):
        """Test initial param Jacobian is zero when no IC function."""
        bkd = self.bkd()
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        phi0 = bkd.ones((npts,))

        physics = AdvectionDiffusionReactionWithParam(
            basis, bkd,
            diffusion_base=1.0,
            diffusion_basis_funs=[phi0],
        )

        ic_jac = physics.initial_param_jacobian()

        expected = bkd.zeros((npts, 1))
        bkd.assert_allclose(ic_jac, expected, atol=1e-14)


class TestFactoryFunctions(Generic[Array], unittest.TestCase):
    """Base test class for factory functions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_create_steady_diffusion(self):
        """Test create_steady_diffusion factory."""
        bkd = self.bkd()
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        physics = create_steady_diffusion(basis, bkd, diffusion=2.0)

        self.assertEqual(physics.ncomponents(), 1)
        self.assertEqual(physics.nstates(), npts)

    def test_create_advection_diffusion(self):
        """Test create_advection_diffusion factory."""
        bkd = self.bkd()
        npts = 8
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)

        velocity = [bkd.ones((npts,))]
        physics = create_advection_diffusion(
            basis, bkd, diffusion=1.0, velocity=velocity
        )

        self.assertEqual(physics.ncomponents(), 1)


class TestADRTransient(Generic[Array], unittest.TestCase):
    """Test transient ADR with manufactured solutions.

    Uses quadratic-in-time manufactured solution so CN (2nd order) is
    exact while BE (1st order) has O(dt) temporal error.
    Solution: u = sin(pi*x)*(1 + T + T**2), homogeneous Dirichlet BCs.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def _setup_transient_adr(self):
        """Create ADR physics with quadratic-in-time manufactured solution."""
        bkd = self.bkd()
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        mesh_obj = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        man_sol = ManufacturedAdvectionDiffusionReaction(
            sol_str="sin(pi*x)*(1 + T + T**2)",
            nvars=1,
            diff_str="0.1",
            react_str="0",
            vel_strs=["0"],
            bkd=bkd,
            oned=True,
        )

        def forcing_fn(t):
            return man_sol.functions["forcing"](nodes[None, :], t)

        physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=0.1, forcing=forcing_fn
        )

        # Homogeneous Dirichlet BCs (sin(pi*x) = 0 at x = +-1)
        left_idx = mesh_obj.boundary_indices(0)
        right_idx = mesh_obj.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        return physics, man_sol, nodes, bkd

    def _run_transient_adr(self, method, atol):
        physics, man_sol, nodes, bkd = self._setup_transient_adr()
        model = CollocationModel(physics, bkd)

        u0 = man_sol.functions["solution"](nodes[None, :], 0.0)

        final_time = 0.1
        config = TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=final_time,
            deltat=0.005,
        )

        solutions, times = model.solve_transient(u0, config)
        t_final = float(bkd.to_numpy(times[-1]))
        u_exact = man_sol.functions["solution"](nodes[None, :], t_final)

        bkd.assert_allclose(solutions[:, -1], u_exact, atol=atol)

    def test_transient_backward_euler(self):
        """Test transient ADR with backward Euler.

        BE is 1st order in time, so with dt=0.005 and quadratic-in-time
        solution there is O(dt) temporal error.
        """
        self._run_transient_adr("backward_euler", atol=0.01)

    def test_transient_crank_nicolson(self):
        """Test transient ADR with Crank-Nicolson.

        CN is 2nd order in time. For quadratic-in-time solution,
        CN integrates the time derivative exactly.
        """
        self._run_transient_adr("crank_nicolson", atol=1e-8)


# NumPy backend tests
class TestAdvectionDiffusionReactionNumpy(TestAdvectionDiffusionReaction):
    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


class TestAdvectionDiffusionReactionWithParamNumpy(
    TestAdvectionDiffusionReactionWithParam
):
    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


class TestFactoryFunctionsNumpy(TestFactoryFunctions):
    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


class TestADRTransientNumpy(TestADRTransient):
    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


if __name__ == "__main__":
    unittest.main()
