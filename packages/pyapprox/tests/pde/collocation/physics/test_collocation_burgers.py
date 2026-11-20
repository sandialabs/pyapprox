"""Tests for Burgers physics implementation."""

import math
from typing import Callable, Generic

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import (
    zero_dirichlet_bc,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
)
from pyapprox.pde.collocation.physics.burgers import (
    BurgersPhysics1D,
    create_burgers_1d,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField, NodalSupplier
from pyapprox.pde.collocation.protocols.physics import (
    CollocationInteriorOperatorProtocol,
)
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.manufactured.burgers import (
    ManufacturedBurgers1D,
)
from pyapprox.pde.models.collocation.right_hand_side import (
    CollocationRightHandSide,
)
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.physics_test_utils import PhysicsTestBase


class TestBurgersPhysics(PhysicsTestBase):
    """Tests for BurgersPhysics1D."""

    def test_jacobian_derivative_checker(self, bkd: Backend[Array]) -> None:
        """Test Jacobian matches finite differences using DerivativeChecker."""
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = BurgersPhysics1D(basis, bkd, viscosity=0.1)

        # Random state (positive to avoid any issues)
        state = bkd.array(0.5 + 0.3 * np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_jacobian_non_conservative(self, bkd: Backend[Array]) -> None:
        """Test Jacobian for non-conservative form."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = BurgersPhysics1D(basis, bkd, viscosity=0.05, conservative=False)

        state = bkd.array(0.5 + 0.3 * np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_jacobian_with_forcing(self, bkd: Backend[Array]) -> None:
        """Test Jacobian with non-zero forcing."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        forcing = 0.5 * bkd.sin(math.pi * nodes)

        physics = BurgersPhysics1D(
            basis, bkd, viscosity=0.1, forcing=NodalField(forcing)
        )

        state = bkd.array(0.5 + 0.3 * np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_residual_polynomial(self, bkd: Backend[Array]) -> None:
        """Test residual with polynomial solution.

        For u(x) = x^2 on [-1, 1]:
        du/dx = 2x
        d²u/dx² = 2
        u * du/dx = 2x^3
        Residual = nu*2 - 2*x^3 = 2*nu - 2*x^3
        """
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        nu = 0.5
        physics = BurgersPhysics1D(basis, bkd, viscosity=nu)

        # u = x^2
        nodes = basis.nodes()
        u = nodes**2

        residual = physics.residual(u, time=0.0)

        # Expected: nu * d²(x²)/dx² - x² * d(x²)/dx = nu*2 - x²*2x = 2*nu - 2*x³
        expected = 2.0 * nu - 2.0 * nodes**3
        bkd.assert_allclose(residual, expected, atol=1e-10)

    def test_factory_function(self, bkd: Backend[Array]) -> None:
        """Test create_burgers_1d factory function."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = create_burgers_1d(basis, bkd, viscosity=0.1)

        assert physics.ncomponents() == 1
        assert physics.nstates() == npts
        bkd.assert_allclose(
            physics.viscosity_function().values(0.0), bkd.full((npts,), 0.1)
        )
        assert physics.is_conservative()

    def test_transient_manufactured_solution(self, bkd: Backend[Array]) -> None:
        """Test transient evolution with time-dependent manufactured solution.

        For Burgers: du/dt = nu*d²u/dx² - u*du/dx + f
        Use manufactured solution u(x,t) = exp(-t)*sin(pi*x) which satisfies:
        du/dt = -exp(-t)*sin(pi*x)
        """
        npts = 25
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        nu = 0.1

        # Manufactured solution with time dependence: u = exp(-t)*sin(pi*x)
        man_sol = ManufacturedBurgers1D(
            sol_str="exp(-T)*sin(pi*x)",  # T is time symbol
            visc_str=str(nu),
            bkd=bkd,
            oned=True,
        )

        # Forcing from manufactured solution
        get_forcing: TimeDependent[Array] = TimeDependent(man_sol.functions["forcing"])

        physics = BurgersPhysics1D(
            basis,
            bkd,
            viscosity=nu,
            forcing=NodalSupplier(get_forcing, nodes[None, :]),
        )

        # Set homogeneous Dirichlet BCs (sin(pi*x) = 0 at x = ±1)
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)

        # Initial condition at t=0
        u0 = bkd.sin(math.pi * nodes)

        # Time integration
        final_time = 0.5
        config: TimeIntegrationConfig[Array] = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=final_time,
            deltat=0.01,
            newton_tol=1e-10,
            newton_maxiter=20,
            verbosity=0,
        )

        solutions, times = model.solve_transient(u0, config)

        # Compare to exact solution at final time
        u_exact_final = math.exp(-final_time) * bkd.sin(math.pi * nodes)
        # Use atol for boundary points near zero
        bkd.assert_allclose(solutions[:, -1], u_exact_final, rtol=0.05, atol=1e-10)

    def test_transient_manufactured_solution_crank_nicolson(
        self, bkd: Backend[Array]
    ) -> None:
        """Test transient Burgers with Crank-Nicolson and manufactured solution.

        Uses polynomial-in-space (exact for Chebyshev basis) and
        quadratic-in-time (exact for CN): u = (1-x^2)^2 * (1+T+T^2).
        With manufactured forcing, only spatial discretization error remains.
        """
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        nu = 0.1

        man_sol = ManufacturedBurgers1D(
            sol_str="(1 - x**2)**2*(1 + T + T**2)",
            visc_str=str(nu),
            bkd=bkd,
            oned=True,
        )

        get_forcing: TimeDependent[Array] = TimeDependent(man_sol.functions["forcing"])

        physics = BurgersPhysics1D(
            basis,
            bkd,
            viscosity=nu,
            forcing=NodalSupplier(get_forcing, nodes[None, :]),
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)

        u0 = man_sol.functions["solution"](nodes[None, :], 0.0)

        final_time = 0.1
        config: TimeIntegrationConfig[Array] = TimeIntegrationConfig(
            method="crank_nicolson",
            init_time=0.0,
            final_time=final_time,
            deltat=0.01,
            newton_tol=1e-10,
            newton_maxiter=20,
            verbosity=0,
        )

        solutions, times = model.solve_transient(u0, config)
        t_final = float(bkd.to_numpy(times[-1]))
        u_exact = man_sol.functions["solution"](nodes[None, :], t_final)

        bkd.assert_allclose(solutions[:, -1], u_exact, atol=1e-8)

    def test_transient_diffusion_dominated(self, bkd: Backend[Array]) -> None:
        """Test transient Burgers with high viscosity (diffusion dominated).

        When viscosity is large relative to convection, solution should
        behave like heat equation.
        """
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        # High viscosity makes it diffusion-dominated
        nu = 1.0
        physics = BurgersPhysics1D(basis, bkd, viscosity=nu)

        # Set homogeneous Dirichlet BCs
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_left = zero_dirichlet_bc(bkd, left_idx)
        bc_right = zero_dirichlet_bc(bkd, right_idx)
        physics.set_boundary_conditions([bc_left, bc_right])

        model = CollocationModel(physics, bkd)

        # Initial condition: sin(pi*x)
        u0 = bkd.sin(math.pi * nodes)

        config: TimeIntegrationConfig[Array] = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=0.5,
            deltat=0.01,
            newton_tol=1e-10,
            newton_maxiter=20,
            verbosity=0,
        )

        solutions, times = model.solve_transient(u0, config)

        # For diffusion-dominated, solution should decay like exp(-nu*pi²*t)
        # This is approximate since there's still some nonlinear convection
        decay_rate = nu * math.pi**2
        expected_decay = math.exp(-decay_rate * 0.5)

        # Check that solution has decayed
        initial_norm = float(bkd.norm(u0))
        final_norm = float(bkd.norm(solutions[:, -1]))
        assert final_norm / initial_norm < expected_decay * 1.5


_NPTS = 15
_NU = 0.1


def _basis(bkd: Backend[Array]) -> ChebyshevBasis1D[Array]:
    return ChebyshevBasis1D(TransformedMesh1D(_NPTS, bkd), bkd)


def _burgers_1d(bkd: Backend[Array], conservative: bool) -> BurgersPhysics1D[Array]:
    return BurgersPhysics1D(_basis(bkd), bkd, viscosity=_NU, conservative=conservative)


def _interpolatory_weights(bkd: Backend[Array], nodes: Array) -> Array:
    """Weights integrating every polynomial of degree < npts on the
    nodes exactly, from Chebyshev moments."""
    x = bkd.to_numpy(nodes)
    npts = x.shape[0]
    moments = np.array([2.0 / (1.0 - k**2) if k % 2 == 0 else 0.0 for k in range(npts)])
    vandermonde = np.polynomial.chebyshev.chebvander(x, npts - 1)
    return bkd.asarray(np.linalg.solve(vandermonde.T, moments))


def _random_state(bkd: Backend[Array], seed: int = 4) -> Array:
    return bkd.asarray(np.random.default_rng(seed).normal(0.0, 1.0, _NPTS))


def _flux_jump(bkd: Backend[Array], state: Array) -> Array:
    """F(1) - F(-1) for F = u^2/2 - nu u_x, read at the end nodes."""
    basis = _basis(bkd)
    nodes = basis.nodes()
    flux = 0.5 * state * state - _NU * (basis.derivative_matrix(1, 0) @ state)
    right = int(bkd.to_float(bkd.argmax(nodes)))
    left = int(bkd.to_float(bkd.argmin(nodes)))
    return flux[right] - flux[left]


class TestBurgersForms:
    def test_conservative_form_conserves(self, bkd: Backend[Array]) -> None:
        """The conservative residual integrates exactly to minus the
        boundary flux jump, for any nodal state."""
        physics = _burgers_1d(bkd, conservative=True)
        state = _random_state(bkd)
        weights = _interpolatory_weights(bkd, _basis(bkd).nodes())
        total = bkd.sum(weights * physics.residual(state, 0.0))
        jump = _flux_jump(bkd, state)
        bkd.assert_allclose(
            bkd.reshape(total, (1,)), bkd.reshape(-jump, (1,)), rtol=1e-10, atol=1e-10
        )

    def test_advective_form_does_not_conserve(self, bkd: Backend[Array]) -> None:
        """Control: u u_x at the nodes aliases, so the advective residual
        misses the flux jump."""
        physics = _burgers_1d(bkd, conservative=False)
        state = _random_state(bkd)
        weights = _interpolatory_weights(bkd, _basis(bkd).nodes())
        total = bkd.sum(weights * physics.residual(state, 0.0))
        jump = _flux_jump(bkd, state)
        assert abs(bkd.to_float(total + jump)) > 1e-3

    def test_forms_agree_without_aliasing(self, bkd: Backend[Array]) -> None:
        """Limit: for a state of degree 2, u^2 is resolved and the two
        forms coincide."""
        conservative = _burgers_1d(bkd, conservative=True)
        advective = _burgers_1d(bkd, conservative=False)
        x = _basis(bkd).nodes()
        state = 1.0 + x + 0.5 * x * x
        bkd.assert_allclose(
            conservative.residual(state, 0.0),
            advective.residual(state, 0.0),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("conservative", [True, False])
    def test_interior_bundle(self, bkd: Backend[Array], conservative: bool) -> None:
        """A general (state-dependent) Jacobian the checker accepts. With
        no parameters the bundle carries no second order."""
        physics = _burgers_1d(bkd, conservative)
        assert isinstance(physics, CollocationInteriorOperatorProtocol)
        assert physics.interior_is_time_invariant()
        rhs = CollocationRightHandSide(physics)
        jacobian = rhs.derivatives().jacobian
        assert jacobian is not None and not jacobian.affine()
        report = ODEResidualBundleChecker(rhs).check(_random_state(bkd), 0.3)
        assert report.failures() == []

    @pytest.mark.parametrize("conservative", [True, False])
    def test_state_state_hvp(self, bkd: Backend[Array], conservative: bool) -> None:
        """lambda^T (d^2 f/du^2) w is the derivative of J(u)^T lambda, by
        DerivativeChecker; the HVP assembled column by column."""
        physics = _burgers_1d(bkd, conservative)
        curvature = physics.interior_state_derivatives().state_state_hvp
        assert curvature is not None
        adj = _random_state(bkd, seed=5)
        eye = bkd.eye(_NPTS)

        def fun(state: Array) -> Array:
            return physics.jacobian(state, 0.3).T @ adj

        def jac(state: Array) -> Array:
            return bkd.stack(
                [curvature(state, adj, eye[:, k], 0.3) for k in range(_NPTS)],
                axis=1,
            )

        checker = DerivativeChecker(_VectorFunctionWrapper(bkd, _NPTS, fun, jac))
        errors = checker.check_derivatives(_random_state(bkd)[:, None], verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-6


class _VectorFunctionWrapper(Generic[Array]):
    """A function of a 1D state and its Jacobian, as DerivativeChecker
    reads them (sample (nvars, 1), value (nqoi, 1))."""

    def __init__(
        self,
        bkd: Backend[Array],
        nvars: int,
        fun: Callable[[Array], Array],
        jac: Callable[[Array], Array],
    ) -> None:
        self._bkd = bkd
        self._nvars = nvars
        self._fun = fun
        self._jac = jac

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return self._nvars

    def __call__(self, sample: Array) -> Array:
        return self._fun(sample[:, 0])[:, None]

    def jacobian(self, sample: Array) -> Array:
        return self._jac(sample[:, 0])

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.first_order(jacobian=self.jacobian)
