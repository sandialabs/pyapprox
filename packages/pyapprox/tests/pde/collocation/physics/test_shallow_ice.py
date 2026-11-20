"""Tests for Shallow Ice physics implementation."""

from typing import Optional, Tuple

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis2D
from pyapprox.pde.collocation.boundary import (
    constant_dirichlet_bc,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    TransformedMesh2D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.physics.coefficients import (
    CoefficientInput,
    NodalField,
    NodalSupplier,
)
from pyapprox.pde.collocation.physics.shallow_ice import (
    ShallowIcePhysics,
    create_shallow_ice,
)
from pyapprox.pde.collocation.protocols.basis import TensorProductBasisProtocol
from pyapprox.pde.collocation.protocols.physics import (
    CollocationInteriorOperatorProtocol,
)
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    TimeDependent,
    TimeIndependent,
)
from pyapprox.pde.manufactured.shallow_ice import (
    ManufacturedShallowIce,
)
from pyapprox.pde.models.collocation.right_hand_side import (
    CollocationRightHandSide,
)
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.physics_test_utils import (
    PhysicsDerivativeWrapper,
    PhysicsTestBase,
)


class TestShallowIcePhysics(PhysicsTestBase):
    """Tests for ShallowIcePhysics."""

    def test_jacobian_derivative_checker(self, bkd: Backend[Array]) -> None:
        """Test Jacobian matches finite differences using DerivativeChecker."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        # Flat bed
        bed = bkd.zeros((npts,))

        physics = ShallowIcePhysics(
            basis, bkd, bed=NodalField(bed), friction=1e6, A=1e-16, rho=917.0
        )

        # Positive ice thickness (physically meaningful)
        state = 100.0 + 50.0 * (1.0 - nodes**2)

        # Exact Jacobian: limited only by the checker's FD noise
        # (measured sweep bottom ~1.8e-7)
        self.check_jacobian(bkd, physics, state, time=0.0, atol=1e-6)

    def test_jacobian_sloped_bed(self, bkd: Backend[Array]) -> None:
        """Test Jacobian with sloped bed topography."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        # Sloped bed
        bed = 100.0 * nodes

        physics = ShallowIcePhysics(
            basis, bkd, bed=NodalField(bed), friction=1e6, A=1e-16, rho=917.0
        )

        # Positive ice thickness
        state = 150.0 + 30.0 * (1.0 - nodes**2)

        # Exact Jacobian: limited only by the checker's FD noise
        # (measured sweep bottom ~8e-8)
        self.check_jacobian(bkd, physics, state, time=0.0, atol=1e-6)

    def test_factory_function(self, bkd: Backend[Array]) -> None:
        """Test create_shallow_ice factory function."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        bed = bkd.zeros((npts,))

        physics = create_shallow_ice(
            basis, bkd, bed=NodalField(bed), friction=1e6, A=1e-16, rho=917.0
        )

        assert physics.ncomponents() == 1
        assert physics.nstates() == npts

    def _setup_transient_shallow_ice(
        self, bkd: Backend[Array]
    ) -> Tuple[Array, ManufacturedShallowIce[Array], ShallowIcePhysics[Array]]:
        """Set up transient shallow ice with manufactured solution.

        Uses polynomial-in-space (exact for Chebyshev) and quadratic-in-time
        (exact for CN). Boundary values are constant: H(±1,t)=2.
        Normalized parameters (A=1, rho=1) for numerical stability.
        """
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        bc_mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        A = 1.0
        rho = 1.0

        man_sol = ManufacturedShallowIce(
            sol_str="2 + 0.5*(1 - x**2)*(1 + T + T**2)",
            nvars=1,
            bed_str="0.1*x",
            friction_str="1.0",
            A=A,
            rho=rho,
            bkd=bkd,
            oned=True,
        )

        manufactured_forcing: TimeDependent[Array] = TimeDependent(
            man_sol.functions["forcing"]
        )

        bed = man_sol.functions["bed"](nodes[None, :])
        if bed.ndim == 2:
            bed = bed[:, 0]

        physics = ShallowIcePhysics(
            basis,
            bkd,
            bed=NodalField(bed),
            friction=1.0,
            A=A,
            rho=rho,
            forcing=NodalSupplier(manufactured_forcing, nodes[None, :]),
        )

        exact_at_0 = man_sol.functions["solution"](nodes[None, :], 0.0)
        left_idx = bc_mesh.boundary_indices(0)
        right_idx = bc_mesh.boundary_indices(1)
        li = bkd.to_int(left_idx)
        ri = bkd.to_int(right_idx)
        bc_left = constant_dirichlet_bc(bkd, left_idx, bkd.to_float(exact_at_0[li]))
        bc_right = constant_dirichlet_bc(bkd, right_idx, bkd.to_float(exact_at_0[ri]))
        physics.set_boundary_conditions([bc_left, bc_right])

        return nodes, man_sol, physics

    def _run_transient_shallow_ice(
        self, bkd: Backend[Array], method: str, atol: float
    ) -> None:
        """Run transient shallow ice test with given method and tolerance."""
        nodes, man_sol, physics = self._setup_transient_shallow_ice(bkd)
        model = CollocationModel(physics, bkd)

        state0 = man_sol.functions["solution"](nodes[None, :], 0.0)

        final_time = 0.1
        config: TimeIntegrationConfig[Array] = TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=final_time,
            deltat=0.01,
            newton_tol=1e-10,
            newton_maxiter=20,
            verbosity=0,
        )

        solutions, times = model.solve_transient(state0, config)
        t_final = float(bkd.to_numpy(times[-1]))
        exact_final = man_sol.functions["solution"](nodes[None, :], t_final)

        bkd.assert_allclose(solutions[:, -1], exact_final, atol=atol)

    @staticmethod
    def _make_shallow_ice(
        bkd: Backend[Array],
        ndim: int,
        bed_type: str,
        friction: float,
        glen_exponent: float,
        forcing: Optional[CoefficientInput[Array]] = None,
    ) -> Tuple[ShallowIcePhysics[Array], int]:
        """Build normalized (A=1, rho=1) shallow-ice physics on a
        1D or 2D Chebyshev basis with flat or varying bed."""
        basis: TensorProductBasisProtocol[Array]
        if ndim == 1:
            basis = ChebyshevBasis1D(TransformedMesh1D(10, bkd), bkd)
            coords = [basis.nodes()]
        else:
            mesh = TransformedMesh2D(5, 5, bkd)
            basis = ChebyshevBasis2D(mesh, bkd)
            pts = mesh.points()
            coords = [pts[0], pts[1]]
        npts = basis.npts()
        # O(1) bed keeps residual magnitudes moderate so the
        # DerivativeChecker FD sweep bottoms out cleanly
        if bed_type == "flat":
            bed = bkd.zeros((npts,))
        else:
            bed = 0.5 * coords[0]
            if ndim == 2:
                bed = bed + 0.25 * coords[1] ** 2
        physics = ShallowIcePhysics(
            basis,
            bkd,
            bed=NodalField(bed),
            friction=friction,
            A=1.0,
            rho=1.0,
            forcing=forcing,
            glen_exponent=glen_exponent,
        )
        return physics, npts

    @pytest.mark.parametrize("glen_exponent", [3.0, 4.0])
    @pytest.mark.parametrize("ndim", [1, 2])
    @pytest.mark.parametrize("bed_type", ["flat", "varying"])
    @pytest.mark.parametrize("friction", [float("inf"), 1.0])
    def test_jacobian_parametrized(
        self,
        bkd: Backend[Array],
        glen_exponent: float,
        ndim: int,
        bed_type: str,
        friction: float,
    ) -> None:
        """Exact Jacobian vs DerivativeChecker across n, dimension,
        bed topography, and sliding regimes.

        friction=inf makes friction_frac exactly zero (no sliding
        term); H > 0 is required for the fractional powers H^(n+1).
        """
        physics, npts = self._make_shallow_ice(
            bkd, ndim, bed_type, friction, glen_exponent
        )
        wrapper = PhysicsDerivativeWrapper(physics, time=0.0)
        checker = DerivativeChecker(wrapper)
        for _ in range(3):
            state = bkd.asarray(2.0 + 0.5 * np.random.uniform(-1.0, 1.0, (npts,)))
            errors = checker.check_derivatives(state[:, None], verbosity=0)
            # The FD sweep is V-shaped (truncation decay then roundoff
            # climb) bottoming near 9e-8; a wrong Jacobian plateaus
            # instead. Assert the measured bottom plus a calibrated
            # ratio (marginal cases reach ~1.3e-6).
            assert float(bkd.min(errors[0])) <= 1e-6
            assert checker.error_ratio(errors[0]) <= 5e-6

    def test_jacobian_forcing_independent(self, bkd: Backend[Array]) -> None:
        """dR/dH must be identical with and without forcing (additive,
        state-independent term)."""
        physics_plain, npts = self._make_shallow_ice(bkd, 2, "varying", 1.0, 3.0)
        forcing_vals = bkd.asarray(np.random.uniform(-1.0, 1.0, (npts,)))
        physics_forced, _ = self._make_shallow_ice(
            bkd, 2, "varying", 1.0, 3.0, forcing=NodalField(forcing_vals)
        )
        state = bkd.asarray(2.0 + 0.5 * np.random.uniform(-1.0, 1.0, (npts,)))
        jac_plain = physics_plain.jacobian(state, 0.0)
        jac_forced = physics_forced.jacobian(state, 0.0)
        bkd.assert_allclose(jac_forced, jac_plain, rtol=1e-12)

        wrapper = PhysicsDerivativeWrapper(physics_forced, time=0.0)
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(state[:, None], verbosity=0)
        assert checker.error_ratio(errors[0]) <= 1e-6

    def test_diffusion_derivatives_n3_identity(self, bkd: Backend[Array]) -> None:
        """Exact n=3 collapse pinning the eps-regularization convention.

        For n=3: kappa_G = gamma*H^5 exactly (G^0 = 1, no eps
        dependence) and kappa_H = 5*gamma*H^4*(grad_s_sq+eps) +
        2*friction_frac*H.
        """
        friction = 2.0
        physics, npts = self._make_shallow_ice(bkd, 1, "varying", friction, 3.0)
        state = bkd.asarray(2.0 + 0.5 * np.random.uniform(-1.0, 1.0, (npts,)))
        gravity = 9.81
        gamma = 2.0 * 1.0 * (1.0 * gravity) ** 3 / 5.0
        friction_frac = 1.0 * gravity / friction
        _, grad_s_sq = physics._compute_surface_gradient(state, 0.0)
        kappa_h, kappa_g = physics._compute_diffusion_derivatives(
            state, grad_s_sq, bkd.full((npts,), friction_frac)
        )
        eps = 1e-12
        bkd.assert_allclose(kappa_g, gamma * state**5, rtol=1e-12)
        bkd.assert_allclose(
            kappa_h,
            5.0 * gamma * state**4 * (grad_s_sq + eps) + 2.0 * friction_frac * state,
            rtol=1e-12,
        )

    def test_jacobian_nonsymmetric_sloped_bed(self, bkd: Backend[Array]) -> None:
        """Guard against accidental symmetrization: the exact SIA
        Jacobian is nonsymmetric on a sloped bed."""
        physics, npts = self._make_shallow_ice(bkd, 1, "varying", 1.0, 3.0)
        state = bkd.asarray(2.0 + 0.5 * np.random.uniform(-1.0, 1.0, (npts,)))
        jac = physics.jacobian(state, 0.0)
        asym_norm = float(bkd.norm(jac - jac.T))
        assert asym_norm > 1e-8 * float(bkd.norm(jac))

    def test_transient_manufactured_backward_euler(self, bkd: Backend[Array]) -> None:
        """Test transient shallow ice with backward Euler.

        Time-discretization-limited (measured final error ~7.8e-8 at
        deltat=0.01 on both backends).
        """
        self._run_transient_shallow_ice(bkd, "backward_euler", atol=1e-6)

    def test_transient_manufactured_crank_nicolson(self, bkd: Backend[Array]) -> None:
        """Test transient shallow ice with Crank-Nicolson.

        Uses polynomial-in-space and quadratic-in-time manufactured solution.
        CN integrates quadratic-in-time exactly, so only roundoff remains
        (measured final error ~1.3e-13).
        """
        self._run_transient_shallow_ice(bkd, "crank_nicolson", atol=1e-11)


_NPTS_SIA = 10


def _sia_nodes(bkd: Backend[Array]) -> Array:
    return ChebyshevBasis1D(TransformedMesh1D(_NPTS_SIA, bkd), bkd).nodes()


def _growing(coords: Array, time: float) -> Array:
    return (1.0 + time) * (1.5 + 0.5 * coords[0])


def _fixed(coords: Array) -> Array:
    return 1.5 + 0.5 * coords[0]


def _sia(
    bkd: Backend[Array],
    bed: CoefficientInput[Array],
    friction: CoefficientInput[Array] = 1.0,
    forcing: Optional[CoefficientInput[Array]] = None,
) -> ShallowIcePhysics[Array]:
    basis = ChebyshevBasis1D(TransformedMesh1D(_NPTS_SIA, bkd), bkd)
    return ShallowIcePhysics(
        basis, bkd, bed=bed, friction=friction, A=1.0, rho=1.0, forcing=forcing
    )


def _declared(bkd: Backend[Array], dependent: bool) -> NodalSupplier[Array]:
    nodes = bkd.reshape(_sia_nodes(bkd), (1, _NPTS_SIA))
    if dependent:
        growing: TimeDependent[Array] = TimeDependent(_growing)
        return NodalSupplier(growing, nodes)
    fixed: TimeIndependent[Array] = TimeIndependent(_fixed)
    return NodalSupplier(fixed, nodes)


def _sia_state(bkd: Backend[Array]) -> Array:
    rng = np.random.default_rng(5)
    return bkd.asarray(2.0 + 0.5 * rng.uniform(-1.0, 1.0, _NPTS_SIA))


class TestShallowIceInterior:
    def test_time_dependent_forcing(self, bkd: Backend[Array]) -> None:
        """A time-dependent forcing leaves the Jacobian time-invariant but
        not the operator; the bundle passes the checker."""
        physics = _sia(
            bkd, NodalField(0.1 * _sia_nodes(bkd)), forcing=_declared(bkd, True)
        )
        assert isinstance(physics, CollocationInteriorOperatorProtocol)
        assert physics.interior_jacobian_is_time_invariant()
        assert not physics.interior_is_time_invariant()
        rhs = CollocationRightHandSide(physics)
        report = ODEResidualBundleChecker(rhs).check(_sia_state(bkd), 0.3)
        assert report.failures() == []

    def test_declared_coefficients(self, bkd: Backend[Array]) -> None:
        """Bed and friction carry the Jacobian's time dependence; declared
        time-independent suppliers keep it invariant."""
        flat = NodalField(bkd.zeros((_NPTS_SIA,)))
        moving_bed = _sia(bkd, _declared(bkd, True))
        assert not moving_bed.interior_jacobian_is_time_invariant()
        moving_friction = _sia(bkd, flat, friction=_declared(bkd, True))
        assert not moving_friction.interior_jacobian_is_time_invariant()
        fixed = _sia(bkd, _declared(bkd, False), friction=_declared(bkd, False))
        assert fixed.interior_is_time_invariant()

    @pytest.mark.parametrize("coefficient", ["bed", "friction"])
    def test_coefficient_read_at_evaluation_time(
        self, bkd: Backend[Array], coefficient: str
    ) -> None:
        """A time-dependent bed or friction gives, at time t, the residual
        and Jacobian of the physics built from its values at t."""
        time = 0.7
        frozen = NodalField(_declared(bkd, True).values(time))
        flat = NodalField(bkd.zeros((_NPTS_SIA,)))
        if coefficient == "bed":
            moving = _sia(bkd, _declared(bkd, True))
            reference = _sia(bkd, frozen)
        else:
            moving = _sia(bkd, flat, friction=_declared(bkd, True))
            reference = _sia(bkd, flat, friction=frozen)
        state = _sia_state(bkd)
        residual = moving.residual(state, time)
        bkd.assert_allclose(residual, reference.residual(state, 0.0), rtol=1e-12)
        bkd.assert_allclose(
            moving.jacobian(state, time),
            reference.jacobian(state, 0.0),
            rtol=1e-12,
        )
        # and it changes with time, so the comparison is not vacuous
        change = bkd.max(bkd.abs(moving.residual(state, 0.0) - residual))
        assert float(change) > 1e-3
