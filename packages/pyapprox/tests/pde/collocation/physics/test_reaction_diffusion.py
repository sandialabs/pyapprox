"""Tests for reaction-diffusion physics implementations."""

import math
from typing import Optional

import numpy as np
import pytest

from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import (
    zero_dirichlet_bc,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
)
from pyapprox.pde.collocation.physics.coefficients import (
    CoefficientInput,
    NodalField,
    NodalSupplier,
)
from pyapprox.pde.collocation.physics.fitzhugh_nagumo import (
    FitzHughNagumoPhysics,
    create_fitzhugh_nagumo,
)
from pyapprox.pde.collocation.physics.reaction_diffusion import (
    FitzHughNagumoReaction,
    LinearReaction,
    ReactionProtocol,
    SymbolicReactionProtocol,
    TwoSpeciesReactionDiffusionPhysics,
    create_two_species_reaction_diffusion,
)
from pyapprox.pde.collocation.protocols.physics import (
    CollocationInteriorOperatorProtocol,
)
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.manufactured import (
    ManufacturedTwoSpeciesReactionDiffusion,
)
from pyapprox.pde.models.collocation.right_hand_side import (
    CollocationRightHandSide,
)
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.physics_test_utils import (
    ManufacturedColumn,
    PhysicsTestBase,
)


class TestTwoSpeciesReactionDiffusion(PhysicsTestBase):
    """Tests for TwoSpeciesReactionDiffusionPhysics."""

    def test_jacobian_linear_reaction(self, bkd: Backend[Array]) -> None:
        """Test Jacobian with linear reaction."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        # Linear reaction: R0 = -u0 + 0.5*u1, R1 = 0.5*u0 - u1
        reaction = LinearReaction(-1.0, 0.5, 0.5, -1.0, bkd)

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis, bkd, diffusion0=0.1, diffusion1=0.05, reaction=reaction
        )

        # Random state
        state = bkd.array(0.5 + 0.3 * np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_jacobian_fhn_reaction(self, bkd: Backend[Array]) -> None:
        """Test Jacobian with FitzHugh-Nagumo reaction."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        reaction = FitzHughNagumoReaction(
            alpha=0.1, eps=0.01, beta=0.5, gamma=1.0, bkd=bkd
        )

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis, bkd, diffusion0=1e-3, diffusion1=0.0, reaction=reaction
        )

        # State in reasonable range for FHN
        state = bkd.array(0.5 + 0.2 * np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_jacobian_with_forcing(self, bkd: Backend[Array]) -> None:
        """Test Jacobian with forcing terms."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        reaction = LinearReaction(0.0, 0.0, 0.0, 0.0, bkd)
        forcing0 = bkd.sin(math.pi * nodes)
        forcing1 = bkd.cos(math.pi * nodes)

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis,
            bkd,
            diffusion0=0.1,
            diffusion1=0.1,
            reaction=reaction,
            forcing0=NodalField(forcing0),
            forcing1=NodalField(forcing1),
        )

        state = bkd.array(np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_factory_function(self, bkd: Backend[Array]) -> None:
        """Test create_two_species_reaction_diffusion factory."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        reaction = LinearReaction(0.0, 0.0, 0.0, 0.0, bkd)

        physics = create_two_species_reaction_diffusion(
            basis,
            bkd,
            diffusion0=0.1,
            diffusion1=0.05,
            reaction=reaction,
        )

        assert physics.ncomponents() == 2
        assert physics.nstates() == 2 * npts

    def test_transient_diffusion_only(self, bkd: Backend[Array]) -> None:
        """Test transient evolution with pure diffusion (no reaction).

        For u0, u1 = sin(pi*x) with homogeneous Dirichlet BCs and no reaction:
        du/dt = D * d²u/dx²
        Exact solution: u(x,t) = exp(-D*pi²*t) * sin(pi*x)
        """
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        D0, D1 = 0.1, 0.05
        reaction = LinearReaction(0.0, 0.0, 0.0, 0.0, bkd)

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis, bkd, diffusion0=D0, diffusion1=D1, reaction=reaction
        )

        # Set homogeneous Dirichlet BCs for both species
        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_u0_left = zero_dirichlet_bc(bkd, bkd.array([left_idx]))
        bc_u0_right = zero_dirichlet_bc(bkd, bkd.array([right_idx]))
        bc_u1_left = zero_dirichlet_bc(bkd, bkd.array([npts + left_idx]))
        bc_u1_right = zero_dirichlet_bc(bkd, bkd.array([npts + right_idx]))
        physics.set_boundary_conditions(
            [bc_u0_left, bc_u0_right, bc_u1_left, bc_u1_right]
        )

        model = CollocationModel(physics, bkd)

        # Initial conditions
        u0_init = bkd.sin(math.pi * nodes)
        u1_init = bkd.sin(math.pi * nodes)
        state0 = bkd.hstack([u0_init, u1_init])

        final_time = 0.1
        config: TimeIntegrationConfig[Array] = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=final_time,
            deltat=0.005,
            newton_tol=1e-10,
            newton_maxiter=20,
            verbosity=0,
        )

        solutions, times = model.solve_transient(state0, config)

        # Exact solutions at final time
        u0_exact = math.exp(-D0 * math.pi**2 * final_time) * bkd.sin(math.pi * nodes)
        u1_exact = math.exp(-D1 * math.pi**2 * final_time) * bkd.sin(math.pi * nodes)
        exact_final = bkd.hstack([u0_exact, u1_exact])

        bkd.assert_allclose(solutions[:, -1], exact_final, rtol=0.02, atol=1e-10)

    def test_transient_diffusion_only_crank_nicolson(self, bkd: Backend[Array]) -> None:
        """Test transient reaction-diffusion with CN and manufactured solution.

        Uses polynomial-in-space (exact for Chebyshev) and quadratic-in-time
        (exact for CN) with zero reaction. Tight tolerance since both spatial
        and temporal discretization errors are near machine precision.
        """
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        D0, D1 = 0.1, 0.05
        reaction = LinearReaction(0.0, 0.0, 0.0, 0.0, bkd)

        man_sol = ManufacturedTwoSpeciesReactionDiffusion(
            sol_strs=[
                "(1 - x**2)**2*(1 + T + T**2)",
                "(1 - x**2)**2*x*(1 + T + T**2)",
            ],
            nvars=1,
            diff_strs=[str(D0), str(D1)],
            reaction=reaction,
            bkd=bkd,
            oned=True,
        )

        forcing0_fn: TimeDependent[Array] = TimeDependent(
            ManufacturedColumn(man_sol.functions["forcing"], 0)
        )
        forcing1_fn: TimeDependent[Array] = TimeDependent(
            ManufacturedColumn(man_sol.functions["forcing"], 1)
        )

        physics = TwoSpeciesReactionDiffusionPhysics(
            basis,
            bkd,
            diffusion0=D0,
            diffusion1=D1,
            reaction=reaction,
            forcing0=NodalSupplier(forcing0_fn, nodes[None, :]),
            forcing1=NodalSupplier(forcing1_fn, nodes[None, :]),
        )

        left_idx = mesh.boundary_indices(0)
        right_idx = mesh.boundary_indices(1)
        bc_u0_left = zero_dirichlet_bc(bkd, bkd.array([left_idx]))
        bc_u0_right = zero_dirichlet_bc(bkd, bkd.array([right_idx]))
        bc_u1_left = zero_dirichlet_bc(bkd, bkd.array([npts + left_idx]))
        bc_u1_right = zero_dirichlet_bc(bkd, bkd.array([npts + right_idx]))
        physics.set_boundary_conditions(
            [bc_u0_left, bc_u0_right, bc_u1_left, bc_u1_right]
        )

        model = CollocationModel(physics, bkd)

        u_exact_0 = man_sol.functions["solution"](nodes[None, :], 0.0)
        state0 = bkd.concatenate([u_exact_0[:, 0], u_exact_0[:, 1]])

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

        solutions, times = model.solve_transient(state0, config)
        t_final = float(bkd.to_numpy(times[-1]))
        u_exact_final = man_sol.functions["solution"](nodes[None, :], t_final)
        exact_final = bkd.concatenate([u_exact_final[:, 0], u_exact_final[:, 1]])

        bkd.assert_allclose(solutions[:, -1], exact_final, atol=1e-8)


class TestFitzHughNagumoPhysics(PhysicsTestBase):
    """Tests for FitzHughNagumoPhysics."""

    def test_jacobian_derivative_checker(self, bkd: Backend[Array]) -> None:
        """Test Jacobian matches finite differences."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = FitzHughNagumoPhysics(
            basis, bkd, diffusion_v=1e-3, alpha=0.1, eps=0.01, beta=0.5, gamma=1.0
        )

        # State in reasonable FHN range
        np.random.seed(42)
        state = bkd.array(0.3 + 0.2 * np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_jacobian_different_parameters(self, bkd: Backend[Array]) -> None:
        """Test Jacobian with different FHN parameters."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        # Different parameter values
        physics = FitzHughNagumoPhysics(
            basis, bkd, diffusion_v=5e-4, alpha=0.2, eps=0.1, beta=1.0, gamma=0.5
        )

        np.random.seed(123)
        state = bkd.array(0.5 + 0.3 * np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_parameter_accessors(self, bkd: Backend[Array]) -> None:
        """Test parameter accessor methods."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = FitzHughNagumoPhysics(
            basis, bkd, alpha=0.15, eps=0.02, beta=0.6, gamma=0.9
        )

        assert abs(physics.alpha() - 0.15) < 1e-7
        assert abs(physics.eps() - 0.02) < 1e-7
        assert abs(physics.beta() - 0.6) < 1e-7
        assert abs(physics.gamma() - 0.9) < 1e-7

    def test_set_parameters(self, bkd: Backend[Array]) -> None:
        """Test updating FHN parameters."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = FitzHughNagumoPhysics(basis, bkd)

        # Update parameters
        physics.set_parameters(alpha=0.2, eps=0.05)

        assert abs(physics.alpha() - 0.2) < 1e-7
        assert abs(physics.eps() - 0.05) < 1e-7
        # Unchanged parameters
        assert abs(physics.beta() - 0.5) < 1e-7
        assert abs(physics.gamma() - 1.0) < 1e-7

    def test_factory_function(self, bkd: Backend[Array]) -> None:
        """Test create_fitzhugh_nagumo factory."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = create_fitzhugh_nagumo(
            basis, bkd, diffusion_v=1e-3, alpha=0.1, eps=0.01
        )

        assert physics.ncomponents() == 2
        assert physics.nstates() == 2 * npts
        assert abs(physics.alpha() - 0.1) < 1e-7

    def test_no_diffusion_on_recovery(self, bkd: Backend[Array]) -> None:
        """Verify recovery variable has zero diffusion."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = FitzHughNagumoPhysics(basis, bkd, diffusion_v=1e-3)

        d0, d1 = physics.diffusion_functions()
        assert d0 is not None
        bkd.assert_allclose(d0.values(0.0), bkd.full((npts,), 1e-3))
        assert d1 is None

    def test_transient_evolution(self, bkd: Backend[Array]) -> None:
        """Test FitzHugh-Nagumo transient behavior.

        FHN is inherently transient with excitable dynamics. We test that
        the system evolves and doesn't blow up with physically reasonable
        initial conditions.
        """
        npts = 25
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        physics = FitzHughNagumoPhysics(
            basis, bkd, diffusion_v=1e-2, alpha=0.1, eps=0.01, beta=0.5, gamma=1.0
        )

        model = CollocationModel(physics, bkd)

        # Initial condition: small localized excitation
        v0 = 0.5 * bkd.exp(-10.0 * nodes**2)  # Excitatory variable
        w0 = bkd.zeros((npts,))  # Recovery variable starts at rest
        state0 = bkd.hstack([v0, w0])

        config: TimeIntegrationConfig[Array] = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=1.0,
            deltat=0.01,
            newton_tol=1e-10,
            newton_maxiter=20,
            verbosity=0,
        )

        solutions, times = model.solve_transient(state0, config)

        # Check solution is finite and bounded
        final_state = solutions[:, -1]
        assert bkd.isfinite(bkd.norm(final_state))
        assert float(bkd.norm(final_state)) < 100.0

        # Check that the system evolved (not stuck at initial)
        initial_norm = float(bkd.norm(state0))
        final_norm = float(bkd.norm(final_state))
        # FHN dynamics should change the state
        assert abs(initial_norm - final_norm) >= 10 ** (-2)


class TestFitzHughNagumoReaction:
    """Unit tests for FitzHughNagumoReaction class."""

    def test_reaction_values(self, bkd: Backend[Array]) -> None:
        """Test reaction term evaluation."""
        reaction = FitzHughNagumoReaction(0.1, 0.01, 0.5, 1.0, bkd)

        u0 = bkd.array([0.0, 0.5, 1.0])
        u1 = bkd.array([0.0, 0.1, 0.0])

        R0, R1 = reaction(u0, u1)

        # Manual calculation:
        # R0[0] = 0*(1-0)*(0-0.1) - 0 = 0
        # R0[1] = 0.5*(1-0.5)*(0.5-0.1) - 0.1 = 0.5*0.5*0.4 - 0.1 = 0.1 - 0.1 = 0
        # R0[2] = 1*(1-1)*(1-0.1) - 0 = 0
        expected_R0 = bkd.array([0.0, 0.0, 0.0])
        bkd.assert_allclose(R0, expected_R0, atol=1e-10)

        # R1[0] = 0.01*(0.5*0 - 1*0) = 0
        # R1[1] = 0.01*(0.5*0.5 - 1*0.1) = 0.01*(0.25 - 0.1) = 0.0015
        # R1[2] = 0.01*(0.5*1 - 1*0) = 0.005
        expected_R1 = bkd.array([0.0, 0.0015, 0.005])
        bkd.assert_allclose(R1, expected_R1, atol=1e-10)

    def test_reaction_jacobian(self, bkd: Backend[Array]) -> None:
        """Test reaction Jacobian computation."""
        reaction = FitzHughNagumoReaction(0.1, 0.01, 0.5, 1.0, bkd)

        u0 = bkd.array([0.5])
        u1 = bkd.array([0.1])

        dR0_du0, dR0_du1, dR1_du0, dR1_du1 = reaction.jacobian(u0, u1)

        # Check against finite differences
        eps = 1e-7
        R0_base, R1_base = reaction(u0, u1)

        R0_plus, _ = reaction(u0 + eps, u1)
        fd_dR0_du0 = (R0_plus - R0_base) / eps
        bkd.assert_allclose(dR0_du0, fd_dR0_du0, atol=1e-5)

        _, R1_plus = reaction(u0 + eps, u1)
        fd_dR1_du0 = (R1_plus - R1_base) / eps
        bkd.assert_allclose(dR1_du0, fd_dR1_du0, atol=1e-5)


_NPTS_RD = 12


def _rd_basis(bkd: Backend[Array]) -> ChebyshevBasis1D[Array]:
    return ChebyshevBasis1D(TransformedMesh1D(_NPTS_RD, bkd), bkd)


def _pulsing(coords: Array, time: float) -> Array:
    return (1.0 + time) * coords[0]


def _rd_physics(
    bkd: Backend[Array],
    diffusion0: CoefficientInput[Array],
    forcing0: Optional[CoefficientInput[Array]] = None,
) -> TwoSpeciesReactionDiffusionPhysics[Array]:
    return TwoSpeciesReactionDiffusionPhysics(
        _rd_basis(bkd),
        bkd,
        diffusion0=diffusion0,
        diffusion1=0.0,
        reaction=FitzHughNagumoReaction(0.1, 0.01, 0.5, 1.0, bkd),
        forcing0=forcing0,
    )


def _time_dependent(bkd: Backend[Array]) -> NodalSupplier[Array]:
    supplier: TimeDependent[Array] = TimeDependent(_pulsing)
    nodes = bkd.reshape(_rd_basis(bkd).nodes(), (1, _NPTS_RD))
    return NodalSupplier(supplier, nodes)


def _rd_state(bkd: Backend[Array], seed: int = 3) -> Array:
    return bkd.asarray(np.random.default_rng(seed).uniform(0.0, 1.0, 2 * _NPTS_RD))


class TestReactionProtocols:
    def test_reactions_satisfy_the_protocols(self, bkd: Backend[Array]) -> None:
        linear = LinearReaction(1.0, -0.5, 0.2, -1.0, bkd)
        kinetics = FitzHughNagumoReaction(0.1, 0.01, 0.5, 1.0, bkd)
        for reaction in (linear, kinetics):
            assert isinstance(reaction, ReactionProtocol)
            assert isinstance(reaction, SymbolicReactionProtocol)

    def test_non_reaction_refused(self, numpy_bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="ReactionProtocol"):
            TwoSpeciesReactionDiffusionPhysics(
                _rd_basis(numpy_bkd),
                numpy_bkd,
                diffusion0=1.0,
                diffusion1=0.0,
                reaction=object(),  # type: ignore[arg-type]
            )


class TestReactionDiffusionInterior:
    def test_variable_diffusion_and_time_dependent_forcing(
        self, bkd: Backend[Array]
    ) -> None:
        """A time-dependent forcing leaves the Jacobian time-invariant but
        not the operator; the bundle passes the checker."""
        x = _rd_basis(bkd).nodes()
        physics = _rd_physics(
            bkd, NodalField(0.1 + 0.05 * x * x), forcing0=_time_dependent(bkd)
        )
        assert isinstance(physics, CollocationInteriorOperatorProtocol)
        assert physics.interior_jacobian_is_time_invariant()
        assert not physics.interior_is_time_invariant()
        rhs = CollocationRightHandSide(physics)
        jacobian = rhs.derivatives().jacobian
        assert jacobian is not None and not jacobian.affine()
        report = ODEResidualBundleChecker(rhs).check(_rd_state(bkd), 0.3)
        assert report.failures() == []

    def test_time_dependent_diffusion(self, bkd: Backend[Array]) -> None:
        """The diffusivity carries the Jacobian's time dependence, and the
        interface flux, which takes no time, refuses it."""
        physics = _rd_physics(bkd, _time_dependent(bkd))
        assert not physics.interior_jacobian_is_time_invariant()
        with pytest.raises(ValueError, match="takes no time"):
            physics.compute_interface_flux(
                _rd_state(bkd), bkd.array([0], dtype=int), bkd.array([1.0])
            )
