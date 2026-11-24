"""Tests for Helmholtz physics implementation."""

import numpy as np

from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary.essential_rows import DirichletPart
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.pde.collocation.physics.helmholtz import (
    HelmholtzPhysics,
    create_helmholtz,
)
from pyapprox.pde.collocation.protocols.physics import (
    CollocationInteriorOperatorProtocol,
)
from pyapprox.pde.models.collocation.right_hand_side import (
    CollocationRightHandSide,
)
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.physics_test_utils import (
    PhysicsTestBase,
    solve_steady_dirichlet,
)


class TestHelmholtzPhysics(PhysicsTestBase):
    """Tests for HelmholtzPhysics."""

    def test_jacobian_derivative_checker(self, bkd: Backend[Array]) -> None:
        """Test Jacobian matches finite differences using DerivativeChecker."""
        npts = 15
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = HelmholtzPhysics(basis, bkd, wave_number_sq=2.0)

        # Random state
        state = bkd.array(np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_jacobian_with_forcing(self, bkd: Backend[Array]) -> None:
        """Test Jacobian with non-zero forcing."""
        npts = 12
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        # Add forcing term
        forcing = bkd.cos(nodes)

        physics = HelmholtzPhysics(
            basis, bkd, wave_number_sq=2.0, forcing=NodalField(forcing)
        )

        state = bkd.array(np.random.randn(physics.nstates()))

        self.check_jacobian(bkd, physics, state, time=0.0)

    def test_factory_function(self, bkd: Backend[Array]) -> None:
        """Test create_helmholtz factory function."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        physics = create_helmholtz(basis, bkd, wave_number_sq=2.0)

        assert physics.ncomponents() == 1
        assert physics.nstates() == npts
        assert abs(bkd.to_float(physics.wave_number_sq()) - 2.0) < 1e-7


def _helmholtz_1d(
    bkd: Backend[Array], wave_number_sq: float, npts: int = 21
) -> HelmholtzPhysics[Array]:
    basis = ChebyshevBasis1D(TransformedMesh1D(npts, bkd), bkd)
    return HelmholtzPhysics(
        basis, bkd, wave_number_sq=wave_number_sq, forcing=NodalField(bkd.ones((npts,)))
    )


def _solve_unit_forcing(bkd: Backend[Array], wave_number_sq: float) -> Array:
    """-u'' + k^2 u = 1 on (-1, 1) with u(+-1) = 0."""
    physics = _helmholtz_1d(bkd, wave_number_sq)
    npts = physics.npts()
    mesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
    return solve_steady_dirichlet(
        physics,
        1,
        [
            DirichletPart(mesh.boundary_indices(0), 0.0),
            DirichletPart(mesh.boundary_indices(1), 0.0),
        ],
        bkd.zeros((npts,)),
        atol=1e-12,
        rtol=1e-12,
        maxiters=20,
    )


class TestHelmholtzSignConvention:
    """The screened sign, -lap(u) + k^2 u = f, pinned without a derived
    manufactured solution."""

    def test_wave_number_enters_the_residual_as_minus_k2_u(
        self, bkd: Backend[Array]
    ) -> None:
        k2 = 2.5
        screened = _helmholtz_1d(bkd, k2)
        poisson = _helmholtz_1d(bkd, 0.0)
        state = bkd.asarray(np.random.default_rng(1).normal(0.0, 1.0, screened.npts()))
        residual = screened.residual(state, 0.0)
        difference = residual - poisson.residual(state, 0.0)
        # Both residuals hold the Laplacian of a random state, large on
        # Chebyshev nodes; their difference keeps its rounding error.
        scale = bkd.to_float(bkd.max(bkd.abs(residual)))
        bkd.assert_allclose(difference, -k2 * state, rtol=1e-12, atol=1e-13 * scale)

    def test_comparison_principle(self, bkd: Backend[Array]) -> None:
        """Unit forcing, zero Dirichlet ends: a positive k^2 lowers the
        positive Poisson solution; a negative wave_number_sq (the
        unscreened form, below the first eigenvalue pi^2/4) raises it."""
        screened = _solve_unit_forcing(bkd, 1.0)
        poisson = _solve_unit_forcing(bkd, 0.0)
        unscreened = _solve_unit_forcing(bkd, -1.0)
        interior = slice(1, -1)
        assert bkd.to_bool(bkd.all(screened[interior] > 0.0))
        assert bkd.to_bool(bkd.all(screened[interior] < poisson[interior]))
        assert bkd.to_bool(bkd.all(unscreened[interior] > poisson[interior]))

    def test_interior_operator(self, bkd: Backend[Array]) -> None:
        """Helmholtz inherits ADR's interior declarations: constant
        coefficients give a constant Jacobian the checker accepts."""
        physics = _helmholtz_1d(bkd, 2.0)
        assert isinstance(physics, CollocationInteriorOperatorProtocol)
        rhs = CollocationRightHandSide(physics)
        jacobian = rhs.derivatives().jacobian
        assert jacobian is not None
        assert jacobian.affine() and jacobian.time_invariant()
        state = bkd.asarray(np.random.default_rng(2).normal(0.0, 1.0, physics.npts()))
        assert ODEResidualBundleChecker(rhs).check(state, 0.3).failures() == []
