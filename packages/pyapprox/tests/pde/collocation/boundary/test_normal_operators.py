"""Tests for normal operators."""

import numpy as np
import pytest

from pyapprox.pde.collocation.basis import (
    ChebyshevBasis1D,
    ChebyshevBasis2D,
)
from pyapprox.pde.collocation.boundary.normal_operators import (
    FluxNormalOperator,
    GradientNormalOperator,
    TractionNormalOperator,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    TransformedMesh2D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionFlux,
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.burgers import (
    BurgersPhysics1D,
    BurgersTotalFlux,
    BurgersViscousFlux,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.pde.collocation.physics.linear_elasticity import (
    LinearElasticityPhysics,
)
from pyapprox.pde.constitutive.advection import (
    ConservativeAdvection,
    NonConservativeAdvection,
)
from pyapprox.util.backends.protocols import Array, Backend


class TestGradientNormalOperator:
    """Base test class for GradientNormalOperator."""

    def test_1d_left_boundary(self, bkd: Backend[Array]) -> None:
        """Test grad(u).n at left boundary for u = x^2."""
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
        result = normal_op(state, 0.0)
        bkd.assert_allclose(result, bkd.array([2.0]), atol=1e-10)

    def test_1d_right_boundary(self, bkd: Backend[Array]) -> None:
        """Test grad(u).n at right boundary for u = x^2."""
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
        result = normal_op(state, 0.0)
        bkd.assert_allclose(result, bkd.array([2.0]), atol=1e-10)

    def test_jacobian_is_constant(self, bkd: Backend[Array]) -> None:
        """Test that gradient normal operator Jacobian is state-independent."""
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
        jac1 = normal_op.jacobian(state1, 0.0)
        jac2 = normal_op.jacobian(state2, 0.0)
        bkd.assert_allclose(jac1, jac2, atol=1e-14)

    def test_jacobian_consistency(self, bkd: Backend[Array]) -> None:
        """Test that jacobian @ state matches __call__(state)."""
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
        result = normal_op(state, 0.0)
        jac = normal_op.jacobian(state, 0.0)
        result_from_jac = jac @ state
        bkd.assert_allclose(result, result_from_jac, atol=1e-12)

    def test_2d_constant_normals(self, bkd: Backend[Array]) -> None:
        """Test 2D gradient normal operator with constant normals."""
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
        normal_op = GradientNormalOperator(bkd, left_idx, normals, [Dx, Dy])

        # u = x*y, grad = (y, x), grad.n at x=-1 = y*(-1) + x*0 = -y
        physical_pts = mesh.points()
        x_phys = physical_pts[0, :]
        y_phys = physical_pts[1, :]
        state = x_phys * y_phys

        result = normal_op(state, 0.0)
        # Expected: -y at left boundary points
        expected = -y_phys[left_idx]
        bkd.assert_allclose(result, expected, atol=1e-10)


class TestFluxNormalOperator:
    """Base test class for FluxNormalOperator."""

    def test_flux_with_adr_physics_1d(self, bkd: Backend[Array]) -> None:
        """Test FluxNormalOperator with ADR physics as flux provider."""
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
        result = flux_op(state, 0.0)
        bkd.assert_allclose(result, bkd.array([-4.0]), atol=1e-10)

    def test_linearity_follows_the_provider(self, bkd: Backend[Array]) -> None:
        """The operator declares itself linear exactly when its flux
        provider does: ADR and the Burgers viscous flux are linear, the
        Burgers total flux, carrying u^2/2, is not."""
        mesh = TransformedMesh1D(10, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        index, normals = bkd.array([0], dtype=int), bkd.array([[-1.0]])
        burgers = BurgersPhysics1D(basis, bkd, viscosity=0.1)
        for provider, linear in (
            (AdvectionDiffusionReaction(basis, bkd, diffusion=2.0), True),
            (BurgersViscousFlux(burgers), True),
            (BurgersTotalFlux(burgers), False),
        ):
            operator = FluxNormalOperator(bkd, index, normals, provider)
            assert operator.is_linear_in_state() is linear

    def test_flux_with_velocity(self, bkd: Backend[Array]) -> None:
        """Test FluxNormalOperator with conservative ADR physics including
        velocity, whose flux carries the advective part."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        umesh = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)

        diffusion = 1.0
        velocity = [bkd.full((npts,), 3.0)]  # v = 3
        physics = AdvectionDiffusionReaction(
            basis,
            bkd,
            diffusion=diffusion,
            velocity=velocity,
            advection_form=ConservativeAdvection(),
        )

        right_idx = umesh.boundary_indices(1)
        normals = bkd.array([[1.0]])

        flux_op = FluxNormalOperator(bkd, right_idx, normals, physics)

        # u = x, du/dx = 1
        # flux = -D*du/dx + v*u = -1 + 3*x
        # At x=1: flux = -1 + 3 = 2, flux.n = 2*(+1) = 2
        nodes = basis.nodes()
        state = bkd.copy(nodes)
        result = flux_op(state, 0.0)
        bkd.assert_allclose(result, bkd.array([2.0]), atol=1e-10)

    def test_injected_flux_on_non_conservative_physics(
        self, bkd: Backend[Array]
    ) -> None:
        """The non-conservative physics' own flux has no advective part
        (-D u' . n = -1 at x = 1 for u = x); the total flux of the same
        equation, injected as AdvectionDiffusionFlux with the conservative
        form, adds (v . n) u = 3, and its Jacobian matches the operator's
        action on the linear state."""
        npts = 10
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        physics = AdvectionDiffusionReaction(
            basis,
            bkd,
            diffusion=1.0,
            velocity=[bkd.full((npts,), 3.0)],
            advection_form=NonConservativeAdvection(),
        )
        right_idx = mesh.boundary_indices(1)
        normals = bkd.array([[1.0]])
        state = bkd.copy(basis.nodes())
        own = FluxNormalOperator(bkd, right_idx, normals, physics)
        bkd.assert_allclose(own(state, 0.0), bkd.array([-1.0]), atol=1e-10)
        total = FluxNormalOperator(
            bkd,
            right_idx,
            normals,
            AdvectionDiffusionFlux(physics, ConservativeAdvection()),
        )
        bkd.assert_allclose(total(state, 0.0), bkd.array([2.0]), atol=1e-10)
        bkd.assert_allclose(
            total.jacobian(state, 0.0) @ state, total(state, 0.0), atol=1e-10
        )

    def test_injected_flux_refuses_a_non_form(self, numpy_bkd: Backend[Array]) -> None:
        bkd = numpy_bkd
        basis = ChebyshevBasis1D(TransformedMesh1D(5, bkd), bkd)
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        with pytest.raises(TypeError, match="AdvectionFormProtocol"):
            AdvectionDiffusionFlux(physics, object())  # type: ignore[arg-type]

    def test_flux_jacobian_consistency(self, bkd: Backend[Array]) -> None:
        """Test flux operator Jacobian via finite differences."""
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
        jac = flux_op.jacobian(state, 0.0)

        # Finite difference check
        eps = 1e-7
        f0 = flux_op(state, 0.0)
        for j in range(npts):
            state_pert = bkd.copy(state)
            state_pert[j] = state_pert[j] + eps
            f1 = flux_op(state_pert, 0.0)
            fd_col = (f1 - f0) / eps
            bkd.assert_allclose(bkd.reshape(jac[:, j], fd_col.shape), fd_col, atol=1e-5)


class TestTractionNormalOperator:
    """Base test class for TractionNormalOperator."""

    def test_traction_at_boundary(self, bkd: Backend[Array]) -> None:
        """Test traction at left boundary for known polynomial displacement."""
        npts_x, npts_y = 8, 8
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        physics = LinearElasticityPhysics(basis, bkd, lamda=1.0, mu=1.0)

        # Left boundary (x = -1), normal = [-1, 0]
        left_idx = mesh.boundary_indices(0)
        nboundary = left_idx.shape[0]
        normals = bkd.zeros((nboundary, 2))
        normals = bkd.copy(normals)
        for i in range(nboundary):
            normals[i, 0] = -1.0

        traction_op_x = TractionNormalOperator(bkd, left_idx, normals, physics, 0)
        traction_op_y = TractionNormalOperator(bkd, left_idx, normals, physics, 1)

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
        result_x = traction_op_x(state, 0.0)
        result_y = traction_op_y(state, 0.0)

        expected_x = bkd.full((nboundary,), 7.0)
        expected_y = -y_phys[left_idx]
        bkd.assert_allclose(result_x, expected_x, atol=1e-10)
        bkd.assert_allclose(result_y, expected_y, atol=1e-10)

    def test_jacobian_consistency(self, bkd: Backend[Array]) -> None:
        """Test traction operator Jacobian via finite differences."""
        npts_x, npts_y = 6, 6
        npts = npts_x * npts_y
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        physics = LinearElasticityPhysics(basis, bkd, lamda=2.0, mu=0.5)

        left_idx = mesh.boundary_indices(0)
        nboundary = left_idx.shape[0]
        normals = bkd.zeros((nboundary, 2))
        normals = bkd.copy(normals)
        for i in range(nboundary):
            normals[i, 0] = -1.0

        for component in [0, 1]:
            traction_op = TractionNormalOperator(
                bkd, left_idx, normals, physics, component
            )

            np.random.seed(42)
            state = bkd.asarray(np.random.randn(2 * npts))
            jac = traction_op.jacobian(state, 0.0)

            # Finite difference check
            eps = 1e-7
            f0 = traction_op(state, 0.0)
            for j in range(2 * npts):
                state_pert = bkd.copy(state)
                state_pert[j] = state_pert[j] + eps
                f1 = traction_op(state_pert, 0.0)
                fd_col = (f1 - f0) / eps
                bkd.assert_allclose(
                    bkd.reshape(jac[:, j], fd_col.shape), fd_col, atol=1e-5
                )

    def test_jacobian_is_constant(self, bkd: Backend[Array]) -> None:
        """Test that traction Jacobian is state-independent."""
        npts_x, npts_y = 6, 6
        npts = npts_x * npts_y
        mesh = TransformedMesh2D(npts_x, npts_y, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        physics = LinearElasticityPhysics(basis, bkd, lamda=1.0, mu=1.0)

        left_idx = mesh.boundary_indices(0)
        nboundary = left_idx.shape[0]
        normals = bkd.zeros((nboundary, 2))
        normals = bkd.copy(normals)
        for i in range(nboundary):
            normals[i, 0] = -1.0

        traction_op = TractionNormalOperator(bkd, left_idx, normals, physics, 0)
        assert traction_op.is_linear_in_state()

        state1 = bkd.ones((2 * npts,))
        state2 = bkd.ones((2 * npts,)) * 5.0
        jac1 = traction_op.jacobian(state1, 0.0)
        jac2 = traction_op.jacobian(state2, 0.0)
        bkd.assert_allclose(jac1, jac2, atol=1e-14)

    def test_traction_follows_material_update(self, bkd: Backend[Array]) -> None:
        """The traction reads the physics' current material: after the
        Lame fields are set to c times their values, traction and
        Jacobian are c times their former values (linear in the
        material at fixed strain). A traction holding the material it
        was built with stays unchanged."""
        mesh = TransformedMesh2D(6, 6, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()
        mu = NodalField(bkd.full((npts,), 0.5))
        lamda = NodalField(bkd.full((npts,), 2.0))
        physics = LinearElasticityPhysics(basis, bkd, lamda=lamda, mu=mu)
        left_idx = mesh.boundary_indices(0)
        normals = mesh.boundary_normals(0)
        np.random.seed(3)
        state = bkd.asarray(np.random.randn(2 * npts))
        for component in (0, 1):
            op = TractionNormalOperator(bkd, left_idx, normals, physics, component)
            assert op.has_coefficient_dependence()
            t0, j0 = op(state, 0.0), op.jacobian(state, 0.0)
            mu.set_values(bkd.full((npts,), 1.5))
            lamda.set_values(bkd.full((npts,), 6.0))
            bkd.assert_allclose(op(state, 0.0), 3.0 * t0, rtol=1e-12)
            bkd.assert_allclose(op.jacobian(state, 0.0), 3.0 * j0, rtol=1e-12)
            mu.set_values(bkd.full((npts,), 0.5))
            lamda.set_values(bkd.full((npts,), 2.0))

    def test_non_stress_source_refused(self, numpy_bkd: Backend[Array]) -> None:
        bkd = numpy_bkd
        with pytest.raises(TypeError, match="CauchyStressSourceProtocol"):
            TractionNormalOperator(
                bkd,
                bkd.array([0], dtype=int),
                bkd.ones((1, 2)),
                object(),  # type: ignore[arg-type]
                0,
            )


# NumPy backend
# Torch backend
