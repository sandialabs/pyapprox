"""Tests for transient forward model classes."""

import math

import numpy as np
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.interface.functions.protocols import (
    FunctionProtocol,
)
from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import (
    gradient_robin_bc,
    zero_dirichlet_bc,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.field_maps.basis_expansion import (
    BasisExpansion,
)
from pyapprox.pde.models.collocation import create_collocation_model
from pyapprox.pde.models.collocation.transient import (
    TransientForwardModel,
)
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)


def _create_parameterized_transient_diffusion_problem(bkd, npts=15):
    """Create a parameterized transient diffusion problem for testing.

    Uses base AdvectionDiffusionReaction + DiffusionParameterization.

    Problem: du/dt = div(D(x) * grad(u)) with u(-1) = 0, u(1) = 0
    D(x) = D_base + p0 * phi0(x) + p1 * phi1(x)
    phi0(x) = 1 (constant), phi1(x) = x (linear)
    IC: u(x, 0) = sin(pi*x)

    Returns
    -------
    physics, parameterization, init_state, time_config
    """
    mesh = TransformedMesh1D(npts, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    mesh_obj = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
    nodes = basis.nodes()

    phi0 = bkd.ones((npts,))
    phi1 = nodes

    physics = AdvectionDiffusionReaction(
        basis,
        bkd,
        diffusion=2.0,
    )

    left_idx = mesh_obj.boundary_indices(0)
    right_idx = mesh_obj.boundary_indices(1)
    bc_left = zero_dirichlet_bc(bkd, left_idx)
    bc_right = zero_dirichlet_bc(bkd, right_idx)
    physics.set_boundary_conditions([bc_left, bc_right])

    fm = BasisExpansion(bkd, 2.0, [phi0, phi1])
    param = create_diffusion_parameterization(physics, bkd, fm)

    init_state = bkd.sin(math.pi * nodes)

    time_config = TimeIntegrationConfig(
        method="backward_euler",
        init_time=0.0,
        final_time=0.1,
        deltat=0.02,
        newton_tol=1e-10,
        newton_maxiter=20,
        lumped_mass=False,
        verbosity=0,
    )

    return physics, param, init_state, time_config


class TestTransientForwardModel:
    def test_call_matches_direct_solve(self, bkd):
        """__call__ with default functional matches solve_transient result."""
        physics, param, init_state, time_config = (
            _create_parameterized_transient_diffusion_problem(bkd)
        )

        forward_model = TransientForwardModel(
            physics, bkd, init_state, time_config, parameterization=param
        )

        param_1d = bkd.array([0.3, 0.1])
        samples = param_1d[:, None]

        # Direct solve
        param.apply(param_1d)
        model = create_collocation_model(physics, bkd, parameterization=param)
        solutions, times = model.solve_transient(init_state, time_config)

        # Forward model
        qoi = forward_model(samples)

        # Default functional returns all states at final time
        # atol covers the exactly-zero Dirichlet endpoints, whose
        # roundoff-level representations differ between the two paths.
        bkd.assert_allclose(
            qoi[:, 0], solutions[:, -1], rtol=1e-10, atol=1e-14
        )

    def test_call_multiple_samples(self, bkd):
        """__call__ handles multiple parameter samples correctly."""
        physics, param, init_state, time_config = (
            _create_parameterized_transient_diffusion_problem(bkd)
        )

        forward_model = TransientForwardModel(
            physics, bkd, init_state, time_config, parameterization=param
        )

        np.random.seed(42)
        nsamples = 3
        param_array = np.random.uniform(-0.1, 0.3, (2, nsamples))
        samples = bkd.asarray(param_array)

        result = forward_model(samples)
        assert result.shape[0] == forward_model.nqoi()
        assert result.shape[1] == nsamples

        # Verify each sample individually
        for ii in range(nsamples):
            single_result = forward_model(samples[:, ii : ii + 1])
            bkd.assert_allclose(result[:, ii : ii + 1], single_result, rtol=1e-10)

    def test_nvars_nqoi(self, bkd):
        """nvars and nqoi are correct for default functional."""
        physics, param, init_state, time_config = (
            _create_parameterized_transient_diffusion_problem(bkd)
        )

        forward_model = TransientForwardModel(
            physics, bkd, init_state, time_config, parameterization=param
        )

        assert forward_model.nvars() == 2
        assert forward_model.nqoi() == physics.nstates()

    def test_jacobian_vector_qoi_derivative_checker(self, bkd):
        """Jacobian with default functional (vector QoI) passes DerivativeChecker."""
        physics, param, init_state, time_config = (
            _create_parameterized_transient_diffusion_problem(bkd)
        )

        forward_model = TransientForwardModel(
            physics, bkd, init_state, time_config, parameterization=param
        )

        param_2d = bkd.array([0.3, 0.1])[:, None]

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=forward_model.nqoi(),
            nvars=forward_model.nvars(),
            fun=forward_model,
            jacobian=forward_model.derivatives().jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(param_2d, direction=None, relative=True)[0]
        assert float(bkd.min(errors) / bkd.max(errors)) <= 1e-5

    def test_jacobian_scalar_qoi_endpoint(self, bkd):
        """Jacobian with EndpointFunctional (scalar QoI) passes DerivativeChecker."""
        physics, param, init_state, time_config = (
            _create_parameterized_transient_diffusion_problem(bkd)
        )
        nstates = physics.nstates()
        nparams = param.nparams()

        state_idx = nstates // 2
        functional = EndpointFunctional(state_idx, nstates, nparams, bkd)

        forward_model = TransientForwardModel(
            physics,
            bkd,
            init_state,
            time_config,
            functional=functional,
            parameterization=param,
        )
        assert forward_model.nqoi() == 1

        param_2d = bkd.array([0.3, 0.1])[:, None]

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=forward_model.nqoi(),
            nvars=forward_model.nvars(),
            fun=forward_model,
            jacobian=forward_model.derivatives().jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(param_2d, direction=None, relative=True)[0]
        assert float(bkd.min(errors) / bkd.max(errors)) <= 1e-5

    def test_scalar_qoi_matches_vector_qoi_row(self, bkd):
        """Scalar QoI Jacobian matches corresponding row of vector QoI Jacobian."""
        physics, param, init_state, time_config = (
            _create_parameterized_transient_diffusion_problem(bkd)
        )
        nstates = physics.nstates()
        nparams = param.nparams()

        param_2d = bkd.array([0.3, 0.1])[:, None]

        # Vector QoI (default: all states at final time)
        fwd_vector = TransientForwardModel(
            physics, bkd, init_state, time_config, parameterization=param
        )
        jac_vector = fwd_vector.derivatives().jacobian(param_2d)

        # Scalar QoI for a specific state
        state_idx = nstates // 2
        functional = EndpointFunctional(state_idx, nstates, nparams, bkd)
        fwd_scalar = TransientForwardModel(
            physics,
            bkd,
            init_state,
            time_config,
            functional=functional,
            parameterization=param,
        )
        jac_scalar = fwd_scalar.derivatives().jacobian(param_2d)

        # Row state_idx of vector Jacobian should match scalar Jacobian
        bkd.assert_allclose(
            jac_vector[state_idx : state_idx + 1, :], jac_scalar, rtol=1e-8
        )

    def test_torch_autograd_jacobian(self, torch_bkd):
        """Torch autograd.functional.jacobian matches fwd.jacobian."""
        import torch

        bkd = torch_bkd
        physics, param, init_state, tc = (
            _create_parameterized_transient_diffusion_problem(bkd)
        )
        fwd = TransientForwardModel(
            physics, bkd, init_state, tc, parameterization=param
        )
        sample = torch.tensor([0.3, 0.1], dtype=torch.float64)

        def fwd_call(p):
            return fwd(p[:, None])[:, 0]

        autograd_jac = torch.autograd.functional.jacobian(fwd_call, sample)
        analytical_jac = fwd.derivatives().jacobian(sample[:, None])
        bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-6, atol=1e-12)

    def test_protocol_isinstance(self, bkd):
        """Protocol isinstance checks with parameterization path."""
        physics, param, init_state, tc = (
            _create_parameterized_transient_diffusion_problem(bkd)
        )
        fwd = TransientForwardModel(
            physics, bkd, init_state, tc, parameterization=param
        )
        assert isinstance(fwd, FunctionProtocol)
        assert isinstance(fwd, FunctionProtocol)
        assert fwd.derivatives().jacobian is not None


def _create_robin_transient_problem(bkd, npts=15):
    """Create a parameterized transient diffusion problem with Robin BCs.

    Uses base AdvectionDiffusionReaction + DiffusionParameterization.

    Same PDE as the Dirichlet test but with gradient Robin BCs on both
    boundaries:
      left:  1*u + 1*grad(u).n = 0
      right: 2*u + 1*grad(u).n = 0

    Asymmetric alpha tests that Robin coupling is handled correctly.

    Returns
    -------
    physics, parameterization, init_state, time_config
    """
    mesh = TransformedMesh1D(npts, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    nodes = basis.nodes()

    phi0 = bkd.ones((npts,))
    phi1 = nodes

    physics = AdvectionDiffusionReaction(
        basis,
        bkd,
        diffusion=2.0,
    )

    left_idx = mesh.boundary_indices(0)
    right_idx = mesh.boundary_indices(1)
    left_normals = mesh.boundary_normals(0)
    right_normals = mesh.boundary_normals(1)
    D = basis.derivative_matrix()

    bc_left = gradient_robin_bc(
        bkd,
        left_idx,
        left_normals,
        [D],
        1.0,
        1.0,
        0.0,
    )
    bc_right = gradient_robin_bc(
        bkd,
        right_idx,
        right_normals,
        [D],
        2.0,
        1.0,
        0.0,
    )
    physics.set_boundary_conditions([bc_left, bc_right])

    fm = BasisExpansion(bkd, 2.0, [phi0, phi1])
    param = create_diffusion_parameterization(physics, bkd, fm)

    init_state = bkd.sin(math.pi * nodes)

    time_config = TimeIntegrationConfig(
        method="backward_euler",
        init_time=0.0,
        final_time=0.1,
        deltat=0.02,
        newton_tol=1e-10,
        newton_maxiter=20,
        lumped_mass=False,
        verbosity=0,
    )

    return physics, param, init_state, time_config


def _create_mixed_bc_transient_problem(bkd, npts=15):
    """Create a parameterized transient diffusion problem with mixed BCs.

    Uses base AdvectionDiffusionReaction + DiffusionParameterization.

    Left: Robin (alpha=1, beta=1, g=0)
    Right: Dirichlet (u = 0)

    Returns
    -------
    physics, parameterization, init_state, time_config
    """
    mesh = TransformedMesh1D(npts, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    nodes = basis.nodes()

    phi0 = bkd.ones((npts,))
    phi1 = nodes

    physics = AdvectionDiffusionReaction(
        basis,
        bkd,
        diffusion=2.0,
    )

    left_idx = mesh.boundary_indices(0)
    right_idx = mesh.boundary_indices(1)
    left_normals = mesh.boundary_normals(0)
    D = basis.derivative_matrix()

    bc_left = gradient_robin_bc(
        bkd,
        left_idx,
        left_normals,
        [D],
        1.0,
        1.0,
        0.0,
    )
    bc_right = zero_dirichlet_bc(bkd, right_idx)
    physics.set_boundary_conditions([bc_left, bc_right])

    fm = BasisExpansion(bkd, 2.0, [phi0, phi1])
    param = create_diffusion_parameterization(physics, bkd, fm)

    init_state = bkd.sin(math.pi * nodes)

    time_config = TimeIntegrationConfig(
        method="backward_euler",
        init_time=0.0,
        final_time=0.1,
        deltat=0.02,
        newton_tol=1e-10,
        newton_maxiter=20,
        lumped_mass=False,
        verbosity=0,
    )

    return physics, param, init_state, time_config


class TestTransientRobinBC:
    """Tests for transient forward model with Robin BCs on both boundaries."""
    def test_vector_qoi_derivative_checker(self, bkd):
        """Forward sensitivity Jacobian passes DerivativeChecker with Robin BCs."""
        physics, param, init_state, tc = _create_robin_transient_problem(bkd)

        forward_model = TransientForwardModel(
            physics, bkd, init_state, tc, parameterization=param
        )

        param_2d = bkd.array([0.3, 0.1])[:, None]

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=forward_model.nqoi(),
            nvars=forward_model.nvars(),
            fun=forward_model,
            jacobian=forward_model.derivatives().jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(param_2d, direction=None, relative=True)[0]
        assert float(bkd.min(errors) / bkd.max(errors)) <= 1e-5

    def test_scalar_qoi_adjoint_derivative_checker(self, bkd):
        """Adjoint gradient passes DerivativeChecker with Robin BCs."""
        physics, param, init_state, tc = _create_robin_transient_problem(bkd)
        nstates = physics.nstates()
        nparams = param.nparams()

        state_idx = nstates // 2
        functional = EndpointFunctional(state_idx, nstates, nparams, bkd)

        forward_model = TransientForwardModel(
            physics,
            bkd,
            init_state,
            tc,
            functional=functional,
            parameterization=param,
        )
        assert forward_model.nqoi() == 1

        param_2d = bkd.array([0.3, 0.1])[:, None]

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=forward_model.nqoi(),
            nvars=forward_model.nvars(),
            fun=forward_model,
            jacobian=forward_model.derivatives().jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(param_2d, direction=None, relative=True)[0]
        assert float(bkd.min(errors) / bkd.max(errors)) <= 2e-5

    def test_scalar_matches_vector_row(self, bkd):
        """Scalar QoI adjoint Jacobian matches vector QoI forward sensitivity row."""
        physics, param, init_state, tc = _create_robin_transient_problem(bkd)
        nstates = physics.nstates()
        nparams = param.nparams()

        param_2d = bkd.array([0.3, 0.1])[:, None]

        fwd_vector = TransientForwardModel(
            physics, bkd, init_state, tc, parameterization=param
        )
        jac_vector = fwd_vector.derivatives().jacobian(param_2d)

        state_idx = nstates // 2
        functional = EndpointFunctional(state_idx, nstates, nparams, bkd)
        fwd_scalar = TransientForwardModel(
            physics,
            bkd,
            init_state,
            tc,
            functional=functional,
            parameterization=param,
        )
        jac_scalar = fwd_scalar.derivatives().jacobian(param_2d)

        bkd.assert_allclose(
            jac_vector[state_idx : state_idx + 1, :], jac_scalar, rtol=1e-8
        )


class TestTransientMixedBC:
    """Tests for transient forward model with mixed Robin + Dirichlet BCs."""
    def test_vector_qoi_derivative_checker(self, bkd):
        """Forward sensitivity Jacobian passes DerivativeChecker with mixed BCs."""
        physics, param, init_state, tc = _create_mixed_bc_transient_problem(bkd)

        forward_model = TransientForwardModel(
            physics, bkd, init_state, tc, parameterization=param
        )

        param_2d = bkd.array([0.3, 0.1])[:, None]

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=forward_model.nqoi(),
            nvars=forward_model.nvars(),
            fun=forward_model,
            jacobian=forward_model.derivatives().jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(param_2d, direction=None, relative=True)[0]
        assert float(bkd.min(errors) / bkd.max(errors)) <= 1e-5

    def test_scalar_qoi_adjoint_derivative_checker(self, bkd):
        """Adjoint gradient passes DerivativeChecker with mixed BCs."""
        physics, param, init_state, tc = _create_mixed_bc_transient_problem(bkd)
        nstates = physics.nstates()
        nparams = param.nparams()

        state_idx = nstates // 2
        functional = EndpointFunctional(state_idx, nstates, nparams, bkd)

        forward_model = TransientForwardModel(
            physics,
            bkd,
            init_state,
            tc,
            functional=functional,
            parameterization=param,
        )
        assert forward_model.nqoi() == 1

        param_2d = bkd.array([0.3, 0.1])[:, None]

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=forward_model.nqoi(),
            nvars=forward_model.nvars(),
            fun=forward_model,
            jacobian=forward_model.derivatives().jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(param_2d, direction=None, relative=True)[0]
        assert float(bkd.min(errors) / bkd.max(errors)) <= 1e-5

    def test_scalar_matches_vector_row(self, bkd):
        """Scalar QoI adjoint Jacobian matches vector QoI forward sensitivity row."""
        physics, param, init_state, tc = _create_mixed_bc_transient_problem(bkd)
        nstates = physics.nstates()
        nparams = param.nparams()

        param_2d = bkd.array([0.3, 0.1])[:, None]

        fwd_vector = TransientForwardModel(
            physics, bkd, init_state, tc, parameterization=param
        )
        jac_vector = fwd_vector.derivatives().jacobian(param_2d)

        state_idx = nstates // 2
        functional = EndpointFunctional(state_idx, nstates, nparams, bkd)
        fwd_scalar = TransientForwardModel(
            physics,
            bkd,
            init_state,
            tc,
            functional=functional,
            parameterization=param,
        )
        jac_scalar = fwd_scalar.derivatives().jacobian(param_2d)

        bkd.assert_allclose(
            jac_vector[state_idx : state_idx + 1, :], jac_scalar, rtol=1e-8
        )


class TestTransientForwardModelTiers:
    """Construction-time capability and the build-once/rebind pipeline."""

    def _scalar_model(self, bkd, npts=15, dirichlet_value=0.0):
        (
            physics,
            param,
            init_state,
            time_config,
        ) = _create_parameterized_transient_diffusion_problem(bkd, npts)
        if dirichlet_value != 0.0:
            from pyapprox.pde.collocation.boundary import (
                constant_dirichlet_bc,
            )

            mesh_obj = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
            physics.set_boundary_conditions(
                [
                    constant_dirichlet_bc(
                        bkd, mesh_obj.boundary_indices(0), dirichlet_value
                    ),
                    zero_dirichlet_bc(bkd, mesh_obj.boundary_indices(1)),
                ]
            )
        functional = EndpointFunctional(
            npts // 2, npts, param.nparams(), bkd
        )
        fwd = TransientForwardModel(
            physics,
            bkd,
            init_state,
            time_config,
            functional=functional,
            parameterization=param,
        )
        return fwd, param

    def test_second_order_tier_for_scalar_qoi(self, bkd):
        """Second-order bundle + HVP adapter + scalar HVP functional
        exposes Derivatives.second_order."""
        fwd, _ = self._scalar_model(bkd)
        derivs = fwd.derivatives()
        assert derivs.jacobian is not None
        assert derivs.hvp is not None

    def test_jacobian_and_hvp_derivative_checker(self, numpy_bkd):
        """DerivativeChecker validates the model's jacobian and hvp
        with the transient V-bottom convention, plus the exact
        symmetry identity at 1e-12."""
        bkd = numpy_bkd
        fwd, _ = self._scalar_model(bkd)
        sample = bkd.asarray(np.array([[0.3], [-0.2]]))

        checker = DerivativeChecker(fwd)
        errors = checker.check_derivatives(sample, verbosity=0)
        # Measured V-bottom 1.15e-7 (trajectory-FD floor of the
        # backward-Euler chain); ratio stays 5 orders below a plateau.
        assert bkd.to_float(bkd.min(errors[0])) <= 5e-7
        assert checker.error_ratio(errors[0]) <= 1e-5
        # Measured V-bottom 1.7e-6: the second-order adjoint through
        # the trajectory chain has a higher FD floor than the scalar
        # jacobian; the decay spans six decades (no plateau).
        assert bkd.to_float(bkd.min(errors[1])) <= 5e-6
        assert checker.error_ratio(errors[1]) <= 1e-5

        derivs = fwd.derivatives()
        assert derivs.hvp is not None
        vvec = bkd.asarray(np.array([[0.7], [0.4]]))
        uvec = bkd.asarray(np.array([[-0.5], [0.9]]))
        h_v = derivs.hvp(sample, vvec)
        h_u = derivs.hvp(sample, uvec)
        bkd.assert_allclose(
            bkd.sum(h_v * uvec), bkd.sum(h_u * vvec), rtol=1e-12
        )

    def test_bc_active_ic_hvp_derivative_checker(self, numpy_bkd):
        """A raw initial condition violating a NONZERO Dirichlet value
        must not corrupt the derivative paths: essential values are
        injected once at construction."""
        bkd = numpy_bkd
        fwd, _ = self._scalar_model(bkd, dirichlet_value=0.5)
        # The raw IC sin(pi x) is 0 at the left boundary, not 0.5.
        sample = bkd.asarray(np.array([[0.3], [-0.2]]))
        checker = DerivativeChecker(fwd)
        errors = checker.check_derivatives(sample, verbosity=0)
        # Measured V-bottom 1.15e-7 (trajectory-FD floor of the
        # backward-Euler chain); ratio stays 5 orders below a plateau.
        assert bkd.to_float(bkd.min(errors[0])) <= 5e-7
        assert checker.error_ratio(errors[0]) <= 1e-5
        # Measured V-bottom 1.7e-6: the second-order adjoint through
        # the trajectory chain has a higher FD floor than the scalar
        # jacobian; the decay spans six decades (no plateau).
        assert bkd.to_float(bkd.min(errors[1])) <= 5e-6
        assert checker.error_ratio(errors[1]) <= 1e-5

    def test_vector_rowwise_adjoint_matches_tlm(self, numpy_bkd):
        """When nparams > nqoi the all-states jacobian dispatches to
        the row-wise adjoint; it must equal the tangent-linear result
        computed directly."""
        bkd = numpy_bkd
        from pyapprox.ode.operator.forward_sensitivity import (
            solve_final_forward_sensitivity,
        )

        npts = 8
        (
            physics,
            _,
            init_state,
            time_config,
        ) = _create_parameterized_transient_diffusion_problem(bkd, npts)
        nodes = ChebyshevBasis1D(TransformedMesh1D(npts, bkd), bkd).nodes()
        modes = [bkd.ones((npts,))] + [
            bkd.cos(k * math.pi * nodes) * 0.1 for k in range(1, 10)
        ]
        fm = BasisExpansion(bkd, 2.0, modes)
        param = create_diffusion_parameterization(physics, bkd, fm)
        assert param.nparams() > npts
        fwd = TransientForwardModel(
            physics,
            bkd,
            init_state,
            time_config,
            parameterization=param,
        )
        rng = np.random.default_rng(61)
        sample = bkd.asarray(rng.normal(0.0, 0.05, (param.nparams(), 1)))
        jac_rowwise = fwd.derivatives().jacobian(sample)

        fwd_sols, times = fwd._forward_solve(sample)
        w_final = solve_final_forward_sensitivity(
            fwd.last_integrator().time_residual(), fwd_sols, times, bkd
        )
        bkd.assert_allclose(jac_rowwise, w_final, rtol=1e-9, atol=1e-12)

    def test_adapter_identity_stable_across_samples(self, bkd):
        """The pipeline is built once; evaluations rebind parameters
        without reconstructing the adapter."""
        fwd, _ = self._scalar_model(bkd)
        adapter_before = fwd.adapter()
        samples = bkd.asarray(
            np.array([[0.3, -0.1, 0.2], [-0.2, 0.15, 0.05]])
        )
        fwd(samples)
        assert fwd.adapter() is adapter_before

    def test_inexact_wrapper_smoke(self, numpy_bkd):
        """The modernized model stays FunctionProtocol-consumable by
        the OUU InexactWrapper."""
        bkd = numpy_bkd
        from pyapprox.optimization.minimize.inexact.fixed import (
            FixedSampleStrategy,
        )
        from pyapprox.optimization.minimize.inexact.wrapper import (
            InexactWrapper,
        )
        from pyapprox.risk import SampleAverageMean

        fwd, _ = self._scalar_model(bkd)
        quad_samples = bkd.asarray(np.array([[-0.2, 0.0, 0.2]]))
        quad_weights = bkd.asarray(np.array([1.0 / 4, 1.0 / 2, 1.0 / 4]))
        wrapper = InexactWrapper(
            model=fwd,
            stat=SampleAverageMean(bkd),
            strategy=FixedSampleStrategy(quad_samples, quad_weights, bkd),
            design_indices=[1],
            bkd=bkd,
        )
        design_sample = bkd.asarray(np.array([[0.1]]))
        value = wrapper(design_sample)
        assert value.shape[0] == 1
        assert math.isfinite(bkd.to_float(value[0, 0]))


class TestTransientPolarTripleAgreement:
    """Transformed-domain check: adjoint == TLM == FD on a polar mesh."""

    def test_adjoint_tlm_fd_agree(self, numpy_bkd):
        bkd = numpy_bkd
        from pyapprox.ode.functionals.all_states_endpoint import (
            AllStatesEndpointFunctional,
        )
        from pyapprox.pde.collocation.basis import ChebyshevBasis2D
        from pyapprox.pde.collocation.mesh import TransformedMesh2D
        from pyapprox.pde.collocation.mesh.transforms import PolarTransform

        npts_1d = 6
        transform = PolarTransform(
            (1.0, 2.0), (-math.pi / 2, math.pi / 2), bkd
        )
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd, transform)
        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=2.0)
        bcs = []
        for bndry in range(4):
            bcs.append(
                zero_dirichlet_bc(bkd, mesh.boundary_indices(bndry))
            )
        physics.set_boundary_conditions(bcs)

        pts = bkd.to_numpy(mesh.points())
        modes = [
            bkd.ones((npts,)),
            bkd.asarray(0.3 * np.sin(pts[0] + pts[1])),
        ]
        fm = BasisExpansion(bkd, 2.0, modes)
        param = create_diffusion_parameterization(physics, bkd, fm)

        interior = npts // 2 + npts_1d // 2
        time_config = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=0.05,
            deltat=0.0125,
            newton_tol=1e-11,
            newton_maxiter=20,
            lumped_mass=False,
            verbosity=0,
        )
        init_state = bkd.asarray(
            np.sin(math.pi * (pts[0] ** 2 + pts[1] ** 2) / 4.0)
        )
        scalar = TransientForwardModel(
            physics,
            bkd,
            init_state,
            time_config,
            functional=EndpointFunctional(interior, npts, 2, bkd),
            parameterization=param,
        )
        sample = bkd.asarray(np.array([[0.2], [-0.1]]))

        # (a) adjoint
        jac_adjoint = scalar.derivatives().jacobian(sample)

        # (b) tangent-linear: the all-states model dispatches to the
        # TLM (nparams = 2 <= nqoi); extract the same QoI row.
        vector = TransientForwardModel(
            physics,
            bkd,
            init_state,
            time_config,
            functional=AllStatesEndpointFunctional(npts, 2, bkd),
            parameterization=param,
        )
        jac_tlm = vector.derivatives().jacobian(sample)
        bkd.assert_allclose(
            jac_adjoint,
            jac_tlm[interior : interior + 1, :],
            rtol=1e-9,
            atol=1e-13,
        )

        # (c) finite differences
        checker = DerivativeChecker(scalar)
        errors = checker.check_derivatives(sample, verbosity=0)
        # Measured V-bottom 1.15e-7 (trajectory-FD floor of the
        # backward-Euler chain); ratio stays 5 orders below a plateau.
        assert bkd.to_float(bkd.min(errors[0])) <= 5e-7
        assert checker.error_ratio(errors[0]) <= 1e-5
