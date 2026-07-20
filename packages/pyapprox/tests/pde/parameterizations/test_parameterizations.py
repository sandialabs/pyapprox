"""Tests for physics parameterization implementations."""

import math

import pytest
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.optimization.implicitfunction.functionals.weighted_sum import (
    WeightedSumFunctional,
)
from pyapprox.optimization.implicitfunction.operator.check_derivatives import (
    ImplicitFunctionDerivativeChecker,
)
from pyapprox.optimization.implicitfunction.operator.operator_with_hvp import (
    AdjointOperatorWithJacobianAndHVP,
)
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.field_maps.basis_expansion import (
    BasisExpansion,
)
from pyapprox.pde.field_maps.kle_factory import (
    create_lognormal_kle_field_map,
)
from pyapprox.pde.field_maps.scalar import (
    ScalarAmplitude,
)
from pyapprox.pde.models.collocation.steady import (
    SteadyForwardModel,
)
from pyapprox.pde.parameterizations.composite import (
    CompositeParameterization,
)
from pyapprox.pde.parameterizations.derivatives import (
    ParamDerivatives,
)
from pyapprox.pde.parameterizations.diffusion import (
    DiffusionParameterization,
    create_diffusion_parameterization,
)
from pyapprox.pde.parameterizations.forcing import (
    ForcingParameterization,
)
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.pde.parameterizations.reaction import (
    ReactionParameterization,
)

from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    create_uniform_mesh_1d,
)


def _create_diffusion_physics_and_basis(bkd, npts=20):
    """Create base ADR physics with BCs for testing parameterizations."""
    mesh = TransformedMesh1D(npts, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    mesh_obj = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
    nodes = basis.nodes()

    def forcing(t):
        return (math.pi**2) * bkd.sin(math.pi * nodes)

    physics = AdvectionDiffusionReaction(
        basis,
        bkd,
        diffusion=1.0,
        forcing=forcing,
    )

    left_idx = mesh_obj.boundary_indices(0)
    right_idx = mesh_obj.boundary_indices(1)
    bc_left = zero_dirichlet_bc(bkd, left_idx)
    bc_right = zero_dirichlet_bc(bkd, right_idx)
    physics.set_boundary_conditions([bc_left, bc_right])

    return physics, basis, nodes


class _EvalOnlyParam:
    """Parameterization with no derivative capability."""

    def __init__(self, physics):
        self._physics = physics

    def nparams(self) -> int:
        return 1

    def physics(self):
        return self._physics

    def apply(self, params_1d):
        pass

    def param_derivatives(self):
        return ParamDerivatives.none()


class _MockSecondOrderParam:
    """Second-order-capable mock for the toy residual

        R(y, p) = scale * (sum(p)^2 / 2) * y

    so every derivative below is the analytic consequence of one
    consistent model (the IC does not depend on p):

    - dR/dp_j              = scale * sum(p) * y
    - lam^T d2R/dp2 v      = scale * (lam . y) * sum(v) * ones(np)
    - (d2R/dydp . v)^T lam = scale * sum(p) * sum(v) * lam (state-shaped)
    - lam^T d2R/dpdy w     = scale * sum(p) * (lam . w) * ones(np)
    """

    def __init__(self, physics, bkd, nparams, nstates, scale):
        self._physics = physics
        self._bkd = bkd
        self._np = nparams
        self._nstates = nstates
        self._scale = scale
        self._derivs = ParamDerivatives.second_order(
            self._param_jacobian,
            self._initial_param_jacobian,
            self._param_param_hvp,
            self._state_param_hvp,
            self._param_state_hvp,
        )

    def nparams(self) -> int:
        return self._np

    def physics(self):
        return self._physics

    def apply(self, params_1d):
        pass

    def param_derivatives(self):
        return self._derivs

    def _param_jacobian(self, state, time, params_1d):
        return (
            self._scale
            * self._bkd.sum(params_1d)
            * state[:, None]
            * self._bkd.ones((1, self._np))
        )

    def _initial_param_jacobian(self, params_1d):
        return self._bkd.zeros((self._nstates, self._np))

    def _param_param_hvp(self, state, time, params_1d, adj_state, vvec):
        return (
            self._scale
            * self._bkd.sum(adj_state * state)
            * self._bkd.sum(vvec)
            * self._bkd.ones((self._np,))
        )

    def _state_param_hvp(self, state, time, params_1d, adj_state, vvec):
        return (
            self._scale
            * self._bkd.sum(params_1d)
            * self._bkd.sum(vvec)
            * adj_state
        )

    def _param_state_hvp(self, state, time, params_1d, adj_state, wvec):
        return (
            self._scale
            * self._bkd.sum(params_1d)
            * self._bkd.sum(adj_state * wvec)
            * self._bkd.ones((self._np,))
        )


class _ToyCompositeStateEquation:
    """Adapts the composite bundle to the state-equation checker protocol.

    Implements ParameterizedStateEquationWithJacobianAndHVPProtocol (2D
    column convention) for the toy residual

        R(y, p) = kappa(p) y - c,   kappa(p) = sum_i s_i (sum(p_i)^2 / 2)

    The residual, state jacobian, and solve are the analytic ground truth;
    ALL parameter derivatives delegate to the composite's ParamDerivatives
    bundle under test, so ImplicitFunctionDerivativeChecker's FD sweeps
    validate the bundle callables (and their composite assembly) against
    the residual itself. The bundle is narrowed ONCE at construction
    (this adapter is the HVP tier), per the D8/D10 consumer rule.
    """

    def __init__(self, comp, kappa, cvec, bkd):
        self._comp = comp
        self._kappa = kappa  # callable p_1d -> scalar
        self._cvec = cvec  # (nstates,)
        self._bkd = bkd
        derivs = comp.param_derivatives()
        if (
            derivs.param_jacobian is None
            or derivs.param_param_hvp is None
            or derivs.state_param_hvp is None
            or derivs.param_state_hvp is None
        ):
            raise TypeError(
                "composite must provide param_jacobian and all three HVPs"
            )
        self._param_jacobian = derivs.param_jacobian
        self._param_param_hvp = derivs.param_param_hvp
        self._state_param_hvp = derivs.state_param_hvp
        self._param_state_hvp = derivs.param_state_hvp

    def bkd(self):
        return self._bkd

    def nparams(self) -> int:
        return self._comp.nparams()

    def nstates(self) -> int:
        return self._cvec.shape[0]

    def __call__(self, state, param):
        return self._kappa(param[:, 0]) * state - self._cvec[:, None]

    def solve(self, init_state, param):
        return self._cvec[:, None] / self._kappa(param[:, 0])

    def state_jacobian(self, state, param):
        return self._kappa(param[:, 0]) * self._bkd.eye(self.nstates())

    def param_jacobian(self, state, param):
        return self._param_jacobian(state[:, 0], 0.0, param[:, 0])

    def state_state_hvp(self, state, param, adj_state, wvec):
        # R is linear in y
        return self._bkd.zeros((self.nstates(), 1))

    def param_param_hvp(self, state, param, adj_state, vvec):
        return self._param_param_hvp(
            state[:, 0], 0.0, param[:, 0], adj_state[:, 0], vvec[:, 0]
        )[:, None]

    def state_param_hvp(self, state, param, adj_state, vvec):
        return self._state_param_hvp(
            state[:, 0], 0.0, param[:, 0], adj_state[:, 0], vvec[:, 0]
        )[:, None]

    def param_state_hvp(self, state, param, adj_state, wvec):
        return self._param_state_hvp(
            state[:, 0], 0.0, param[:, 0], adj_state[:, 0], wvec[:, 0]
        )[:, None]


class TestParameterizations:
    def test_diffusion_isinstance(self, bkd) -> None:
        """DiffusionParameterization satisfies ParameterizationProtocol."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 1.0, [phi0])
        dp = create_diffusion_parameterization(physics, bkd, basis, fm)
        assert isinstance(dp, ParameterizationProtocol)
        derivs = dp.param_derivatives()
        assert derivs.param_jacobian is not None
        assert derivs.initial_param_jacobian is not None
        assert derivs.bc_flux_param_sensitivity is not None
        assert derivs.param_param_hvp is None

    def test_diffusion_init_type_error(self, bkd) -> None:
        """DiffusionParameterization raises TypeError for non-FieldMap."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        with pytest.raises(TypeError):
            DiffusionParameterization(physics, "not_a_field_map", [], bkd)

    def test_diffusion_apply_sets_field(self, bkd) -> None:
        """DiffusionParameterization.apply sets diffusion on physics."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        phi1 = nodes
        fm = BasisExpansion(bkd, 1.0, [phi0, phi1])
        dp = create_diffusion_parameterization(physics, bkd, basis, fm)

        params = bkd.array([0.5, -0.3])
        dp.apply(params)

        # After apply, diffusion should be 1.0 + 0.5*1 + (-0.3)*nodes
        expected_diff = bkd.full((npts,), 1.0) + 0.5 * phi0 + (-0.3) * phi1
        actual_diff = physics._get_diffusion(0.0)
        bkd.assert_allclose(actual_diff, expected_diff, rtol=1e-12)

    def test_diffusion_param_jacobian_fd(self, bkd) -> None:
        """DiffusionParameterization.param_jacobian matches FD."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        num_kle_terms = 2
        mesh_coords = ((nodes + 1.0) / 2.0)[None, :]
        mean_log = bkd.zeros((npts,))
        fm = create_lognormal_kle_field_map(
            mesh_coords,
            mean_log,
            bkd,
            num_kle_terms=num_kle_terms,
            sigma=0.3,
        )
        dp = create_diffusion_parameterization(physics, bkd, basis, fm)

        # Get a non-trivial state by solving with some parameters
        state = bkd.sin(math.pi * nodes)
        time = 0.0

        def residual_of_params(samples):
            results = []
            for i in range(samples.shape[1]):
                p = samples[:, i]
                dp.apply(p)
                res = physics.residual(state, time)
                results.append(res)
            return bkd.stack(results, axis=1)

        def jac_of_params(sample):
            p = sample[:, 0]
            dp.apply(p)
            return dp.param_jacobian(state, time, p)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=npts,
            nvars=dp.nparams(),
            fun=residual_of_params,
            jacobian=jac_of_params,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        params = bkd.array([0.1, -0.1])[:, None]
        errors = checker.check_derivatives(params)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5

    def test_diffusion_param_jacobian_autograd(self, torch_bkd) -> None:
        """Torch autograd matches DiffusionParameterization.param_jacobian."""
        import torch

        bkd = torch_bkd
        # Use physics WITHOUT BCs so residual is pure PDE operator
        npts = 20
        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        def forcing(t):
            return (math.pi**2) * bkd.sin(math.pi * nodes)

        physics = AdvectionDiffusionReaction(
            basis,
            bkd,
            diffusion=1.0,
            forcing=forcing,
        )

        phi0 = bkd.ones((npts,))
        phi1 = nodes
        fm = BasisExpansion(bkd, 1.0, [phi0, phi1])
        dp = create_diffusion_parameterization(physics, bkd, basis, fm)

        # Use a state that's non-zero everywhere to avoid near-zero issues
        state = bkd.cos(0.5 * math.pi * nodes) + 1.0
        time = 0.0

        params = torch.tensor([0.3, -0.1], dtype=torch.float64)

        def torch_residual(p):
            dp.apply(p)
            return physics.residual(state, time)

        autograd_jac = torch.autograd.functional.jacobian(torch_residual, params)
        dp.apply(params)
        analytical_jac = dp.param_jacobian(state, time, params)
        bkd.assert_allclose(analytical_jac, autograd_jac, atol=1e-12)

    def test_diffusion_initial_param_jacobian_zeros(self, bkd) -> None:
        """DiffusionParameterization.initial_param_jacobian returns zeros."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 1.0, [phi0])
        dp = create_diffusion_parameterization(physics, bkd, basis, fm)
        params = bkd.array([0.5])
        result = dp.initial_param_jacobian(params)
        expected = bkd.zeros((npts, 1))
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_forcing_apply_and_jacobian(self, bkd) -> None:
        """ForcingParameterization.apply and param_jacobian work correctly."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        base_forcing = bkd.sin(math.pi * nodes)
        fm = ScalarAmplitude(bkd, base_forcing)
        fp = ForcingParameterization(physics, fm, bkd)

        assert isinstance(fp, ParameterizationProtocol)
        assert fp.nparams() == 1
        assert fp.param_derivatives().param_jacobian is not None
        assert fp.param_derivatives().bc_flux_param_sensitivity is None

        state = bkd.sin(math.pi * nodes)
        time = 0.0

        def residual_of_params(samples):
            results = []
            for i in range(samples.shape[1]):
                p = samples[:, i]
                fp.apply(p)
                res = physics.residual(state, time)
                results.append(res)
            return bkd.stack(results, axis=1)

        def jac_of_params(sample):
            p = sample[:, 0]
            fp.apply(p)
            return fp.param_jacobian(state, time, p)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=npts,
            nvars=1,
            fun=residual_of_params,
            jacobian=jac_of_params,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        params = bkd.array([2.0])[:, None]
        errors = checker.check_derivatives(params)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5

    def test_reaction_apply_and_jacobian(self, bkd) -> None:
        """ReactionParameterization.apply and param_jacobian work correctly."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 0.0, [phi0])
        rp = ReactionParameterization(physics, fm, bkd)

        assert isinstance(rp, ParameterizationProtocol)
        assert rp.nparams() == 1
        assert rp.param_derivatives().param_jacobian is not None
        assert rp.param_derivatives().bc_flux_param_sensitivity is None

        state = bkd.sin(math.pi * nodes)
        time = 0.0

        def residual_of_params(samples):
            results = []
            for i in range(samples.shape[1]):
                p = samples[:, i]
                rp.apply(p)
                res = physics.residual(state, time)
                results.append(res)
            return bkd.stack(results, axis=1)

        def jac_of_params(sample):
            p = sample[:, 0]
            rp.apply(p)
            return rp.param_jacobian(state, time, p)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=npts,
            nvars=1,
            fun=residual_of_params,
            jacobian=jac_of_params,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        params = bkd.array([-0.5])[:, None]
        errors = checker.check_derivatives(params)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5

    def test_composite_isinstance(self, bkd) -> None:
        """CompositeParameterization satisfies ParameterizationProtocol."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm_d = BasisExpansion(bkd, 1.0, [phi0])
        dp = create_diffusion_parameterization(physics, bkd, basis, fm_d)

        base_forcing = bkd.sin(math.pi * nodes)
        fm_f = ScalarAmplitude(bkd, base_forcing)
        fp = ForcingParameterization(physics, fm_f, bkd)

        comp = CompositeParameterization([dp, fp], bkd)
        assert isinstance(comp, ParameterizationProtocol)

    def test_composite_init_type_error(self, bkd) -> None:
        """CompositeParameterization raises TypeError for non-protocol part."""
        with pytest.raises(TypeError):
            CompositeParameterization(["not_a_param"], bkd)

    def test_composite_nparams(self, bkd) -> None:
        """CompositeParameterization.nparams is sum of parts."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        phi1 = nodes
        fm_d = BasisExpansion(bkd, 1.0, [phi0, phi1])
        dp = create_diffusion_parameterization(physics, bkd, basis, fm_d)

        base_forcing = bkd.sin(math.pi * nodes)
        fm_f = ScalarAmplitude(bkd, base_forcing)
        fp = ForcingParameterization(physics, fm_f, bkd)

        comp = CompositeParameterization([dp, fp], bkd)
        assert comp.nparams() == 3

    def test_composite_param_jacobian_fd(self, bkd) -> None:
        """CompositeParameterization.param_jacobian matches FD."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        num_kle_terms = 2
        mesh_coords = ((nodes + 1.0) / 2.0)[None, :]
        mean_log = bkd.zeros((npts,))
        fm_d = create_lognormal_kle_field_map(
            mesh_coords,
            mean_log,
            bkd,
            num_kle_terms=num_kle_terms,
            sigma=0.3,
        )
        dp = create_diffusion_parameterization(physics, bkd, basis, fm_d)

        base_forcing = bkd.sin(math.pi * nodes)
        fm_f = ScalarAmplitude(bkd, base_forcing)
        fp = ForcingParameterization(physics, fm_f, bkd)

        comp = CompositeParameterization([dp, fp], bkd)
        state = bkd.sin(math.pi * nodes)
        time = 0.0

        def residual_of_params(samples):
            results = []
            for i in range(samples.shape[1]):
                p = samples[:, i]
                comp.apply(p)
                res = physics.residual(state, time)
                results.append(res)
            return bkd.stack(results, axis=1)

        comp_param_jac = comp.param_derivatives().param_jacobian
        assert comp_param_jac is not None

        def jac_of_params(sample):
            p = sample[:, 0]
            comp.apply(p)
            return comp_param_jac(state, time, p)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=npts,
            nvars=comp.nparams(),
            fun=residual_of_params,
            jacobian=jac_of_params,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        params = bkd.array([0.1, -0.1, 2.0])[:, None]
        errors = checker.check_derivatives(params)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5

    def test_composite_initial_param_jacobian(self, bkd) -> None:
        """CompositeParameterization.initial_param_jacobian block structure."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm_d = BasisExpansion(bkd, 1.0, [phi0])
        dp = create_diffusion_parameterization(physics, bkd, basis, fm_d)

        base_forcing = bkd.ones((npts,))
        fm_f = ScalarAmplitude(bkd, base_forcing)
        fp = ForcingParameterization(physics, fm_f, bkd)

        comp = CompositeParameterization([dp, fp], bkd)
        params = bkd.array([0.5, 1.0])
        init_jac_fn = comp.param_derivatives().initial_param_jacobian
        assert init_jac_fn is not None
        result = init_jac_fn(params)
        expected = bkd.zeros((npts, 2))
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_composite_bundle_all_differentiable(self, bkd) -> None:
        """Composite of first-order parts exposes param_jacobian in bundle."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 1.0, [phi0])
        dp = create_diffusion_parameterization(physics, bkd, basis, fm)
        comp = CompositeParameterization([dp], bkd)
        derivs = comp.param_derivatives()
        assert derivs.param_jacobian is not None
        assert derivs.initial_param_jacobian is not None
        # no part provides HVPs -> composite bundle has none
        assert derivs.param_param_hvp is None
        assert derivs.state_param_hvp is None
        assert derivs.param_state_hvp is None

    def test_composite_bundle_eval_only(self, bkd) -> None:
        """Composite with eval-only part declares no capability."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        comp = CompositeParameterization([_EvalOnlyParam(physics)], bkd)
        assert comp.param_derivatives().param_jacobian is None
        assert comp.param_derivatives().initial_param_jacobian is None

    def test_composite_append_removes_param_jacobian(self, bkd) -> None:
        """Appending non-differentiable part rebuilds bundle without it."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 1.0, [phi0])
        dp = create_diffusion_parameterization(physics, bkd, basis, fm)

        comp = CompositeParameterization([dp], bkd)
        assert comp.param_derivatives().param_jacobian is not None

        comp.append(_EvalOnlyParam(physics))
        assert comp.param_derivatives().param_jacobian is None

    def test_composite_unavailable_capability_raises(self, bkd) -> None:
        """The guarded private impl raises when capability is absent."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        comp = CompositeParameterization([_EvalOnlyParam(physics)], bkd)
        state = bkd.zeros((basis.npts(),))
        params = bkd.array([0.5])
        with pytest.raises(RuntimeError, match="param_jacobian is unavailable"):
            comp._param_jacobian(state, 0.0, params)

    def test_composite_second_order_bundle(self, bkd) -> None:
        """Composite of second-order parts exposes all three HVPs."""
        physics = object()
        p1 = _MockSecondOrderParam(physics, bkd, 2, 5, 2.0)
        p2 = _MockSecondOrderParam(physics, bkd, 3, 5, -1.5)
        comp = CompositeParameterization([p1, p2], bkd)
        derivs = comp.param_derivatives()
        assert derivs.param_param_hvp is not None
        assert derivs.state_param_hvp is not None
        assert derivs.param_state_hvp is not None

    def test_composite_state_param_hvp_state_shaped(self, bkd) -> None:
        """Regression (D9.4): state_param_hvp sums part results to (nstates,).

        The old block assembly returned a (total_nparams,) vector, which
        the HVP consumer adds into an (nstates,)-shaped RHS.
        """
        nstates = 5
        s1, s2 = 2.0, -1.5
        physics = object()
        p1 = _MockSecondOrderParam(physics, bkd, 2, nstates, s1)
        p2 = _MockSecondOrderParam(physics, bkd, 3, nstates, s2)
        comp = CompositeParameterization([p1, p2], bkd)
        fn = comp.param_derivatives().state_param_hvp
        assert fn is not None

        state = bkd.array([0.4, -1.0, 2.0, 0.7, -0.2])
        adj = bkd.array([1.0, -0.5, 0.25, 2.0, -1.0])
        params = bkd.array([0.3, -0.1, 0.8, 0.5, -0.4])
        vvec = bkd.array([1.0, 2.0, -1.0, 0.5, 3.0])

        result = fn(state, 0.0, params, adj, vvec)
        assert result.shape == (nstates,)
        expected = (
            s1 * bkd.sum(params[:2]) * bkd.sum(vvec[:2]) * adj
            + s2 * bkd.sum(params[2:]) * bkd.sum(vvec[2:]) * adj
        )
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_composite_param_shaped_hvp_blocks(self, bkd) -> None:
        """param_param_hvp / param_state_hvp assemble per-part blocks."""
        nstates = 5
        s1, s2 = 2.0, -1.5
        physics = object()
        p1 = _MockSecondOrderParam(physics, bkd, 2, nstates, s1)
        p2 = _MockSecondOrderParam(physics, bkd, 3, nstates, s2)
        comp = CompositeParameterization([p1, p2], bkd)
        derivs = comp.param_derivatives()
        assert derivs.param_param_hvp is not None
        assert derivs.param_state_hvp is not None

        state = bkd.array([0.4, -1.0, 2.0, 0.7, -0.2])
        adj = bkd.array([1.0, -0.5, 0.25, 2.0, -1.0])
        params = bkd.array([0.3, -0.1, 0.8, 0.5, -0.4])
        vvec = bkd.array([1.0, 2.0, -1.0, 0.5, 3.0])
        wvec = bkd.array([-0.3, 1.2, 0.1, -0.8, 0.6])

        pp = derivs.param_param_hvp(state, 0.0, params, adj, vvec)
        assert pp.shape == (5,)
        lam_dot_y = bkd.sum(adj * state)
        expected_pp = bkd.concatenate(
            [
                s1 * lam_dot_y * bkd.sum(vvec[:2]) * bkd.ones((2,)),
                s2 * lam_dot_y * bkd.sum(vvec[2:]) * bkd.ones((3,)),
            ],
            axis=0,
        )
        bkd.assert_allclose(pp, expected_pp, rtol=1e-12)

        ps = derivs.param_state_hvp(state, 0.0, params, adj, wvec)
        assert ps.shape == (5,)
        lam_dot_w = bkd.sum(adj * wvec)
        expected_ps = bkd.concatenate(
            [
                s1 * bkd.sum(params[:2]) * lam_dot_w * bkd.ones((2,)),
                s2 * bkd.sum(params[2:]) * lam_dot_w * bkd.ones((3,)),
            ],
            axis=0,
        )
        bkd.assert_allclose(ps, expected_ps, rtol=1e-12)

    def test_composite_derivatives_via_implicit_function_checker(self, bkd) -> None:
        """Orchestrated FD validation of the whole composite bundle.

        ImplicitFunctionDerivativeChecker runs all 14 checks (state/param
        jacobians, all four state-equation HVP blocks, functional
        derivatives, and the assembled adjoint gradient and HVP) against
        finite differences of the mocks' toy residual — so an algebra
        error in the mock formulas or in the composite's block assembly
        cannot cancel against matching test expectations.
        """
        nstates = 5
        s1, s2 = 2.0, -1.5
        physics = object()
        p1 = _MockSecondOrderParam(physics, bkd, 2, nstates, s1)
        p2 = _MockSecondOrderParam(physics, bkd, 3, nstates, s2)
        comp = CompositeParameterization([p1, p2], bkd)

        def kappa(p_1d):
            return (
                s1 * bkd.sum(p_1d[:2]) ** 2 / 2.0
                + s2 * bkd.sum(p_1d[2:]) ** 2 / 2.0
            )

        cvec = bkd.array([1.0, -0.5, 0.25, 2.0, -1.0])
        state_eq = _ToyCompositeStateEquation(comp, kappa, cvec, bkd)
        functional = WeightedSumFunctional(
            bkd.ones((nstates, 1)), comp.nparams(), bkd
        )
        adjoint_op = AdjointOperatorWithJacobianAndHVP(state_eq, functional)
        checker = ImplicitFunctionDerivativeChecker(adjoint_op)
        tols = checker.get_derivative_tolerances(1e-6)
        param = bkd.array([0.3, -0.1, 0.8, 0.5, -0.4])[:, None]
        init_state = bkd.zeros((nstates, 1))
        checker.check_derivatives(init_state, param, tols)

    def test_composite_append_type_error(self, bkd) -> None:
        """CompositeParameterization.append raises TypeError for non-protocol."""
        comp = CompositeParameterization([], bkd)
        with pytest.raises(TypeError):
            comp.append("not_a_param")

    def test_forcing_init_type_error(self, bkd) -> None:
        """ForcingParameterization raises TypeError for non-FieldMap."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        with pytest.raises(TypeError):
            ForcingParameterization(physics, "not_a_field_map", bkd)

    def test_reaction_init_type_error(self, bkd) -> None:
        """ReactionParameterization raises TypeError for non-FieldMap."""
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        with pytest.raises(TypeError):
            ReactionParameterization(physics, "not_a_field_map", bkd)


class TestCompositeWithSteadyForwardModel:
    """Integration test: CompositeParameterization with SteadyForwardModel."""
    def _create_composite_forward_model(self, bkd) :
        physics, basis, nodes = _create_diffusion_physics_and_basis(bkd)
        npts = basis.npts()
        num_kle_terms = 2
        mesh_coords = ((nodes + 1.0) / 2.0)[None, :]
        mean_log = bkd.zeros((npts,))
        fm_d = create_lognormal_kle_field_map(
            mesh_coords,
            mean_log,
            bkd,
            num_kle_terms=num_kle_terms,
            sigma=0.3,
        )
        dp = create_diffusion_parameterization(physics, bkd, basis, fm_d)

        base_forcing = bkd.sin(math.pi * nodes)
        fm_f = ScalarAmplitude(bkd, base_forcing)
        fp = ForcingParameterization(physics, fm_f, bkd)

        comp = CompositeParameterization([dp, fp], bkd)
        init_state = bkd.zeros((npts,))
        fwd = SteadyForwardModel(physics, bkd, init_state, parameterization=comp)
        return fwd

    def test_nvars_is_sum(self, bkd) -> None:
        """Forward model nvars = nkle + 1 (diffusion + forcing)."""
        fwd = self._create_composite_forward_model(bkd)
        assert fwd.nvars() == 3

    def test_call_works(self, bkd) -> None:
        """Forward model __call__ works with CompositeParameterization."""
        fwd = self._create_composite_forward_model(bkd)
        samples = bkd.array([0.1, -0.1, 2.0])[:, None]
        result = fwd(samples)
        assert result.shape[0] == fwd.nqoi()
        assert result.shape[1] == 1

    def test_jacobian_derivative_checker(self, bkd) -> None:
        """Forward model Jacobian passes DerivativeChecker."""
        fwd = self._create_composite_forward_model(bkd)
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=fwd.nqoi(),
            nvars=fwd.nvars(),
            fun=fwd,
            jacobian=fwd.derivatives().jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.array([0.1, -0.1, 2.0])[:, None]
        errors = checker.check_derivatives(sample)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5
