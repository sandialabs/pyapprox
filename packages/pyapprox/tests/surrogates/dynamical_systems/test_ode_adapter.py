"""Tests for VectorFieldODEAdapter."""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.implicit_steppers.backward_euler import BackwardEulerAdjoint
from pyapprox.ode.implicit_steppers.crank_nicolson import CrankNicolsonAdjoint
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.protocols.ode_residual import (
    ODEResidualProtocol,
    ODEResidualWithParamJacobianProtocol,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.dynamical_systems.ode_adapter import (
    VectorFieldODEAdapter,
)
from pyapprox.surrogates.dynamical_systems.vector_fields import (
    BasisExpansionVectorField,
)
from pyapprox.util.rootfinding.newton import NewtonSolver


def _make_vf(bkd, nvars=2, max_level=2, seed=0):
    marginals = [UniformMarginal(-2.0, 2.0, bkd) for _ in range(nvars)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    exp = BasisExpansion(basis, bkd, nqoi=nvars)
    rng = np.random.RandomState(seed)
    exp.set_coefficients(bkd.array(rng.randn(exp.nterms(), nvars)))
    return BasisExpansionVectorField(exp)


class _AdapterStateFunction:
    """Wraps adapter __call__ as FunctionWithJacobianProtocol for DerivativeChecker."""

    def __init__(self, adapter, bkd):
        self._adapter = adapter
        self._bkd = bkd

    def bkd(self):
        return self._bkd

    def nvars(self):
        return self._adapter._vf.nstates()

    def nqoi(self):
        return self._adapter._vf.nstates()

    def __call__(self, samples):
        state = samples[:, 0]
        result = self._adapter(state)
        return result[:, None]

    def jacobian(self, sample):
        state = sample[:, 0]
        return self._adapter.jacobian(state)


class _AdapterParamFunction:
    """Wraps adapter as function of params for DerivativeChecker."""

    def __init__(self, adapter, fixed_state, bkd):
        self._adapter = adapter
        self._fixed_state = fixed_state
        self._bkd = bkd

    def bkd(self):
        return self._bkd

    def nvars(self):
        return self._adapter.nparams()

    def nqoi(self):
        return self._adapter._vf.nstates()

    def __call__(self, params):
        p = params[:, 0]
        self._adapter.set_param(p)
        result = self._adapter(self._fixed_state)
        return result[:, None]

    def jacobian(self, params):
        p = params[:, 0]
        self._adapter.set_param(p)
        return self._adapter.param_jacobian(self._fixed_state)


class TestVectorFieldODEAdapter:
    def test_protocol_conformance(self, bkd):
        vf = _make_vf(bkd)
        adapter = VectorFieldODEAdapter(vf)
        assert isinstance(adapter, ODEResidualProtocol)
        assert isinstance(adapter, ODEResidualWithParamJacobianProtocol)

    def test_call_shape(self, bkd):
        vf = _make_vf(bkd)
        adapter = VectorFieldODEAdapter(vf)
        state = bkd.array(np.random.RandomState(0).uniform(-1, 1, (2,)))
        result = adapter(state)
        assert result.shape == (2,)

    def test_call_matches_vf(self, bkd):
        vf = _make_vf(bkd)
        adapter = VectorFieldODEAdapter(vf)
        rng = np.random.RandomState(0)
        state_1d = bkd.array(rng.uniform(-1, 1, (2,)))
        state_2d = state_1d[:, None]
        bkd.assert_allclose(adapter(state_1d), vf(state_2d)[:, 0])

    def test_jacobian_shape(self, bkd):
        vf = _make_vf(bkd)
        adapter = VectorFieldODEAdapter(vf)
        state = bkd.array(np.random.RandomState(0).uniform(-1, 1, (2,)))
        jac = adapter.jacobian(state)
        assert jac.shape == (2, 2)

    def test_jacobian_matches_vf(self, bkd):
        vf = _make_vf(bkd)
        adapter = VectorFieldODEAdapter(vf)
        rng = np.random.RandomState(0)
        state_1d = bkd.array(rng.uniform(-1, 1, (2,)))
        state_2d = state_1d[:, None]
        bkd.assert_allclose(
            adapter.jacobian(state_1d), vf.state_jacobian(state_2d)[0]
        )

    def test_param_jacobian_shape(self, bkd):
        vf = _make_vf(bkd)
        adapter = VectorFieldODEAdapter(vf)
        state = bkd.array(np.random.RandomState(0).uniform(-1, 1, (2,)))
        jac = adapter.param_jacobian(state)
        nparams = adapter.nparams()
        assert jac.shape == (2, nparams)

    def test_param_jacobian_matches_vf(self, bkd):
        vf = _make_vf(bkd)
        adapter = VectorFieldODEAdapter(vf)
        rng = np.random.RandomState(0)
        state_1d = bkd.array(rng.uniform(-1, 1, (2,)))
        state_2d = state_1d[:, None]
        bkd.assert_allclose(
            adapter.param_jacobian(state_1d), vf.param_jacobian(state_2d)[0]
        )

    def test_mass_matrix(self, bkd):
        vf = _make_vf(bkd)
        adapter = VectorFieldODEAdapter(vf)
        M = adapter.mass_matrix(2)
        bkd.assert_allclose(M, bkd.eye(2))

    def test_apply_mass_matrix(self, bkd):
        vf = _make_vf(bkd)
        adapter = VectorFieldODEAdapter(vf)
        vec = bkd.array([3.0, -1.0])
        bkd.assert_allclose(adapter.apply_mass_matrix(vec), vec)

    def test_set_time(self, bkd):
        vf = _make_vf(bkd)
        adapter = VectorFieldODEAdapter(vf)
        adapter.set_time(1.5)
        state = bkd.array(np.random.RandomState(0).uniform(-1, 1, (2,)))
        result = adapter(state)
        assert result.shape == (2,)

    def test_set_param(self, bkd):
        vf = _make_vf(bkd)
        adapter = VectorFieldODEAdapter(vf)
        rng = np.random.RandomState(0)
        state = bkd.array(rng.uniform(-1, 1, (2,)))
        result_before = bkd.copy(adapter(state))
        new_params = bkd.array(rng.randn(adapter.nparams()))
        adapter.set_param(new_params)
        result_after = adapter(state)
        assert not np.allclose(
            bkd.to_numpy(result_before), bkd.to_numpy(result_after)
        )

    def test_set_param_2d(self, bkd):
        vf = _make_vf(bkd)
        adapter = VectorFieldODEAdapter(vf)
        rng = np.random.RandomState(0)
        params_2d = bkd.array(rng.randn(adapter.nparams(), 1))
        adapter.set_param(params_2d)
        state = bkd.array(rng.uniform(-1, 1, (2,)))
        result = adapter(state)
        assert result.shape == (2,)

    def test_initial_param_jacobian(self, bkd):
        vf = _make_vf(bkd)
        adapter = VectorFieldODEAdapter(vf)
        jac = adapter.initial_param_jacobian()
        assert jac.shape == (2, adapter.nparams())
        bkd.assert_allclose(jac, bkd.zeros((2, adapter.nparams())))

    def test_nparams(self, bkd):
        vf = _make_vf(bkd)
        adapter = VectorFieldODEAdapter(vf)
        assert adapter.nparams() == vf.hyp_list().nactive_params()

    def test_state_jacobian_derivative_check(self, bkd):
        vf = _make_vf(bkd, max_level=3)
        adapter = VectorFieldODEAdapter(vf)
        wrapper = _AdapterStateFunction(adapter, bkd)
        checker = DerivativeChecker(wrapper)
        rng = np.random.RandomState(42)
        sample = bkd.array(rng.uniform(-1, 1, (2, 1)))
        errors = checker.check_derivatives(sample)
        assert checker.error_ratio(errors[0]) <= 1e-6

    def test_param_jacobian_derivative_check(self, bkd):
        vf = _make_vf(bkd)
        adapter = VectorFieldODEAdapter(vf)
        rng = np.random.RandomState(42)
        fixed_state = bkd.array(rng.uniform(-1, 1, (2,)))
        wrapper = _AdapterParamFunction(adapter, fixed_state, bkd)
        checker = DerivativeChecker(wrapper)
        params = bkd.array(rng.randn(adapter.nparams(), 1))
        errors = checker.check_derivatives(params, relative=False)
        assert bkd.min(errors[0]) < 1e-10

    def test_invalid_vf_raises(self, bkd):
        with pytest.raises(TypeError, match="ParametricVectorFieldProtocol"):
            VectorFieldODEAdapter("not a vector field")

    def test_3d_system(self, bkd):
        vf = _make_vf(bkd, nvars=3)
        adapter = VectorFieldODEAdapter(vf)
        state = bkd.array(np.random.RandomState(0).uniform(-1, 1, (3,)))
        result = adapter(state)
        assert result.shape == (3,)
        jac = adapter.jacobian(state)
        assert jac.shape == (3, 3)


def _make_decay_adapter(bkd):
    """Build adapter for dx/dt = -x (scalar exponential decay)."""
    marginals = [UniformMarginal(-2.0, 2.0, bkd)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(1, 1, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    exp = BasisExpansion(basis, bkd, nqoi=1)
    c1 = -2.0 / np.sqrt(3)
    exp.set_coefficients(bkd.array([[0.0], [c1]]))
    vf = BasisExpansionVectorField(exp)
    return VectorFieldODEAdapter(vf)


def _solve_decay(adapter, bkd, dt, final_time=1.0):
    stepper_cls = BackwardEulerAdjoint
    stepper = stepper_cls(adapter)
    newton = NewtonSolver(stepper)
    newton.set_options(maxiters=20, atol=1e-14, rtol=0.0)
    integrator = TimeIntegrator(0.0, final_time, dt, newton)
    states, times = integrator.solve(bkd.array([1.0]))
    return bkd.to_numpy(states[0, -1])


class TestODEAdapterConvergence:
    def test_backward_euler_first_order(self, numpy_bkd):
        bkd = numpy_bkd
        adapter = _make_decay_adapter(bkd)
        exact = np.exp(-1.0)
        dts = [0.1, 0.05, 0.025]
        errors = []
        for dt in dts:
            stepper = BackwardEulerAdjoint(adapter)
            newton = NewtonSolver(stepper)
            newton.set_options(maxiters=20, atol=1e-14, rtol=0.0)
            integrator = TimeIntegrator(0.0, 1.0, dt, newton)
            states, _ = integrator.solve(bkd.array([1.0]))
            errors.append(abs(bkd.to_numpy(states[0, -1]) - exact))
        rate = np.log(errors[0] / errors[-1]) / np.log(dts[0] / dts[-1])
        assert rate > 0.9

    def test_crank_nicolson_second_order(self, numpy_bkd):
        bkd = numpy_bkd
        adapter = _make_decay_adapter(bkd)
        exact = np.exp(-1.0)
        dts = [0.1, 0.05, 0.025]
        errors = []
        for dt in dts:
            stepper = CrankNicolsonAdjoint(adapter)
            newton = NewtonSolver(stepper)
            newton.set_options(maxiters=20, atol=1e-14, rtol=0.0)
            integrator = TimeIntegrator(0.0, 1.0, dt, newton)
            states, _ = integrator.solve(bkd.array([1.0]))
            errors.append(abs(bkd.to_numpy(states[0, -1]) - exact))
        rate = np.log(errors[0] / errors[-1]) / np.log(dts[0] / dts[-1])
        assert rate > 1.9
