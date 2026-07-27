"""Tests for TimeIntegratedWeightedL2Functional.

Pattern under test (the ONLY sanctioned way to weight a time-integrated
functional): the quadrature comes from the time-integration scheme via
``stepper.trajectory_quadrature(times)`` / model injection — never from
hand-rolled weights, which silently break the order-consistency
between scheme and quadrature.
"""

import numpy as np
import pytest
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.functionals.protocols import (
    TimeQuadratureAwareFunctionalProtocol,
    TransientFunctionalWithJacobianAndHVPProtocol,
)
from pyapprox.ode.functionals.time_integrated_weighted_l2 import (
    TimeIntegratedWeightedL2Functional,
)
from pyapprox.ode.time_quadrature import (
    midpoint_quadrature,
    right_rectangle_quadrature,
    trapezoidal_quadrature,
)

_NSTATES = 4
_NTIMES = 6
_NPARAMS = 3
_DELTAT = 0.1


def _weight_matrix_np():
    rng = np.random.default_rng(11)
    root = rng.normal(0.0, 1.0, (_NSTATES, _NSTATES))
    return root @ root.T + _NSTATES * np.eye(_NSTATES)


def _times(bkd):
    return bkd.asarray(np.linspace(0.0, _DELTAT * (_NTIMES - 1), _NTIMES))


def _make_functional(bkd, quadrature="trapezoid"):
    func = TimeIntegratedWeightedL2Functional(
        bkd.asarray(_weight_matrix_np()), _NPARAMS, bkd
    )
    if quadrature == "trapezoid":
        func.set_time_quadrature(trapezoidal_quadrature(_times(bkd), bkd))
    elif quadrature == "right":
        func.set_time_quadrature(
            right_rectangle_quadrature(_times(bkd), bkd)
        )
    elif quadrature == "midpoint":
        func.set_time_quadrature(midpoint_quadrature(_times(bkd), bkd))
    return func


class TestTimeIntegratedWeightedL2Functional:
    def test_protocol_conformance(self, bkd) -> None:
        func = _make_functional(bkd)
        assert isinstance(
            func, TransientFunctionalWithJacobianAndHVPProtocol
        )
        assert isinstance(func, TimeQuadratureAwareFunctionalProtocol)
        assert func.nqoi() == 1
        assert func.nstates() == _NSTATES
        assert func.nparams() == _NPARAMS
        assert func.nunique_params() == 0

    def test_constructor_validation(self, bkd) -> None:
        with pytest.raises(ValueError, match="nstates, nstates"):
            TimeIntegratedWeightedL2Functional(
                bkd.ones((_NSTATES, _NSTATES + 1)), _NPARAMS, bkd
            )
        asymmetric = np.eye(_NSTATES)
        asymmetric[0, 1] = 1.0
        with pytest.raises(ValueError, match="symmetric"):
            TimeIntegratedWeightedL2Functional(
                bkd.asarray(asymmetric), _NPARAMS, bkd
            )

    def test_evaluation_requires_injected_quadrature(self, bkd) -> None:
        """The functional REFUSES to run on hand-assumed weights: the
        scheme's rule must be injected first."""
        func = _make_functional(bkd, quadrature=None)
        sol = bkd.zeros((_NSTATES, _NTIMES))
        param = bkd.zeros((_NPARAMS, 1))
        with pytest.raises(RuntimeError, match="set_time_quadrature"):
            func(sol, param)
        with pytest.raises(RuntimeError, match="set_time_quadrature"):
            func.state_jacobian(sol, param)

    def test_set_time_quadrature_rejects_non_protocol(self, bkd) -> None:
        func = _make_functional(bkd, quadrature=None)
        with pytest.raises(TypeError, match="TrajectoryQuadrature"):
            func.set_time_quadrature(np.ones(_NTIMES))

    def test_sol_shape_validation(self, bkd) -> None:
        func = _make_functional(bkd)
        param = bkd.zeros((_NPARAMS, 1))
        with pytest.raises(ValueError, match="states"):
            func(bkd.zeros((_NSTATES + 1, _NTIMES)), param)
        with pytest.raises(ValueError, match="times"):
            func(bkd.zeros((_NSTATES, _NTIMES + 1)), param)

    @pytest.mark.parametrize("rule", ["trapezoid", "right", "midpoint"])
    def test_value_matches_quadrature_sum(self, bkd, rule) -> None:
        func = _make_functional(bkd, quadrature=rule)
        sol_np = np.random.default_rng(3).normal(
            0.0, 1.0, (_NSTATES, _NTIMES)
        )
        weight_np = _weight_matrix_np()
        if rule == "trapezoid":
            quad_w = np.full(_NTIMES, _DELTAT)
            quad_w[0] = quad_w[-1] = 0.5 * _DELTAT
            samples = sol_np
        elif rule == "right":
            quad_w = np.full(_NTIMES - 1, _DELTAT)
            samples = sol_np[:, 1:]
        else:
            quad_w = np.full(_NTIMES - 1, _DELTAT)
            samples = 0.5 * (sol_np[:, :-1] + sol_np[:, 1:])
        expected = sum(
            quad_w[jj] * samples[:, jj] @ weight_np @ samples[:, jj]
            for jj in range(quad_w.shape[0])
        )
        value = func(bkd.asarray(sol_np), bkd.zeros((_NPARAMS, 1)))
        assert value.shape == (1, 1)
        bkd.assert_allclose(
            value, bkd.asarray(np.array([[expected]])), rtol=1e-12
        )

    def test_state_jacobian_every_step_nonzero(self, bkd) -> None:
        func = _make_functional(bkd)
        sol_np = np.random.default_rng(5).normal(
            0.0, 1.0, (_NSTATES, _NTIMES)
        )
        dqdu = func.state_jacobian(
            bkd.asarray(sol_np), bkd.zeros((_NPARAMS, 1))
        )
        assert dqdu.shape == (_NSTATES, _NTIMES)
        quad_w = np.full(_NTIMES, _DELTAT)
        quad_w[0] = quad_w[-1] = 0.5 * _DELTAT
        expected = 2.0 * (_weight_matrix_np() @ sol_np) * quad_w
        bkd.assert_allclose(dqdu, bkd.asarray(expected), rtol=1e-12)

    @pytest.mark.parametrize("rule", ["trapezoid", "right", "midpoint"])
    def test_state_jacobian_matches_fd(self, bkd, rule) -> None:
        """FD check of the chain rule through the sampling operator —
        including the time-coupled midpoint rule."""
        func = _make_functional(bkd, quadrature=rule)
        param = bkd.zeros((_NPARAMS, 1))
        nvars = _NSTATES * _NTIMES

        def eval_fn(samples):
            results = [
                func(
                    bkd.reshape(samples[:, ii], (_NSTATES, _NTIMES)),
                    param,
                )
                for ii in range(samples.shape[1])
            ]
            return bkd.hstack(results)

        def jac_fn(sample):
            sol = bkd.reshape(sample[:, 0], (_NSTATES, _NTIMES))
            return bkd.reshape(
                func.state_jacobian(sol, param), (1, nvars)
            )

        wrapped = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=nvars, fun=eval_fn, jacobian=jac_fn, bkd=bkd
        )
        checker = DerivativeChecker(wrapped)
        sample = bkd.asarray(
            np.random.default_rng(7).normal(0.0, 1.0, (nvars, 1))
        )
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        # Observed V-bottom ~1.3e-6 locally (600-var FD accumulates
        # more rounding than the endpoint tests); 1e-5 leaves the
        # cross-platform drift headroom the CI matrix needs.
        assert ratio <= 1e-5

    def test_state_state_hvp_is_scaled_weight_matvec(self, bkd) -> None:
        """Q is quadratic in y: the exact per-step HVP is
        2 omega_n W wvec with omega the scheme's per-column weights."""
        func = _make_functional(bkd)
        sol = bkd.zeros((_NSTATES, _NTIMES))
        param = bkd.zeros((_NPARAMS, 1))
        wvec_np = np.random.default_rng(9).normal(0.0, 1.0, (_NSTATES, 1))
        quad_w = np.full(_NTIMES, _DELTAT)
        quad_w[0] = quad_w[-1] = 0.5 * _DELTAT
        for time_idx in [0, _NTIMES // 2, _NTIMES - 1]:
            expected = (
                2.0 * quad_w[time_idx] * _weight_matrix_np() @ wvec_np
            )
            hvp = func.state_state_hvp(
                sol, param, time_idx, bkd.asarray(wvec_np)
            )
            assert hvp.shape == (_NSTATES, 1)
            bkd.assert_allclose(hvp, bkd.asarray(expected), rtol=1e-12)

    def test_midpoint_rule_refuses_per_step_hvp(self, bkd) -> None:
        """The midpoint rule couples adjacent steps: per-step HVPs
        must raise rather than silently drop the coupling."""
        func = _make_functional(bkd, quadrature="midpoint")
        sol = bkd.zeros((_NSTATES, _NTIMES))
        param = bkd.zeros((_NPARAMS, 1))
        wvec = bkd.ones((_NSTATES, 1))
        with pytest.raises(ValueError, match="couples adjacent"):
            func.state_state_hvp(sol, param, 0, wvec)

    def test_param_terms_are_zero(self, bkd) -> None:
        func = _make_functional(bkd)
        sol = bkd.zeros((_NSTATES, _NTIMES))
        param = bkd.zeros((_NPARAMS, 1))
        wvec = bkd.ones((_NSTATES, 1))
        vvec = bkd.ones((_NPARAMS, 1))
        bkd.assert_allclose(
            func.param_jacobian(sol, param),
            bkd.zeros((1, _NPARAMS)),
            rtol=1e-14,
        )
        bkd.assert_allclose(
            func.state_param_hvp(sol, param, 0, vvec),
            bkd.zeros((_NSTATES, 1)),
            rtol=1e-14,
        )
        bkd.assert_allclose(
            func.param_state_hvp(sol, param, 0, wvec),
            bkd.zeros((_NPARAMS, 1)),
            rtol=1e-14,
        )
        bkd.assert_allclose(
            func.param_param_hvp(sol, param, vvec),
            bkd.zeros((_NPARAMS, 1)),
            rtol=1e-14,
        )
