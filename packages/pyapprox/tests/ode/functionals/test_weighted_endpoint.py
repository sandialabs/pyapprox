"""Tests for WeightedEndpointFunctional."""

import numpy as np
import pytest
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.ode.functionals.protocols import (
    TransientFunctionalWithJacobianAndHVPProtocol,
)
from pyapprox.ode.functionals.weighted_endpoint import (
    WeightedEndpointFunctional,
)

_NSTATES = 4
_NTIMES = 6
_NPARAMS = 3


def _make_functional(bkd, weights_np=None):
    if weights_np is None:
        weights_np = np.arange(1.0, _NSTATES + 1.0)
    weights = bkd.asarray(weights_np.reshape(-1, 1))
    return WeightedEndpointFunctional(weights, _NPARAMS, bkd)


class TestWeightedEndpointFunctional:
    def test_protocol_conformance(self, bkd) -> None:
        func = _make_functional(bkd)
        assert isinstance(
            func, TransientFunctionalWithJacobianAndHVPProtocol
        )
        assert func.nqoi() == 1
        assert func.nstates() == _NSTATES
        assert func.nparams() == _NPARAMS
        assert func.nunique_params() == 0

    def test_weights_shape_validation(self, bkd) -> None:
        with pytest.raises(ValueError, match="weights must have shape"):
            WeightedEndpointFunctional(
                bkd.asarray(np.ones(_NSTATES)), _NPARAMS, bkd
            )

    def test_value_is_weighted_final_state(self, bkd) -> None:
        func = _make_functional(bkd)
        sol_np = np.random.default_rng(3).normal(
            0.0, 1.0, (_NSTATES, _NTIMES)
        )
        sol = bkd.asarray(sol_np)
        param = bkd.zeros((_NPARAMS, 1))
        expected = np.arange(1.0, _NSTATES + 1.0) @ sol_np[:, -1]
        value = func(sol, param)
        assert value.shape == (1, 1)
        bkd.assert_allclose(
            value, bkd.asarray(np.array([[expected]])), rtol=1e-14
        )

    def test_wrong_sol_nstates_raises(self, bkd) -> None:
        func = _make_functional(bkd)
        sol = bkd.zeros((_NSTATES + 1, _NTIMES))
        param = bkd.zeros((_NPARAMS, 1))
        with pytest.raises(ValueError, match="states"):
            func(sol, param)

    def test_state_jacobian_is_weights_at_final_time(self, bkd) -> None:
        func = _make_functional(bkd)
        sol = bkd.zeros((_NSTATES, _NTIMES))
        param = bkd.zeros((_NPARAMS, 1))
        dqdu = func.state_jacobian(sol, param)
        assert dqdu.shape == (_NSTATES, _NTIMES)
        expected = np.zeros((_NSTATES, _NTIMES))
        expected[:, -1] = np.arange(1.0, _NSTATES + 1.0)
        bkd.assert_allclose(dqdu, bkd.asarray(expected), rtol=1e-14)

    def test_param_jacobian_and_hvps_are_zero(self, bkd) -> None:
        func = _make_functional(bkd)
        sol = bkd.zeros((_NSTATES, _NTIMES))
        param = bkd.zeros((_NPARAMS, 1))
        wvec = bkd.zeros((_NSTATES, 1))
        vvec = bkd.zeros((_NPARAMS, 1))
        bkd.assert_allclose(
            func.param_jacobian(sol, param),
            bkd.zeros((1, _NPARAMS)),
            rtol=1e-14,
        )
        bkd.assert_allclose(
            func.state_state_hvp(sol, param, 0, wvec),
            bkd.zeros((_NSTATES, 1)),
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

    def test_state_jacobian_matches_fd(self, bkd) -> None:
        """DerivativeChecker FD validation of the analytical
        state_jacobian on both backends (no autograd dependency)."""
        func = _make_functional(bkd)
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
        assert ratio <= 1e-6

    def test_one_hot_weights_match_endpoint_functional(self, bkd) -> None:
        """A one-hot weight vector reduces exactly to EndpointFunctional."""
        state_idx = 2
        one_hot = np.zeros(_NSTATES)
        one_hot[state_idx] = 1.0
        weighted = _make_functional(bkd, one_hot)
        endpoint = EndpointFunctional(state_idx, _NSTATES, _NPARAMS, bkd)
        sol = bkd.asarray(
            np.random.default_rng(5).normal(0.0, 1.0, (_NSTATES, _NTIMES))
        )
        param = bkd.zeros((_NPARAMS, 1))
        bkd.assert_allclose(
            weighted(sol, param), endpoint(sol, param), rtol=1e-14
        )
        bkd.assert_allclose(
            weighted.state_jacobian(sol, param),
            endpoint.state_jacobian(sol, param),
            rtol=1e-14,
        )
