"""Tests for TikhonovAugmentedFunctional."""

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
    TransientFunctionalWithJacobianProtocol,
)
from pyapprox.ode.functionals.tikhonov import TikhonovAugmentedFunctional
from pyapprox.ode.functionals.time_integrated_weighted_l2 import (
    TimeIntegratedWeightedL2Functional,
)
from pyapprox.ode.functionals.weighted_endpoint import (
    WeightedEndpointFunctional,
)
from pyapprox.ode.time_quadrature import trapezoidal_quadrature

_NSTATES = 4
_NTIMES = 6
_NPARAMS = 3
_ALPHA = 0.7


def _spd_np(nrows, seed):
    root = np.random.default_rng(seed).normal(0.0, 1.0, (nrows, nrows))
    return root @ root.T + nrows * np.eye(nrows)


_INNER_LINEAR_NP = np.arange(1.0, _NPARAMS + 1.0).reshape(1, -1)
_INNER_HESSIAN_NP = _spd_np(_NPARAMS, 17)


class _ParamCoupledEndpoint(WeightedEndpointFunctional):
    """Q = c^T y(T) + g^T p + (1/2) p^T H p: NONZERO inner param
    derivatives, so augmentation must sum both contributions."""

    def __call__(self, sol, param):
        base = super().__call__(sol, param)
        gvec = self._bkd.asarray(_INNER_LINEAR_NP)
        hmat = self._bkd.asarray(_INNER_HESSIAN_NP)
        extra = self._bkd.sum(gvec * param.T) + 0.5 * self._bkd.sum(
            param * self._bkd.dot(hmat, param)
        )
        return base + self._bkd.reshape(extra, (1, 1))

    def param_jacobian(self, sol, param):
        gvec = self._bkd.asarray(_INNER_LINEAR_NP)
        hmat = self._bkd.asarray(_INNER_HESSIAN_NP)
        return gvec + self._bkd.dot(hmat, param).T

    def param_param_hvp(self, sol, param, vvec):
        return self._bkd.dot(self._bkd.asarray(_INNER_HESSIAN_NP), vvec)


def _make_inner(bkd):
    weights = bkd.asarray(np.arange(1.0, _NSTATES + 1.0).reshape(-1, 1))
    return _ParamCoupledEndpoint(weights, _NPARAMS, bkd)


def _make_functional(bkd, weight_np=None):
    weight = None if weight_np is None else bkd.asarray(weight_np)
    return TikhonovAugmentedFunctional(
        _make_inner(bkd), _ALPHA, bkd, weight=weight
    )


class _JacobianOnlyFunctional:
    """Minimal jacobian-tier functional (no HVP methods)."""

    def __init__(self, bkd, nqoi=1):
        self._bkd = bkd
        self._nqoi = nqoi

    def bkd(self):
        return self._bkd

    def nqoi(self):
        return self._nqoi

    def nstates(self):
        return _NSTATES

    def nparams(self):
        return _NPARAMS

    def nunique_params(self):
        return 0

    def __call__(self, sol, param):
        return self._bkd.zeros((self._nqoi, 1))

    def state_jacobian(self, sol, param):
        return self._bkd.zeros(sol.shape)

    def param_jacobian(self, sol, param):
        return self._bkd.zeros((self._nqoi, _NPARAMS))


class TestTikhonovAugmentedFunctional:
    def test_wrapper_matches_inner_tier(self, bkd) -> None:
        """Wrapping an HVP-tier inner yields an HVP-tier wrapper;
        wrapping a jacobian-tier inner must not fake HVP support."""
        hvp_tier = _make_functional(bkd)
        assert isinstance(
            hvp_tier, TransientFunctionalWithJacobianAndHVPProtocol
        )
        jac_tier = TikhonovAugmentedFunctional(
            _JacobianOnlyFunctional(bkd), _ALPHA, bkd
        )
        assert isinstance(
            jac_tier, TransientFunctionalWithJacobianProtocol
        )
        assert not isinstance(
            jac_tier, TransientFunctionalWithJacobianAndHVPProtocol
        )

    def test_constructor_validation(self, bkd) -> None:
        with pytest.raises(TypeError, match="inner must satisfy"):
            TikhonovAugmentedFunctional(object(), _ALPHA, bkd)
        with pytest.raises(ValueError, match="nqoi"):
            TikhonovAugmentedFunctional(
                _JacobianOnlyFunctional(bkd, nqoi=2), _ALPHA, bkd
            )
        with pytest.raises(ValueError, match="shape"):
            _make_functional(bkd, np.eye(_NPARAMS + 1))
        asymmetric = np.eye(_NPARAMS)
        asymmetric[0, 1] = 1.0
        with pytest.raises(ValueError, match="symmetric"):
            _make_functional(bkd, asymmetric)

    @pytest.mark.parametrize("weighted", [False, True])
    def test_value_adds_quadratic_cost(self, bkd, weighted) -> None:
        weight_np = _spd_np(_NPARAMS, 21) if weighted else None
        func = _make_functional(bkd, weight_np)
        inner = _make_inner(bkd)
        rng = np.random.default_rng(3)
        sol_np = rng.normal(0.0, 1.0, (_NSTATES, _NTIMES))
        param_np = rng.normal(0.0, 1.0, (_NPARAMS, 1))
        sol = bkd.asarray(sol_np)
        param = bkd.asarray(param_np)
        wq = np.eye(_NPARAMS) if weight_np is None else weight_np
        cost = 0.5 * _ALPHA * (param_np[:, 0] @ wq @ param_np[:, 0])
        expected = inner(sol, param) + bkd.asarray(np.array([[cost]]))
        bkd.assert_allclose(func(sol, param), expected, rtol=1e-12)

    @pytest.mark.parametrize("weighted", [False, True])
    def test_param_jacobian_and_hvp_sum_both_terms(
        self, bkd, weighted
    ) -> None:
        """Inner param derivatives are NONZERO here: the wrapper must
        return inner + Tikhonov, not either alone."""
        weight_np = _spd_np(_NPARAMS, 21) if weighted else None
        func = _make_functional(bkd, weight_np)
        rng = np.random.default_rng(5)
        sol = bkd.asarray(rng.normal(0.0, 1.0, (_NSTATES, _NTIMES)))
        param_np = rng.normal(0.0, 1.0, (_NPARAMS, 1))
        vvec_np = rng.normal(0.0, 1.0, (_NPARAMS, 1))
        wq = np.eye(_NPARAMS) if weight_np is None else weight_np
        expected_jac = (
            _INNER_LINEAR_NP
            + (_INNER_HESSIAN_NP @ param_np).T
            + _ALPHA * (wq @ param_np).T
        )
        bkd.assert_allclose(
            func.param_jacobian(sol, bkd.asarray(param_np)),
            bkd.asarray(expected_jac),
            rtol=1e-12,
        )
        expected_hvp = (
            _INNER_HESSIAN_NP @ vvec_np + _ALPHA * wq @ vvec_np
        )
        bkd.assert_allclose(
            func.param_param_hvp(
                sol, bkd.asarray(param_np), bkd.asarray(vvec_np)
            ),
            bkd.asarray(expected_hvp),
            rtol=1e-12,
        )

    def test_state_terms_delegate_to_inner(self, bkd) -> None:
        func = _make_functional(bkd)
        inner = _make_inner(bkd)
        rng = np.random.default_rng(7)
        sol = bkd.asarray(rng.normal(0.0, 1.0, (_NSTATES, _NTIMES)))
        param = bkd.asarray(rng.normal(0.0, 1.0, (_NPARAMS, 1)))
        wvec = bkd.asarray(rng.normal(0.0, 1.0, (_NSTATES, 1)))
        bkd.assert_allclose(
            func.state_jacobian(sol, param),
            inner.state_jacobian(sol, param),
            rtol=1e-14,
        )
        bkd.assert_allclose(
            func.state_state_hvp(sol, param, 0, wvec),
            inner.state_state_hvp(sol, param, 0, wvec),
            rtol=1e-14,
        )

    def test_time_quadrature_injection_delegates_to_inner(
        self, bkd
    ) -> None:
        """Wrapping a quadrature-aware inner: the wrapper is itself
        quadrature-aware and forwards the injected rule; wrapping a
        non-aware inner must not fake awareness."""
        weight = bkd.asarray(_spd_np(_NSTATES, 29))
        inner = TimeIntegratedWeightedL2Functional(weight, _NPARAMS, bkd)
        func = TikhonovAugmentedFunctional(inner, _ALPHA, bkd)
        assert isinstance(func, TimeQuadratureAwareFunctionalProtocol)
        times = bkd.asarray(np.linspace(0.0, 0.5, _NTIMES))
        func.set_time_quadrature(trapezoidal_quadrature(times, bkd))
        sol = bkd.asarray(
            np.random.default_rng(31).normal(
                0.0, 1.0, (_NSTATES, _NTIMES)
            )
        )
        param = bkd.zeros((_NPARAMS, 1))
        bkd.assert_allclose(
            func(sol, param), inner(sol, param), rtol=1e-14
        )
        not_aware = TikhonovAugmentedFunctional(
            _make_inner(bkd), _ALPHA, bkd
        )
        assert not isinstance(
            not_aware, TimeQuadratureAwareFunctionalProtocol
        )

    @pytest.mark.parametrize("weighted", [False, True])
    def test_param_jacobian_matches_fd(self, bkd, weighted) -> None:
        """FD check of the SUMMED param jacobian (nonzero inner +
        Tikhonov) as a function of p at fixed sol."""
        weight_np = _spd_np(_NPARAMS, 21) if weighted else None
        func = _make_functional(bkd, weight_np)
        sol = bkd.asarray(
            np.random.default_rng(9).normal(0.0, 1.0, (_NSTATES, _NTIMES))
        )

        def eval_fn(samples):
            results = [
                func(sol, samples[:, ii : ii + 1])
                for ii in range(samples.shape[1])
            ]
            return bkd.hstack(results)

        def jac_fn(sample):
            return func.param_jacobian(sol, sample)

        wrapped = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=_NPARAMS, fun=eval_fn, jacobian=jac_fn, bkd=bkd
        )
        checker = DerivativeChecker(wrapped)
        sample = bkd.asarray(
            np.random.default_rng(13).normal(0.0, 1.0, (_NPARAMS, 1))
        )
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-6
