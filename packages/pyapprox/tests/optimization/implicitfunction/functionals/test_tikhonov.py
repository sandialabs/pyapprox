"""Tests for TikhonovAugmentedSteadyFunctional."""

import dataclasses
from typing import Any, Literal, Optional, Set

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
    FunctionWithJVPFromCallable,
)
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.optimization.implicitfunction.functionals.derivatives import (
    SteadyFunctionalDerivatives,
)
from pyapprox.optimization.implicitfunction.functionals.mean_squared_error import (  # noqa: E501
    MSEFunctional,
)
from pyapprox.optimization.implicitfunction.functionals.subset_of_states import (  # noqa: E501
    SubsetOfStatesFunctional,
)
from pyapprox.optimization.implicitfunction.functionals.tikhonov import (
    TikhonovAugmentedSteadyFunctional,
)
from pyapprox.optimization.implicitfunction.functionals.tikhonov_mean_squared_error import (  # noqa: E501
    TikhonovMSEFunctional,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend

_NSTATES = 4
_NPARAMS = 3
_ALPHA = 0.7
_JAC = {"state_jacobian", "param_jacobian"}
_HVPS = {"state_state_hvp", "state_param_hvp", "param_state_hvp", "param_param_hvp"}


def _spd_np(nrows: int, seed: int) -> Any:
    root = np.random.default_rng(seed).normal(0.0, 1.0, (nrows, nrows))
    return root @ root.T + nrows * np.eye(nrows)


_INNER_LINEAR_NP = np.arange(1.0, _NPARAMS + 1.0).reshape(1, -1)
_INNER_HESSIAN_NP = _spd_np(_NPARAMS, 17)
_OBS_NP = np.arange(1.0, _NSTATES + 1.0).reshape(-1, 1)


class _ParamCoupledMSE(MSEFunctional[Array]):
    """Q = |u - d|^2 / 2 + g^T p + (1/2) p^T H p: NONZERO inner param
    derivatives, so augmentation must sum both contributions."""

    def __call__(self, state: Array, param: Array) -> Array:
        base = super().__call__(state, param)
        gvec = self._bkd.asarray(_INNER_LINEAR_NP)
        hmat = self._bkd.asarray(_INNER_HESSIAN_NP)
        extra = self._bkd.sum(gvec * param.T) + 0.5 * self._bkd.sum(
            param * self._bkd.dot(hmat, param)
        )
        return base + self._bkd.reshape(extra, (1, 1))

    def param_jacobian(self, state: Array, param: Array) -> Array:
        gvec = self._bkd.asarray(_INNER_LINEAR_NP)
        hmat = self._bkd.asarray(_INNER_HESSIAN_NP)
        return gvec + self._bkd.dot(hmat, param).T

    def param_param_hvp(self, state: Array, param: Array, vvec: Array) -> Array:
        return self._bkd.dot(self._bkd.asarray(_INNER_HESSIAN_NP), vvec)


class _ValueOnlyFunctional:
    """A steady functional with no derivatives: the sum of the state."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nqoi(self) -> int:
        return 1

    def nstates(self) -> int:
        return _NSTATES

    def nparams(self) -> int:
        return _NPARAMS

    def nunique_params(self) -> int:
        return 0

    def time_dependence(self) -> Literal["steady"]:
        return "steady"

    def derivatives(self) -> SteadyFunctionalDerivatives[Array]:
        return SteadyFunctionalDerivatives.none()

    def __call__(self, state: Array, param: Array) -> Array:
        return self._bkd.reshape(self._bkd.sum(state), (1, 1))


def _inner(bkd: Backend[Array]) -> _ParamCoupledMSE[Array]:
    inner = _ParamCoupledMSE(_NSTATES, _NPARAMS, bkd)
    inner.set_observations(bkd.asarray(_OBS_NP))
    return inner


def _wrapped(
    bkd: Backend[Array], weight_np: Optional[Any] = None
) -> TikhonovAugmentedSteadyFunctional[Array]:
    weight = None if weight_np is None else bkd.asarray(weight_np)
    return TikhonovAugmentedSteadyFunctional(
        _inner(bkd), _ALPHA, bkd, weight=weight
    )


def _present(bundle: SteadyFunctionalDerivatives[Any]) -> Set[str]:
    return {
        field.name
        for field in dataclasses.fields(bundle)
        if getattr(bundle, field.name) is not None
    }


def _state_param(bkd: Backend[Array], seed: int) -> Any:
    rng = np.random.default_rng(seed)
    state = bkd.asarray(rng.normal(0.0, 1.0, (_NSTATES, 1)))
    param = bkd.asarray(rng.normal(0.0, 1.0, (_NPARAMS, 1)))
    return state, param


class TestBundleMirrorsInner:
    def test_second_order_inner(self, numpy_bkd: NumpyBkd) -> None:
        assert _present(_wrapped(numpy_bkd).derivatives()) == _JAC | _HVPS

    def test_jacobian_only_inner(self, numpy_bkd: NumpyBkd) -> None:
        inner = SubsetOfStatesFunctional(
            _NSTATES, _NPARAMS, numpy_bkd.asarray([1]), numpy_bkd
        )
        wrapped = TikhonovAugmentedSteadyFunctional(inner, _ALPHA, numpy_bkd)
        assert _present(wrapped.derivatives()) == _JAC

    def test_value_only_inner(self, numpy_bkd: NumpyBkd) -> None:
        wrapped = TikhonovAugmentedSteadyFunctional(
            _ValueOnlyFunctional(numpy_bkd), _ALPHA, numpy_bkd
        )
        assert _present(wrapped.derivatives()) == set()

    def test_state_fields_are_the_inner_ones(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """The cost does not depend on the state."""
        inner = _inner(numpy_bkd)
        wrapped = TikhonovAugmentedSteadyFunctional(inner, _ALPHA, numpy_bkd)
        derivs = wrapped.derivatives()
        assert derivs.state_jacobian == inner.state_jacobian
        assert derivs.state_state_hvp == inner.state_state_hvp
        assert derivs.state_param_hvp == inner.state_param_hvp
        assert derivs.param_state_hvp == inner.param_state_hvp


class TestValueAndDerivatives:
    @pytest.mark.parametrize("weighted", [False, True])
    def test_value_adds_quadratic_cost(
        self, bkd: Backend[Array], weighted: bool
    ) -> None:
        weight_np = _spd_np(_NPARAMS, 21) if weighted else None
        state, param = _state_param(bkd, 3)
        param_np = bkd.to_numpy(param)
        wq = np.eye(_NPARAMS) if weight_np is None else weight_np
        cost = 0.5 * _ALPHA * (param_np[:, 0] @ wq @ param_np[:, 0])
        bkd.assert_allclose(
            _wrapped(bkd, weight_np)(state, param),
            _inner(bkd)(state, param) + bkd.asarray(np.array([[cost]])),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("weighted", [False, True])
    def test_param_derivatives_sum_both_terms(
        self, bkd: Backend[Array], weighted: bool
    ) -> None:
        """Inner param derivatives are NONZERO here: the wrapper must
        return inner + Tikhonov, not either alone."""
        weight_np = _spd_np(_NPARAMS, 21) if weighted else None
        derivs = _wrapped(bkd, weight_np).derivatives()
        param_jacobian = derivs.param_jacobian
        param_param_hvp = derivs.param_param_hvp
        assert param_jacobian is not None and param_param_hvp is not None
        state, param = _state_param(bkd, 5)
        param_np = bkd.to_numpy(param)
        vvec_np = np.random.default_rng(6).normal(0.0, 1.0, (_NPARAMS, 1))
        wq = np.eye(_NPARAMS) if weight_np is None else weight_np
        bkd.assert_allclose(
            param_jacobian(state, param),
            bkd.asarray(
                _INNER_LINEAR_NP
                + (_INNER_HESSIAN_NP @ param_np).T
                + _ALPHA * (wq @ param_np).T
            ),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            param_param_hvp(state, param, bkd.asarray(vvec_np)),
            bkd.asarray(_INNER_HESSIAN_NP @ vvec_np + _ALPHA * wq @ vvec_np),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("weighted", [False, True])
    def test_param_jacobian_and_hvp_match_fd(
        self, bkd: Backend[Array], weighted: bool
    ) -> None:
        """Finite differences of the summed value (in p, at fixed state)
        check the Jacobian; of the Jacobian, the HVP.

        The value is quadratic in p, so the forward-difference error of
        the Jacobian falls at first order down to a roundoff floor set by
        the value's magnitude; that floor sits near the usual 1e-6 error
        ratio, so the Jacobian check tests the shape of the error curve
        instead. The Jacobian is linear in p, so differencing it is exact
        up to roundoff and the HVP check keeps the error ratio."""
        weight_np = _spd_np(_NPARAMS, 21) if weighted else None
        func = _wrapped(bkd, weight_np)
        derivs = func.derivatives()
        param_jacobian = derivs.param_jacobian
        param_param_hvp = derivs.param_param_hvp
        assert param_jacobian is not None and param_param_hvp is not None
        state, sample = _state_param(bkd, 9)

        def values(samples: Array) -> Array:
            return bkd.hstack(
                [
                    func(state, samples[:, ii : ii + 1])
                    for ii in range(samples.shape[1])
                ]
            )

        fd_eps = bkd.flip(bkd.logspace(-13, -1, 13))
        jacobian_checker = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=1,
                nvars=_NPARAMS,
                fun=values,
                jacobian=lambda p: param_jacobian(state, p),
                bkd=bkd,
            )
        )
        errors = jacobian_checker.check_derivatives(
            sample, fd_eps=fd_eps, relative=True
        )[0]
        assert jacobian_checker.check_v_shape(errors, fd_eps).passed

        hvp_checker = DerivativeChecker(
            FunctionWithJVPFromCallable(
                nqoi=_NPARAMS,
                nvars=_NPARAMS,
                fun=lambda p: param_jacobian(state, p).T,
                jvp=lambda p, v: param_param_hvp(state, p, v),
                bkd=bkd,
            )
        )
        errors = hvp_checker.check_derivatives(sample, relative=True)[0]
        assert bkd.to_float(hvp_checker.error_ratio(errors)) <= 1e-6

    def test_shape_check_rejects_jacobian_missing_the_cost(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Control: the inner Jacobian alone, against the augmented
        value, fails the check the wrapper passes."""
        bkd = numpy_bkd
        func = _wrapped(bkd)
        inner = _inner(bkd)
        state, sample = _state_param(bkd, 9)

        def values(samples: Array) -> Array:
            return bkd.hstack(
                [
                    func(state, samples[:, ii : ii + 1])
                    for ii in range(samples.shape[1])
                ]
            )

        fd_eps = bkd.flip(bkd.logspace(-13, -1, 13))
        checker = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=1,
                nvars=_NPARAMS,
                fun=values,
                jacobian=lambda p: inner.param_jacobian(state, p),
                bkd=bkd,
            )
        )
        errors = checker.check_derivatives(
            sample, fd_eps=fd_eps, relative=True
        )[0]
        assert not checker.check_v_shape(errors, fd_eps).passed


class TestReplacesTikhonovMSE:
    def test_reproduces_tikhonov_mse(self, bkd: Backend[Array]) -> None:
        """With alpha = 1 and no weight, the wrapper around MSEFunctional
        is the former TikhonovMSEFunctional."""
        obs = bkd.asarray(_OBS_NP)
        old = TikhonovMSEFunctional(_NSTATES, _NPARAMS, bkd)
        old.set_observations(obs)
        mse = MSEFunctional(_NSTATES, _NPARAMS, bkd)
        mse.set_observations(obs)
        new = TikhonovAugmentedSteadyFunctional(mse, 1.0, bkd)
        state, param = _state_param(bkd, 11)
        vvec = bkd.asarray(
            np.random.default_rng(12).normal(0.0, 1.0, (_NPARAMS, 1))
        )
        wvec = bkd.asarray(
            np.random.default_rng(13).normal(0.0, 1.0, (_NSTATES, 1))
        )
        bkd.assert_allclose(new(state, param), old(state, param), rtol=1e-12)
        old_derivs = old.derivatives()
        new_derivs = new.derivatives()
        pairs = [
            ("state_jacobian", (state, param)),
            ("param_jacobian", (state, param)),
            ("state_state_hvp", (state, param, wvec)),
            ("state_param_hvp", (state, param, vvec)),
            ("param_state_hvp", (state, param, wvec)),
            ("param_param_hvp", (state, param, vvec)),
        ]
        for name, args in pairs:
            old_fn = getattr(old_derivs, name)
            new_fn = getattr(new_derivs, name)
            assert old_fn is not None and new_fn is not None
            bkd.assert_allclose(new_fn(*args), old_fn(*args), rtol=1e-12)


class TestConstructorValidation:
    def test_rejects_non_functional(self, numpy_bkd: NumpyBkd) -> None:
        with pytest.raises(TypeError, match="SteadyFunctionalProtocol"):
            TikhonovAugmentedSteadyFunctional(
                object(),  # type: ignore[arg-type]
                _ALPHA,
                numpy_bkd,
            )

    def test_rejects_transient_functional(self, numpy_bkd: NumpyBkd) -> None:
        """A transient functional has every steady member; its declared
        time dependence rejects it."""
        transient = EndpointFunctional(0, _NSTATES, _NPARAMS, numpy_bkd)
        with pytest.raises(TypeError, match="transient"):
            TikhonovAugmentedSteadyFunctional(
                transient,  # type: ignore[arg-type]
                _ALPHA,
                numpy_bkd,
            )

    def test_rejects_vector_functional(self, numpy_bkd: NumpyBkd) -> None:
        inner = SubsetOfStatesFunctional(
            _NSTATES, _NPARAMS, numpy_bkd.asarray([0, 1]), numpy_bkd
        )
        with pytest.raises(ValueError, match="nqoi"):
            TikhonovAugmentedSteadyFunctional(inner, _ALPHA, numpy_bkd)

    def test_rejects_bad_weight(self, numpy_bkd: NumpyBkd) -> None:
        with pytest.raises(ValueError, match="shape"):
            _wrapped(numpy_bkd, np.eye(_NPARAMS + 1))
        asymmetric = np.eye(_NPARAMS)
        asymmetric[0, 1] = 1.0
        with pytest.raises(ValueError, match="symmetric"):
            _wrapped(numpy_bkd, asymmetric)
