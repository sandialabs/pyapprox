"""Derivatives of the constrained residual from the raw ones."""

import pickle
from typing import Any

import numpy as np
from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.boundary import DirichletConstraintSet
from pyapprox.pde.galerkin.boundary import DirectDirichletBC
from pyapprox.pde.models.galerkin.constrained_derivatives import (
    constrain_param_derivatives,
    constrain_state_derivatives,
    constrain_state_state_hvp,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.util.backends.protocols import Array, Backend

_NSTATES, _NPARAMS = 5, 2
_ESSENTIAL = [0, 3]


def _constraints(bkd: Backend[Array]) -> DirichletConstraintSet[Array]:
    return DirichletConstraintSet(
        [DirectDirichletBC(_ESSENTIAL, [1.0, 2.0], bkd)], _NSTATES, bkd
    )


class _Raw:
    """Raw derivatives whose outputs expose what they were given."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def param_jacobian(self, state: Any, time: float, params: Any) -> Any:
        return self._bkd.ones((_NSTATES, _NPARAMS))

    def param_hvp(
        self, state: Any, time: float, params: Any, adj: Any, vec: Any
    ) -> Any:
        # State-shaped output: echoes the weight it received.
        return adj

    def initial_param_hvp(self, params: Any, weight: Any, vvec: Any) -> Any:
        return weight

    def state_state_hvp(
        self, state: Any, adj: Any, wvec: Any, time: float
    ) -> Any:
        return adj


def _ones(bkd: Backend[Array]) -> Array:
    return bkd.ones((_NSTATES,))


def _zeroed_at_essential(bkd: Backend[Array]) -> Array:
    values = np.ones(_NSTATES)
    values[_ESSENTIAL] = 0.0
    return bkd.asarray(values)


class TestConstrainedDerivatives:
    def test_param_jacobian_rows_zeroed(self, bkd: Backend[Array]) -> None:
        raw = _Raw(bkd)
        derivs = constrain_param_derivatives(
            ParamDerivatives(param_jacobian=raw.param_jacobian),
            _constraints(bkd),
        )
        jac = derivs.param_jacobian
        assert jac is not None
        expected = np.ones((_NSTATES, _NPARAMS))
        expected[_ESSENTIAL] = 0.0
        bkd.assert_allclose(
            jac(_ones(bkd), 0.0, bkd.ones((_NPARAMS,))), bkd.asarray(expected)
        )

    def test_contraction_weights_zeroed(self, bkd: Backend[Array]) -> None:
        raw = _Raw(bkd)
        derivs = constrain_param_derivatives(
            ParamDerivatives(
                initial_param_hvp=raw.initial_param_hvp,
                param_param_hvp=raw.param_hvp,
                state_param_hvp=raw.param_hvp,
                param_state_hvp=raw.param_hvp,
            ),
            _constraints(bkd),
        )
        expected = _zeroed_at_essential(bkd)
        params = bkd.ones((_NPARAMS,))
        for hvp in (
            derivs.param_param_hvp,
            derivs.state_param_hvp,
            derivs.param_state_hvp,
        ):
            assert hvp is not None
            bkd.assert_allclose(
                hvp(_ones(bkd), 0.0, params, _ones(bkd), params), expected
            )
        initial = derivs.initial_param_hvp
        assert initial is not None
        bkd.assert_allclose(initial(params, _ones(bkd), params), expected)

    def test_state_state_hvp_weight_zeroed(self, bkd: Backend[Array]) -> None:
        hvp = constrain_state_state_hvp(
            _Raw(bkd).state_state_hvp, _constraints(bkd)
        )
        bkd.assert_allclose(
            hvp(_ones(bkd), _ones(bkd), _ones(bkd), 0.0),
            _zeroed_at_essential(bkd),
        )

    def test_absent_stays_absent(self, bkd: Backend[Array]) -> None:
        derivs = constrain_param_derivatives(
            ParamDerivatives(), _constraints(bkd)
        )
        assert derivs.param_jacobian is None
        assert derivs.param_param_hvp is None
        assert (
            constrain_state_derivatives(
                StateDerivatives.none(), _constraints(bkd)
            ).state_state_hvp
            is None
        )

    def test_initial_param_jacobian_passes_through(
        self, bkd: Backend[Array]
    ) -> None:
        raw = _Raw(bkd)
        derivs = constrain_param_derivatives(
            ParamDerivatives(initial_param_jacobian=raw.initial_param_hvp),
            _constraints(bkd),
        )
        assert derivs.initial_param_jacobian == raw.initial_param_hvp

    def test_pickles(self, numpy_bkd: Backend[Array]) -> None:
        raw = _Raw(numpy_bkd)
        derivs = constrain_param_derivatives(
            ParamDerivatives(param_jacobian=raw.param_jacobian),
            _constraints(numpy_bkd),
        )
        restored = pickle.loads(pickle.dumps(derivs))
        jac, restored_jac = derivs.param_jacobian, restored.param_jacobian
        assert jac is not None and restored_jac is not None
        args = (_ones(numpy_bkd), 0.0, numpy_bkd.ones((_NPARAMS,)))
        numpy_bkd.assert_allclose(restored_jac(*args), jac(*args))
