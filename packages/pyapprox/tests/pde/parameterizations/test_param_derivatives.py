"""Tests for the ParamDerivatives family bundle."""

import dataclasses
import pickle
from typing import Any, Generic

import numpy as np
import pytest

from pyapprox.pde.parameterizations.derivatives import (
    DenseParamJacobian,
    ParamDerivatives,
    ParamJacobianProtocol,
    dense_param_jacobian,
)
from pyapprox.util.backends.protocols import Array, Backend


def _param_jacobian(physics, state, time, params_1d):
    raise NotImplementedError


def _initial_param_jacobian(physics, params_1d):
    raise NotImplementedError


def _hvp(physics, state, time, params_1d, adj_state, vec):
    raise NotImplementedError


def _initial_param_hvp(params_1d, weight, vvec):
    raise NotImplementedError


def _bc_flux(physics, state, time, params_1d, bc_indices, normals):
    raise NotImplementedError


class TestParamDerivatives:
    def test_none_has_no_capability(self) -> None:
        d = ParamDerivatives.none()
        assert d.param_jacobian is None
        assert d.initial_param_jacobian is None
        assert d.param_param_hvp is None
        assert d.state_param_hvp is None
        assert d.param_state_hvp is None
        assert d.bc_flux_param_sensitivity is None

    def test_first_order_populates_jacobians(self) -> None:
        d = ParamDerivatives.first_order(
            _param_jacobian, _initial_param_jacobian
        )
        assert d.param_jacobian is _param_jacobian
        assert d.initial_param_jacobian is _initial_param_jacobian
        assert d.param_param_hvp is None
        assert d.state_param_hvp is None
        assert d.param_state_hvp is None
        assert d.bc_flux_param_sensitivity is None

    def test_first_order_with_bc_flux(self) -> None:
        d = ParamDerivatives.first_order(
            _param_jacobian,
            _initial_param_jacobian,
            bc_flux_param_sensitivity=_bc_flux,
        )
        assert d.bc_flux_param_sensitivity is _bc_flux

    def test_first_order_requires_both_jacobians(self) -> None:
        with pytest.raises(TypeError, match="first_order requires"):
            ParamDerivatives.first_order(_param_jacobian, None)
        with pytest.raises(TypeError, match="first_order requires"):
            ParamDerivatives.first_order(None, _initial_param_jacobian)

    def test_second_order_populates_all(self) -> None:
        d = ParamDerivatives.second_order(
            _param_jacobian,
            _initial_param_jacobian,
            _hvp,
            _hvp,
            _hvp,
            _initial_param_hvp,
        )
        assert d.param_jacobian is _param_jacobian
        assert d.initial_param_jacobian is _initial_param_jacobian
        assert d.param_param_hvp is _hvp
        assert d.state_param_hvp is _hvp
        assert d.param_state_hvp is _hvp
        assert d.initial_param_hvp is _initial_param_hvp

    def test_second_order_requires_all_hvps(self) -> None:
        with pytest.raises(TypeError, match="second_order requires"):
            ParamDerivatives.second_order(
                _param_jacobian,
                _initial_param_jacobian,
                _hvp,
                None,
                _hvp,
                _initial_param_hvp,
            )

    def test_second_order_requires_initial_param_hvp(self) -> None:
        """Zero initial-state curvature must be declared, not assumed."""
        with pytest.raises(TypeError, match="initial_param_hvp"):
            ParamDerivatives.second_order(
                _param_jacobian,
                _initial_param_jacobian,
                _hvp,
                _hvp,
                _hvp,
                None,
            )

    def test_non_callable_field_raises(self) -> None:
        with pytest.raises(TypeError, match="must be callable or None"):
            ParamDerivatives(param_jacobian=1.0)
        with pytest.raises(TypeError, match="must be callable or None"):
            ParamDerivatives(state_param_hvp="not_callable")

    def test_frozen(self) -> None:
        d = ParamDerivatives.none()
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.param_jacobian = _param_jacobian  # type: ignore[misc]


_NSTATES, _NPARAMS = 4, 3


def _matrix() -> Any:
    """Distinct entries, so a transposed product shows."""
    return np.arange(1.0, _NSTATES * _NPARAMS + 1.0).reshape(_NSTATES, _NPARAMS)


class _MatrixProducer(Generic[Array]):
    """A producer keeping its public matrix method, as a parameterization
    does; module level, so its bound method pickles."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def param_jacobian(self, state: Array, time: float, params_1d: Array) -> Array:
        return self._bkd.asarray(_matrix()) * params_1d[0]


class TestDenseParamJacobian:
    def test_actions_are_the_matrix_products(self, bkd: Backend[Array]) -> None:
        jac = DenseParamJacobian(bkd.asarray(_matrix()))
        assert isinstance(jac, ParamJacobianProtocol)
        rng = np.random.default_rng(3)
        vecs = rng.normal(size=(_NPARAMS, 2))
        adjs = rng.normal(size=(_NSTATES, 2))
        bkd.assert_allclose(jac.matrix(), bkd.asarray(_matrix()))
        bkd.assert_allclose(
            jac.apply(bkd.asarray(vecs)), bkd.asarray(_matrix() @ vecs), rtol=1e-14
        )
        bkd.assert_allclose(
            jac.apply_transpose(bkd.asarray(adjs)),
            bkd.asarray(_matrix().T @ adjs),
            rtol=1e-14,
        )

    def test_refuses_a_vector(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="shape"):
            DenseParamJacobian(bkd.ones((_NSTATES,)))

    def test_wraps_a_matrix_function(self, bkd: Backend[Array]) -> None:
        fn = dense_param_jacobian(_MatrixProducer(bkd).param_jacobian)
        jac = fn(bkd.zeros((_NSTATES,)), 0.0, bkd.asarray(np.array([2.0, 0.0, 0.0])))
        assert isinstance(jac, DenseParamJacobian)
        bkd.assert_allclose(jac.matrix(), bkd.asarray(2.0 * _matrix()))

    def test_refuses_a_non_callable(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="callable"):
            dense_param_jacobian(bkd.ones((_NSTATES, _NPARAMS)))  # type: ignore[arg-type]

    def test_wrapped_function_pickles(self, numpy_bkd: Backend[Array]) -> None:
        fn = dense_param_jacobian(_MatrixProducer(numpy_bkd).param_jacobian)
        restored = pickle.loads(pickle.dumps(fn))
        args = (
            numpy_bkd.zeros((_NSTATES,)),
            0.0,
            numpy_bkd.asarray(np.array([3.0, 0.0, 0.0])),
        )
        numpy_bkd.assert_allclose(restored(*args).matrix(), fn(*args).matrix())

    def test_actions_keep_the_graph(self, torch_bkd: Backend[Array]) -> None:
        """A gradient flows through ``apply`` and ``apply_transpose`` to
        the matrix: d/dM of sum(M 1) + sum(M^T 1) is 2 in every entry."""
        import torch

        matrix = torch.tensor(_matrix(), dtype=torch.float64, requires_grad=True)
        jac = DenseParamJacobian(matrix)
        ones_p = torch.ones((_NPARAMS, 1), dtype=torch.float64)
        ones_n = torch.ones((_NSTATES, 1), dtype=torch.float64)
        total = jac.apply(ones_p).sum() + jac.apply_transpose(ones_n).sum()
        (grad,) = torch.autograd.grad(total, matrix)
        assert torch.allclose(grad, torch.full_like(matrix, 2.0))
