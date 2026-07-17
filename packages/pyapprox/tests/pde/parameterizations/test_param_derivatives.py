"""Tests for the ParamDerivatives family bundle."""

import dataclasses

import pytest
from pyapprox.pde.parameterizations.derivatives import (
    ParamDerivatives,
)


def _param_jacobian(physics, state, time, params_1d):
    raise NotImplementedError


def _initial_param_jacobian(physics, params_1d):
    raise NotImplementedError


def _hvp(physics, state, time, params_1d, adj_state, vec):
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
        )
        assert d.param_jacobian is _param_jacobian
        assert d.initial_param_jacobian is _initial_param_jacobian
        assert d.param_param_hvp is _hvp
        assert d.state_param_hvp is _hvp
        assert d.param_state_hvp is _hvp

    def test_second_order_requires_all_hvps(self) -> None:
        with pytest.raises(TypeError, match="second_order requires"):
            ParamDerivatives.second_order(
                _param_jacobian,
                _initial_param_jacobian,
                _hvp,
                None,
                _hvp,
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
