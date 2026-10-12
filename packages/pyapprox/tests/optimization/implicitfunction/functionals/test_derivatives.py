"""Tests for the steady functional derivative bundle."""

import dataclasses
import pickle
from typing import Any

import pytest

from pyapprox.optimization.implicitfunction.functionals.derivatives import (
    SteadyFunctionalDerivatives,
)


def _jacobian(state: Any, param: Any) -> Any:
    return state


def _hvp(state: Any, param: Any, vec: Any) -> Any:
    return vec


class TestSteadyFunctionalDerivatives:
    def test_none_has_no_fields(self) -> None:
        """A value-only functional."""
        bundle: SteadyFunctionalDerivatives[Any] = (
            SteadyFunctionalDerivatives.none()
        )
        for field in dataclasses.fields(bundle):
            assert getattr(bundle, field.name) is None

    def test_first_order_carries_only_the_jacobians(self) -> None:
        bundle: SteadyFunctionalDerivatives[Any] = (
            SteadyFunctionalDerivatives.first_order(_jacobian, _jacobian)
        )
        assert bundle.state_jacobian is _jacobian
        assert bundle.param_jacobian is _jacobian
        assert bundle.state_state_hvp is None

    def test_second_order_carries_everything(self) -> None:
        bundle: SteadyFunctionalDerivatives[Any] = (
            SteadyFunctionalDerivatives.second_order(
                _jacobian, _jacobian, _hvp, _hvp, _hvp, _hvp
            )
        )
        for field in dataclasses.fields(bundle):
            assert getattr(bundle, field.name) is not None

    def test_rejects_one_jacobian(self) -> None:
        with pytest.raises(TypeError, match="together"):
            SteadyFunctionalDerivatives(param_jacobian=_jacobian)

    def test_hvps_require_the_jacobians(self) -> None:
        with pytest.raises(TypeError, match="require"):
            SteadyFunctionalDerivatives(
                state_state_hvp=_hvp,
                state_param_hvp=_hvp,
                param_state_hvp=_hvp,
                param_param_hvp=_hvp,
            )

    def test_rejects_a_partial_hvp_set(self) -> None:
        with pytest.raises(TypeError, match="four HVP fields"):
            SteadyFunctionalDerivatives(
                state_jacobian=_jacobian,
                param_jacobian=_jacobian,
                state_state_hvp=_hvp,
                param_param_hvp=_hvp,
            )

    def test_rejects_a_non_callable_field(self) -> None:
        with pytest.raises(TypeError, match="state_state_hvp"):
            SteadyFunctionalDerivatives(
                state_state_hvp=1.0,  # type: ignore[arg-type]
            )

    def test_is_frozen(self) -> None:
        bundle: SteadyFunctionalDerivatives[Any] = (
            SteadyFunctionalDerivatives.none()
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            bundle.state_state_hvp = _hvp  # type: ignore[misc]

    def test_pickles(self) -> None:
        bundle: SteadyFunctionalDerivatives[Any] = (
            SteadyFunctionalDerivatives.second_order(
                _jacobian, _jacobian, _hvp, _hvp, _hvp, _hvp
            )
        )
        assert pickle.loads(pickle.dumps(bundle)) == bundle
