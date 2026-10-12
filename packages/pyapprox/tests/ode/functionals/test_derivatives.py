"""Tests for the transient functional derivative bundle."""

import dataclasses
import pickle
from typing import Any

import pytest

from pyapprox.ode.functionals.derivatives import (
    TransientFunctionalDerivatives,
)


def _step_hvp(sol: Any, param: Any, time_idx: int, vec: Any) -> Any:
    return vec


def _param_hvp(sol: Any, param: Any, vec: Any) -> Any:
    return vec


def _action(sol: Any, param: Any, time_idx: int, wmat: Any) -> Any:
    return wmat


def _row(qoi_idx: int) -> Any:
    return qoi_idx


class TestTransientFunctionalDerivatives:
    def test_none_has_no_fields(self) -> None:
        bundle: TransientFunctionalDerivatives[Any] = (
            TransientFunctionalDerivatives.none()
        )
        for field in dataclasses.fields(bundle):
            assert getattr(bundle, field.name) is None

    def test_first_order_carries_only_the_extras(self) -> None:
        bundle: TransientFunctionalDerivatives[Any] = (
            TransientFunctionalDerivatives.first_order(
                apply_state_jacobian=_action, row_functional=_row
            )
        )
        assert bundle.apply_state_jacobian is _action
        assert bundle.row_functional is _row
        assert bundle.state_state_hvp is None
        assert bundle.param_param_hvp is None

    def test_second_order_carries_all_four_hvps(self) -> None:
        bundle: TransientFunctionalDerivatives[Any] = (
            TransientFunctionalDerivatives.second_order(
                _step_hvp, _step_hvp, _step_hvp, _param_hvp
            )
        )
        assert bundle.state_state_hvp is _step_hvp
        assert bundle.state_param_hvp is _step_hvp
        assert bundle.param_state_hvp is _step_hvp
        assert bundle.param_param_hvp is _param_hvp
        assert bundle.apply_state_jacobian is None

    def test_rejects_a_partial_hvp_set(self) -> None:
        with pytest.raises(TypeError, match="together"):
            TransientFunctionalDerivatives(state_state_hvp=_step_hvp)

    def test_rejects_a_non_callable_field(self) -> None:
        with pytest.raises(TypeError, match="row_functional"):
            TransientFunctionalDerivatives(
                row_functional=3,  # type: ignore[arg-type]
            )

    def test_is_frozen(self) -> None:
        bundle: TransientFunctionalDerivatives[Any] = (
            TransientFunctionalDerivatives.none()
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            bundle.row_functional = _row  # type: ignore[misc]

    def test_pickles(self) -> None:
        bundle: TransientFunctionalDerivatives[Any] = (
            TransientFunctionalDerivatives.second_order(
                _step_hvp,
                _step_hvp,
                _step_hvp,
                _param_hvp,
                apply_state_jacobian=_action,
            )
        )
        assert pickle.loads(pickle.dumps(bundle)) == bundle
