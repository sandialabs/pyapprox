"""Tests for the steady state-equation derivative bundle."""

import dataclasses
import pickle
from typing import Any

import pytest

from pyapprox.optimization.implicitfunction.state_equations.derivatives import (
    SteadyStateEquationDerivatives,
)


def _hvp(state: Any, param: Any, adj_state: Any, vec: Any) -> Any:
    return vec


class TestSteadyStateEquationDerivatives:
    def test_none_has_no_fields(self) -> None:
        bundle: SteadyStateEquationDerivatives[Any] = (
            SteadyStateEquationDerivatives.none()
        )
        for field in dataclasses.fields(bundle):
            assert getattr(bundle, field.name) is None

    def test_second_order_carries_all_four_hvps(self) -> None:
        bundle: SteadyStateEquationDerivatives[Any] = (
            SteadyStateEquationDerivatives.second_order(_hvp, _hvp, _hvp, _hvp)
        )
        for field in dataclasses.fields(bundle):
            assert getattr(bundle, field.name) is _hvp

    def test_rejects_a_partial_hvp_set(self) -> None:
        with pytest.raises(TypeError, match="together"):
            SteadyStateEquationDerivatives(param_param_hvp=_hvp)

    def test_rejects_a_non_callable_field(self) -> None:
        with pytest.raises(TypeError, match="state_param_hvp"):
            SteadyStateEquationDerivatives(
                state_param_hvp="x",  # type: ignore[arg-type]
            )

    def test_is_frozen(self) -> None:
        bundle: SteadyStateEquationDerivatives[Any] = (
            SteadyStateEquationDerivatives.none()
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            bundle.state_state_hvp = _hvp  # type: ignore[misc]

    def test_pickles(self) -> None:
        bundle: SteadyStateEquationDerivatives[Any] = (
            SteadyStateEquationDerivatives.second_order(_hvp, _hvp, _hvp, _hvp)
        )
        assert pickle.loads(pickle.dumps(bundle)) == bundle
