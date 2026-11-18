"""Tests for linearization points and labeled derivatives."""

import dataclasses

import pytest

from pyapprox.util.linalg.operators import (
    LinearizationPoint,
)
from tests._helpers.linalg.operator_suites.linearization import LinearizedOperatorSuite


class TestLinearizationPoint:
    def test_is_immutable_and_hashable(self) -> None:
        point = LinearizationPoint(2, ("step", 7))
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(point, "parameters_version", 3)
        assert len({point, LinearizationPoint(2, ("step", 7))}) == 1

    def test_equality_distinguishes_every_field(self) -> None:
        point = LinearizationPoint(2, ("step", 7))
        assert point != LinearizationPoint(3, ("step", 7))
        assert point != LinearizationPoint(2, ("step", 8))
        assert point != LinearizationPoint(2, ("step", 7), provisional=True)

    def test_point_independent_default(self) -> None:
        """A derivative of a map linear in u is the same at every point."""
        point = LinearizationPoint(0)
        assert point.point is None and not point.provisional

    def test_rejects_invalid(self) -> None:
        with pytest.raises(TypeError, match="hashable"):
            LinearizationPoint(0, [1, 2])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="nonnegative"):
            LinearizationPoint(-1)
        with pytest.raises(ValueError, match="cannot be provisional"):
            LinearizationPoint(0, None, provisional=True)


class TestLinearizedOperator(LinearizedOperatorSuite):
    pass
