"""Tests for linearization points and labeled derivatives."""

import dataclasses

import pytest

from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    JacobianSnapshotProtocol,
    LinearizationPoint,
    LinearizedOperator,
    OperatorProperties,
    SpaceDescriptor,
)
from tests._helpers.operator_stacks import StackCase, random_matrix, spd_matrix

_U = SpaceDescriptor("u", 5)
_P = SpaceDescriptor("p", 3)
_SPD = OperatorProperties(symmetric=True, spd=True)


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
            LinearizationPoint(0, [1, 2])
        with pytest.raises(ValueError, match="nonnegative"):
            LinearizationPoint(-1)
        with pytest.raises(ValueError, match="cannot be provisional"):
            LinearizationPoint(0, None, provisional=True)


class TestLinearizedOperator:
    def test_carries_its_point_and_delegates(self, case: StackCase) -> None:
        a = random_matrix(3, 5, 0)
        stored = case.build(a, _U, _P)
        point = LinearizationPoint(1, "iterate-2", provisional=True)
        snapshot = LinearizedOperator(stored, point)
        assert isinstance(snapshot, JacobianSnapshotProtocol)
        assert isinstance(snapshot, ActionOperatorProtocol)
        assert snapshot.linearization() is point
        assert snapshot.domain() == _U and snapshot.range() == _P
        assert snapshot.format() is stored.format()
        x, y = case.vec(_U, 2, 1), case.vec(_P, 2, 2)
        case.assert_vector(snapshot.apply(x), case.array(stored.apply(x)))
        case.assert_vector(
            snapshot.apply_transpose(y), case.array(stored.apply_transpose(y))
        )

    def test_keeps_properties_and_solves(self, case: StackCase) -> None:
        """A labeled derivative is still factorable in its format."""
        stored = case.build(spd_matrix(5, 3), _U, properties=_SPD)
        snapshot = LinearizedOperator(stored, LinearizationPoint(0))
        assert snapshot.properties() == _SPD
        b = case.vec(_U, 1, 4)
        x, info = case.direct_solver().setup(snapshot).solve(b)
        assert info.converged
        case.assert_vector(stored.apply(x), case.array(b), rtol=1e-10)

    def test_a_plain_operator_is_not_a_snapshot(self, case: StackCase) -> None:
        assert not isinstance(case.identity(_U), JacobianSnapshotProtocol)
