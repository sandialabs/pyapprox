"""Tests for reuse of solver setups across equal linearizations."""

import numpy as np
import pytest

from pyapprox.util.linalg.operators import (
    CachingSolver,
    LinearizationPoint,
    LinearizedOperator,
    LinearSolverProtocol,
    ReuseKey,
    SpaceDescriptor,
    reuse_key,
)
from tests._helpers.linalg.stacks import A, F, TypedStackCase, V, random_matrix

_U = SpaceDescriptor("u", 5)


def _snapshot(
    case: TypedStackCase[V, F, A],
    seed: int,
    point: LinearizationPoint,
    space: SpaceDescriptor = _U,
) -> LinearizedOperator[V, F]:
    """A well-conditioned 5x5 operator labeled with ``point``."""
    matrix = random_matrix(5, 5, seed) + 5.0 * np.eye(5)
    return LinearizedOperator(case.build(matrix, space), point)


class TestReuseKey:
    def test_key_of_a_snapshot_and_of_a_plain_operator(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        point = LinearizationPoint(0, "a")
        assert reuse_key(_snapshot(case, 0, point)) == ReuseKey(point, _U, _U)
        assert reuse_key(case.identity(_U)) is None


class TestCachingSolver:
    def test_is_a_solver_and_forwards_capabilities(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        cache = CachingSolver(case.direct_solver(), capacity=1)
        assert isinstance(cache, LinearSolverProtocol)
        assert cache.capabilities() == case.direct_solver().capabilities()

    def test_equal_keys_share_one_setup(self, case: TypedStackCase[V, F, A]) -> None:
        cache = CachingSolver(case.direct_solver(), capacity=1)
        point = LinearizationPoint(0, ("step", 3))
        op = _snapshot(case, 1, point)
        first = cache.setup(op)
        second = cache.setup(_snapshot(case, 1, point))
        assert second is first
        assert (cache.setups(), cache.hits()) == (1, 1)
        b = case.vec(_U, 1, 2)
        case.assert_vector(op.apply(second.apply(b)), case.array(b), rtol=1e-10)

    @pytest.mark.parametrize(
        "second",
        [
            LinearizationPoint(0, ("step", 4)),
            LinearizationPoint(1, ("step", 3)),
            LinearizationPoint(0, ("step", 3), provisional=True),
        ],
        ids=["other-point", "new-parameters", "provisional"],
    )
    def test_unequal_keys_set_up_again(
        self, case: TypedStackCase[V, F, A], second: LinearizationPoint
    ) -> None:
        cache = CachingSolver(case.direct_solver(), capacity=4)
        cache.setup(_snapshot(case, 3, LinearizationPoint(0, ("step", 3))))
        cache.setup(_snapshot(case, 3, second))
        assert (cache.setups(), cache.hits()) == (2, 0)

    def test_final_point_never_reuses_a_provisional_setup(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        """The last iterate of an outer solve and its result can share an
        identity; the result must still be set up afresh."""
        cache = CachingSolver(case.direct_solver(), capacity=4)
        cache.setup(_snapshot(case, 4, LinearizationPoint(0, "x", provisional=True)))
        cache.setup(_snapshot(case, 4, LinearizationPoint(0, "x", provisional=True)))
        assert cache.hits() == 1
        cache.setup(_snapshot(case, 4, LinearizationPoint(0, "x")))
        assert (cache.setups(), cache.hits()) == (2, 1)

    def test_point_independent_derivative_is_reused_until_parameters_change(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        cache = CachingSolver(case.direct_solver(), capacity=1)
        for _ in range(5):
            cache.setup(_snapshot(case, 5, LinearizationPoint(0)))
        assert (cache.setups(), cache.hits()) == (1, 4)
        cache.setup(_snapshot(case, 6, LinearizationPoint(1)))
        assert cache.setups() == 2

    def test_rebuilt_space_sets_up_again(self, case: TypedStackCase[V, F, A]) -> None:
        cache = CachingSolver(case.direct_solver(), capacity=4)
        point = LinearizationPoint(0)
        cache.setup(_snapshot(case, 7, point))
        cache.setup(_snapshot(case, 7, point, SpaceDescriptor("u", 5, version=1)))
        assert cache.setups() == 2

    def test_operators_without_a_point_are_never_cached(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        cache = CachingSolver(case.direct_solver(), capacity=4)
        op = case.build(random_matrix(5, 5, 8) + 5.0 * np.eye(5), _U)
        cache.setup(op)
        cache.setup(op)
        assert (cache.setups(), cache.hits()) == (2, 0)

    @pytest.mark.parametrize("capacity, setups", [(1, 3), (2, 2)])
    def test_capacity_evicts_least_recently_used(
        self, case: TypedStackCase[V, F, A], capacity: int, setups: int
    ) -> None:
        """A, B, A: a single slot loses A to B; two slots keep both."""
        cache = CachingSolver(case.direct_solver(), capacity=capacity)
        a, b = LinearizationPoint(0, "a"), LinearizationPoint(0, "b")
        for point, seed in ((a, 9), (b, 10), (a, 9)):
            cache.setup(_snapshot(case, seed, point))
        assert cache.setups() == setups

    def test_clear_drops_kept_handles(self, case: TypedStackCase[V, F, A]) -> None:
        cache = CachingSolver(case.direct_solver(), capacity=1)
        point = LinearizationPoint(0)
        cache.setup(_snapshot(case, 11, point))
        cache.clear()
        cache.setup(_snapshot(case, 11, point))
        assert cache.setups() == 2

    def test_a_keyed_setup_builds_the_operator_only_on_a_miss(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        """Three keyed setups under one key build once and share the
        handle with a plain setup of an equal snapshot; a builder whose
        operator carries another key is refused."""
        cache = CachingSolver(case.direct_solver(), capacity=2)
        point = LinearizationPoint(0, "k")
        key = ReuseKey(point, _U, _U)
        builds = []

        def build() -> LinearizedOperator[V, F]:
            builds.append(point)
            return _snapshot(case, 12, point)

        handles = [cache.setup_keyed(key, build) for _ in range(3)]
        assert len(builds) == 1
        assert all(handle is handles[0] for handle in handles)
        assert cache.setup(_snapshot(case, 12, point)) is handles[0]
        assert (cache.setups(), cache.hits()) == (1, 3)
        other = ReuseKey(LinearizationPoint(0, "other"), _U, _U)
        with pytest.raises(ValueError, match="is not the key"):
            cache.setup_keyed(other, build)

    def test_rejects_zero_capacity(self, case: TypedStackCase[V, F, A]) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            CachingSolver(case.direct_solver(), capacity=0)
