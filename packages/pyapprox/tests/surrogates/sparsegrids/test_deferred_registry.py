"""Tests for DeferredRefinementRegistry."""

from pyapprox.surrogates.sparsegrids.hierarchical.deferred_registry import (
    DeferredRefinementRegistry,
)


class TestDeferredRefinementRegistry:
    def test_single_blocker_defer_and_release(self):
        reg = DeferredRefinementRegistry()
        reg.defer(
            point_id=0,
            direction=1,
            target_subspace=(1, 1),
            blockers={(0, 1)},
        )
        assert reg.n_deferred() == 1
        assert not reg.empty()

        released = reg.notify_complete((0, 1))
        assert len(released) == 1
        assert released[0].point_id == 0
        assert released[0].direction == 1
        assert released[0].target_subspace == (1, 1)
        assert reg.empty()

    def test_multi_blocker_not_released_until_all_complete(self):
        reg = DeferredRefinementRegistry()
        reg.defer(
            point_id=5,
            direction=0,
            target_subspace=(2, 1),
            blockers={(1, 1), (2, 0)},
        )
        assert reg.n_deferred() == 1

        released = reg.notify_complete((1, 1))
        assert len(released) == 0
        assert reg.n_deferred() == 1

        released = reg.notify_complete((2, 0))
        assert len(released) == 1
        assert released[0].point_id == 5
        assert reg.empty()

    def test_notify_unknown_subspace_is_noop(self):
        reg = DeferredRefinementRegistry()
        reg.defer(
            point_id=0,
            direction=0,
            target_subspace=(1, 0),
            blockers={(0, 0)},
        )
        released = reg.notify_complete((99, 99))
        assert len(released) == 0
        assert reg.n_deferred() == 1

    def test_multiple_tasks_same_blocker(self):
        reg = DeferredRefinementRegistry()
        reg.defer(0, 0, (1, 1), blockers={(0, 1)})
        reg.defer(1, 1, (1, 1), blockers={(0, 1)})
        assert reg.n_deferred() == 2

        released = reg.notify_complete((0, 1))
        assert len(released) == 2
        assert {t.point_id for t in released} == {0, 1}
        assert reg.empty()

    def test_by_blocker(self):
        reg = DeferredRefinementRegistry()
        reg.defer(0, 0, (1, 1), blockers={(0, 1), (1, 0)})
        reg.defer(1, 1, (2, 0), blockers={(1, 0)})

        tasks_01 = reg.by_blocker((0, 1))
        assert len(tasks_01) == 1
        assert tasks_01[0].point_id == 0

        tasks_10 = reg.by_blocker((1, 0))
        assert len(tasks_10) == 2

    def test_empty_registry(self):
        reg = DeferredRefinementRegistry()
        assert reg.empty()
        assert reg.n_deferred() == 0
        assert reg.notify_complete((0, 0)) == []
