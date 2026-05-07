"""Tests for PointManager."""

import numpy as np

from pyapprox.surrogates.sparsegrids.basis.hierarchical_basis_1d import (
    HierarchicalBasis1D,
)
from pyapprox.surrogates.sparsegrids.basis.hierarchical_basis_nd import (
    HierarchicalBasisND,
)
from pyapprox.surrogates.sparsegrids.hierarchical.point_manager import (
    PointManager,
)


def _make_basis_2d(bkd):
    b1 = HierarchicalBasis1D(bkd, boundary_mode="include")
    b2 = HierarchicalBasis1D(bkd, boundary_mode="include")
    return HierarchicalBasisND(bkd, [b1, b2])


class TestPointManager:
    def test_register_and_get_key(self, bkd):
        basis = _make_basis_2d(bkd)
        pm = PointManager(bkd, basis)
        key = ((0, 0), (1, 1))
        pid = pm.register_point(key)
        assert pid == 0
        assert pm.get_key(pid) == key
        assert pm.get_config_idx(pid) == ()

    def test_idempotent_registration(self, bkd):
        basis = _make_basis_2d(bkd)
        pm = PointManager(bkd, basis)
        key = ((0, 0), (1, 1))
        pid1 = pm.register_point(key)
        pid2 = pm.register_point(key)
        assert pid1 == pid2
        assert pm.n_points() == 1

    def test_pending_samples_roundtrip(self, bkd):
        basis = _make_basis_2d(bkd)
        pm = PointManager(bkd, basis)
        key = ((0, 0), (1, 1))
        pm.register_point(key)
        ids, coords = pm.get_pending_samples()
        assert ids == [0]
        assert coords is not None
        bkd.assert_allclose(coords, bkd.asarray([[0.5], [0.5]]))

    def test_set_values_and_surpluses(self, bkd):
        basis = _make_basis_2d(bkd)
        pm = PointManager(bkd, basis)
        key = ((0, 0), (1, 1))
        pm.register_point(key)

        values = bkd.asarray([[3.0]])
        preds = bkd.asarray([[1.0]])
        pm.set_values_and_surpluses([0], values, preds)

        assert pm.is_evaluated(0)
        assert pm.n_pending() == 0
        bkd.assert_allclose(pm.get_value(0), bkd.asarray([3.0]))
        bkd.assert_allclose(pm.get_surplus(0), bkd.asarray([2.0]))

    def test_active_redundant_disjoint(self, bkd):
        basis = _make_basis_2d(bkd)
        pm = PointManager(bkd, basis)
        pm.register_point(((0, 0), (1, 1)))
        pm.register_point(((1, 0), (0, 1)))
        pm.register_point(((1, 0), (2, 1)))

        pm.mark_active(0)
        pm.mark_redundant(1)
        pm.mark_active(2)

        assert pm.is_active(0)
        assert not pm.is_redundant(0)
        assert pm.is_redundant(1)
        assert not pm.is_active(1)
        assert pm.is_active(2)

        assert pm.get_subspace_active_ids((0, 0)) == {0}
        assert pm.get_subspace_redundant_ids((1, 0)) == {1}
        assert pm.get_subspace_active_ids((1, 0)) == {2}

    def test_refinement_ledger(self, bkd):
        basis = _make_basis_2d(bkd)
        pm = PointManager(bkd, basis)
        pm.register_point(((0, 0), (1, 1)))
        pm.mark_active(0)

        assert not pm.is_refined(0, 0)
        assert not pm.is_refined(0, 1)
        assert not pm.is_point_resolved(0, 2)

        pm.mark_refined(0, 0)
        assert pm.is_refined(0, 0)
        assert not pm.is_point_resolved(0, 2)

        pm.mark_refined(0, 1)
        assert pm.is_point_resolved(0, 2)

    def test_redundant_is_resolved(self, bkd):
        basis = _make_basis_2d(bkd)
        pm = PointManager(bkd, basis)
        pm.register_point(((0, 0), (1, 1)))
        pm.mark_redundant(0)
        assert pm.is_point_resolved(0, 2)

    def test_subspace_complete(self, bkd):
        """Subspace (1,0) is complete when all points in backward neighbor
        (0,0) are resolved."""
        basis = _make_basis_2d(bkd)
        pm = PointManager(bkd, basis)

        # Register root point and mark active but not yet refined
        pm.register_point(((0, 0), (1, 1)))
        pm.mark_active(0)

        assert not pm.is_subspace_complete((1, 0), 2)

        pm.mark_refined(0, 0)
        pm.mark_refined(0, 1)
        assert pm.is_subspace_complete((1, 0), 2)

    def test_subspaces_affected_by(self, bkd):
        basis = _make_basis_2d(bkd)
        pm = PointManager(bkd, basis)
        pm.register_point(((0, 0), (1, 1)))

        affected = pm.subspaces_affected_by(0, 2)
        assert affected == {(1, 0), (0, 1)}

    def test_subspaces_affected_by_nonroot(self, bkd):
        basis = _make_basis_2d(bkd)
        pm = PointManager(bkd, basis)
        pm.register_point(((1, 2), (0, 3)))

        affected = pm.subspaces_affected_by(0, 2)
        assert affected == {(2, 2), (1, 3)}

    def test_multi_fidelity_registration(self, bkd):
        basis = _make_basis_2d(bkd)
        pm = PointManager(bkd, basis)
        key0 = ((0, 0), (1, 1))
        key1 = ((1, 0), (0, 1))

        pm.register_point(key0, config_idx=(0,))
        pm.register_point(key1, config_idx=(1,))

        assert pm.get_config_idx(0) == (0,)
        assert pm.get_config_idx(1) == (1,)

        by_config = pm.points_by_config()
        assert by_config[(0,)] == [0]
        assert by_config[(1,)] == [1]

    def test_get_pending_samples_filters_by_config(self, bkd):
        basis = _make_basis_2d(bkd)
        pm = PointManager(bkd, basis)
        pm.register_point(((0, 0), (1, 1)), config_idx=(0,))
        pm.register_point(((1, 0), (0, 1)), config_idx=(1,))

        ids_0, coords_0 = pm.get_pending_samples(config_idx=(0,))
        assert ids_0 == [0]
        ids_1, coords_1 = pm.get_pending_samples(config_idx=(1,))
        assert ids_1 == [1]
        ids_empty, coords_empty = pm.get_pending_samples(config_idx=(2,))
        assert ids_empty == []
        assert coords_empty is None

    def test_iter_evaluated(self, bkd):
        basis = _make_basis_2d(bkd)
        pm = PointManager(bkd, basis)
        pm.register_point(((0, 0), (1, 1)))
        pm.register_point(((1, 0), (0, 1)))

        vals = bkd.asarray([[1.0, 2.0]])
        preds = bkd.asarray([[0.0, 0.0]])
        pm.set_values_and_surpluses([0, 1], vals, preds)

        items = list(pm.iter_evaluated())
        assert len(items) == 2
        assert items[0][0] == 0
        assert items[1][0] == 1
        bkd.assert_allclose(items[0][2], bkd.asarray([1.0]))
        bkd.assert_allclose(items[1][2], bkd.asarray([2.0]))

    def test_collect_coordinates(self, bkd):
        basis = _make_basis_2d(bkd)
        pm = PointManager(bkd, basis)
        pm.register_point(((0, 0), (1, 1)))
        pm.register_point(((1, 0), (0, 1)))

        coords = pm.collect_coordinates("all")
        assert coords is not None
        assert bkd.to_numpy(coords).shape == (2, 2)

        coords_pending = pm.collect_coordinates("pending")
        assert coords_pending is not None
        assert bkd.to_numpy(coords_pending).shape == (2, 2)

        coords_eval = pm.collect_coordinates("evaluated")
        assert coords_eval is None

    def test_n_points_and_n_evaluated(self, bkd):
        basis = _make_basis_2d(bkd)
        pm = PointManager(bkd, basis)
        assert pm.n_points() == 0
        assert pm.n_evaluated() == 0

        pm.register_point(((0, 0), (1, 1)))
        assert pm.n_points() == 1
        assert pm.n_evaluated() == 0
        assert pm.n_pending() == 1

        vals = bkd.asarray([[5.0]])
        preds = bkd.asarray([[0.0]])
        pm.set_values_and_surpluses([0], vals, preds)
        assert pm.n_evaluated() == 1
        assert pm.n_pending() == 0

    def test_get_subspace_point_ids(self, bkd):
        basis = _make_basis_2d(bkd)
        pm = PointManager(bkd, basis)
        pm.register_point(((0, 0), (1, 1)))
        pm.register_point(((1, 0), (0, 1)))
        pm.register_point(((1, 0), (2, 1)))

        assert pm.get_subspace_point_ids((0, 0)) == {0}
        assert pm.get_subspace_point_ids((1, 0)) == {1, 2}
        assert pm.get_subspace_point_ids((2, 2)) == set()
