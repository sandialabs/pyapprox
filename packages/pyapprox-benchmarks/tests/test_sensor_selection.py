"""Tests for greedy maximin sensor and QoI location selection."""

import numpy as np
import pytest

from pyapprox_benchmarks.expdesign.sensor_selection import (
    get_feasible_mask,
    select_maximin_locations,
    select_qoi_locations,
)


class TestSelectMaximinLocations:
    def test_determinism(self):
        rng = np.random.default_rng(0)
        nodes = rng.uniform(size=(2, 100))
        mask = np.ones(100, dtype=bool)
        idx1 = select_maximin_locations(nodes, mask, 10)
        idx2 = select_maximin_locations(nodes, mask, 10)
        np.testing.assert_array_equal(idx1, idx2)

    def test_output_sorted(self):
        nodes = np.random.default_rng(1).uniform(size=(2, 50))
        mask = np.ones(50, dtype=bool)
        idx = select_maximin_locations(nodes, mask, 8)
        assert np.all(idx[:-1] <= idx[1:])

    def test_output_length(self):
        nodes = np.random.default_rng(2).uniform(size=(2, 80))
        mask = np.ones(80, dtype=bool)
        idx = select_maximin_locations(nodes, mask, 15)
        assert len(idx) == 15

    def test_indices_within_bounds(self):
        nodes = np.random.default_rng(3).uniform(size=(2, 60))
        mask = np.ones(60, dtype=bool)
        idx = select_maximin_locations(nodes, mask, 10)
        assert np.all(idx >= 0)
        assert np.all(idx < 60)

    def test_unique_indices(self):
        nodes = np.random.default_rng(4).uniform(size=(2, 50))
        mask = np.ones(50, dtype=bool)
        idx = select_maximin_locations(nodes, mask, 20)
        assert len(set(idx.tolist())) == 20

    def test_minimum_pairwise_distance(self):
        rng = np.random.default_rng(5)
        nodes = rng.uniform(size=(2, 200))
        mask = np.ones(200, dtype=bool)
        idx = select_maximin_locations(nodes, mask, 10)
        selected = nodes[:, idx]
        # Compute pairwise distances
        dists = np.linalg.norm(
            selected[:, :, None] - selected[:, None, :], axis=0,
        )
        np.fill_diagonal(dists, np.inf)
        min_dist = dists.min()
        # Random selection of 10 from 200 in [0,1]^2 should give
        # min distance > 0.05 with maximin
        assert min_dist > 0.05

    def test_respects_feasibility_mask(self):
        nodes = np.random.default_rng(6).uniform(size=(2, 100))
        mask = np.zeros(100, dtype=bool)
        mask[50:] = True  # only second half feasible
        idx = select_maximin_locations(nodes, mask, 10)
        assert np.all(idx >= 50)

    def test_raises_insufficient_feasible(self):
        nodes = np.random.default_rng(7).uniform(size=(2, 20))
        mask = np.ones(20, dtype=bool)
        with pytest.raises(ValueError, match="Requested 30"):
            select_maximin_locations(nodes, mask, 30)

    def test_works_1d(self):
        nodes = np.linspace(0, 1, 50).reshape(1, -1)
        mask = np.ones(50, dtype=bool)
        idx = select_maximin_locations(nodes, mask, 5)
        assert len(idx) == 5
        assert np.all(idx < 50)

    def test_all_nodes_selected(self):
        nodes = np.random.default_rng(8).uniform(size=(2, 10))
        mask = np.ones(10, dtype=bool)
        idx = select_maximin_locations(nodes, mask, 10)
        np.testing.assert_array_equal(np.sort(idx), np.arange(10))


class TestSelectQoiLocations:
    def test_disjoint_from_candidates(self):
        rng = np.random.default_rng(10)
        nodes = rng.uniform(size=(2, 200))
        mask = np.ones(200, dtype=bool)
        candidates = select_maximin_locations(nodes, mask, 40)
        qoi = select_qoi_locations(
            nodes, mask, 6, candidates, x_threshold=0.3,
        )
        assert len(set(candidates.tolist()) & set(qoi.tolist())) == 0

    def test_downstream_constraint(self):
        rng = np.random.default_rng(11)
        nodes = rng.uniform(size=(2, 200))
        mask = np.ones(200, dtype=bool)
        threshold = 0.7
        candidates = np.array([], dtype=np.intp)
        qoi = select_qoi_locations(
            nodes, mask, 6, candidates, x_threshold=threshold,
        )
        assert np.all(nodes[0, qoi] >= threshold)

    def test_output_length(self):
        rng = np.random.default_rng(12)
        nodes = rng.uniform(size=(2, 200))
        mask = np.ones(200, dtype=bool)
        candidates = np.array([], dtype=np.intp)
        qoi = select_qoi_locations(
            nodes, mask, 6, candidates, x_threshold=0.3,
        )
        assert len(qoi) == 6

    def test_raises_insufficient_downstream(self):
        rng = np.random.default_rng(13)
        nodes = rng.uniform(size=(2, 20))
        mask = np.ones(20, dtype=bool)
        candidates = np.array([], dtype=np.intp)
        with pytest.raises(ValueError, match="Requested"):
            select_qoi_locations(
                nodes, mask, 100, candidates, x_threshold=0.9,
            )


class TestGetFeasibleMask:
    def test_all_feasible(self):
        nodes = np.random.default_rng(20).uniform(size=(2, 50))
        mask = get_feasible_mask(nodes)
        assert mask.shape == (50,)
        assert mask.all()
