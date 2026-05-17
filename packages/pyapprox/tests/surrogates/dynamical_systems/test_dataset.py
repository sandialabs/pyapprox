"""Tests for SnapshotDataset."""

import numpy as np

from pyapprox.surrogates.dynamical_systems.dataset import SnapshotDataset
from pyapprox.surrogates.dynamical_systems.encoders import (
    IdentityEncoder,
    LinearEncoder,
)


class TestSnapshotDataset:
    def test_basic_construction(self, bkd):
        states = bkd.array(np.random.RandomState(0).randn(3, 20))
        derivs = bkd.array(np.random.RandomState(1).randn(3, 20))
        ds = SnapshotDataset(states, derivs, bkd)
        assert ds.nstates() == 3
        assert ds.nsamples() == 20
        assert ds.times() is None
        bkd.assert_allclose(ds.states(), states)
        bkd.assert_allclose(ds.derivatives(), derivs)

    def test_construction_with_times(self, bkd):
        states = bkd.array(np.random.RandomState(0).randn(2, 10))
        derivs = bkd.array(np.random.RandomState(1).randn(2, 10))
        times = bkd.array(np.linspace(0, 1, 10))
        ds = SnapshotDataset(states, derivs, bkd, times=times)
        bkd.assert_allclose(ds.times(), times)

    def test_shape_mismatch_raises(self, bkd):
        import pytest

        states = bkd.array(np.zeros((3, 10)))
        derivs = bkd.array(np.zeros((2, 8)))
        with pytest.raises(ValueError, match="nsamples"):
            SnapshotDataset(states, derivs, bkd)

    def test_rectangular_states_derivatives_allowed(self, bkd):
        states = bkd.array(np.zeros((3, 10)))
        derivs = bkd.array(np.zeros((2, 10)))
        ds = SnapshotDataset(states, derivs, bkd)
        assert ds.nstates_input() == 3
        assert ds.nstates_output() == 2
        assert ds.nsamples() == 10

    def test_times_length_mismatch_raises(self, bkd):
        import pytest

        states = bkd.array(np.zeros((2, 10)))
        derivs = bkd.array(np.zeros((2, 10)))
        times = bkd.array(np.zeros(5))
        with pytest.raises(ValueError, match="times length"):
            SnapshotDataset(states, derivs, bkd, times=times)


class TestSnapshotDatasetFromTrajectory:
    def _make_linear_trajectory(self, bkd):
        """x(t) = [t, 2*t], dx/dt = [1, 2] (constant)."""
        times = bkd.array(np.linspace(0, 1, 21))
        t_np = np.linspace(0, 1, 21)
        traj = bkd.array(np.stack([t_np, 2 * t_np], axis=0))
        return traj, times

    def test_central_differences(self, bkd):
        traj, times = self._make_linear_trajectory(bkd)
        ds = SnapshotDataset.from_trajectory(traj, times, bkd, "central")
        assert ds.nsamples() == 19
        assert ds.nstates() == 2
        bkd.assert_allclose(
            ds.derivatives(), bkd.array(np.ones((2, 19)) * [[1], [2]]),
            rtol=1e-12,
        )

    def test_forward_differences(self, bkd):
        traj, times = self._make_linear_trajectory(bkd)
        ds = SnapshotDataset.from_trajectory(traj, times, bkd, "forward")
        assert ds.nsamples() == 20
        bkd.assert_allclose(
            ds.derivatives(), bkd.array(np.ones((2, 20)) * [[1], [2]]),
            rtol=1e-12,
        )

    def test_backward_differences(self, bkd):
        traj, times = self._make_linear_trajectory(bkd)
        ds = SnapshotDataset.from_trajectory(traj, times, bkd, "backward")
        assert ds.nsamples() == 20
        bkd.assert_allclose(
            ds.derivatives(), bkd.array(np.ones((2, 20)) * [[1], [2]]),
            rtol=1e-12,
        )

    def test_quadratic_central_accuracy(self, bkd):
        """Central FD on x(t)=t^2 should be exact (2nd order method)."""
        t_np = np.linspace(0, 2, 101)
        traj = bkd.array(np.stack([t_np**2, 3 * t_np**2], axis=0))
        times = bkd.array(t_np)
        ds = SnapshotDataset.from_trajectory(traj, times, bkd, "central")
        t_interior = t_np[1:-1]
        expected = bkd.array(
            np.stack([2 * t_interior, 6 * t_interior], axis=0)
        )
        bkd.assert_allclose(ds.derivatives(), expected, rtol=1e-10)


class TestSnapshotDatasetFromTrajectories:
    def test_concatenation(self, bkd):
        t1 = bkd.array(np.linspace(0, 1, 11))
        t2 = bkd.array(np.linspace(0, 2, 21))
        t1_np = np.linspace(0, 1, 11)
        t2_np = np.linspace(0, 2, 21)
        traj1 = bkd.array(np.stack([t1_np, t1_np], axis=0))
        traj2 = bkd.array(np.stack([t2_np, t2_np], axis=0))
        ds = SnapshotDataset.from_trajectories(
            [traj1, traj2], [t1, t2], bkd, "central"
        )
        assert ds.nsamples() == 9 + 19
        assert ds.nstates() == 2


class TestSnapshotDatasetProject:
    def test_identity_projection(self, bkd):
        states = bkd.array(np.random.RandomState(0).randn(3, 15))
        derivs = bkd.array(np.random.RandomState(1).randn(3, 15))
        ds = SnapshotDataset(states, derivs, bkd)
        enc = IdentityEncoder(3, bkd)
        projected = ds.project(enc)
        bkd.assert_allclose(projected.states(), states)
        bkd.assert_allclose(projected.derivatives(), derivs)

    def test_linear_projection(self, bkd):
        rng = np.random.RandomState(0)
        states_np = rng.randn(4, 20)
        derivs_np = rng.randn(4, 20)
        Q, _ = np.linalg.qr(rng.randn(4, 4))
        P_np = Q[:2, :]

        states = bkd.array(states_np)
        derivs = bkd.array(derivs_np)
        P = bkd.array(P_np)
        enc = LinearEncoder(P, bkd)
        projected = SnapshotDataset(states, derivs, bkd).project(enc)

        expected_states = bkd.array(P_np @ states_np)
        expected_derivs = bkd.array(P_np @ derivs_np)
        bkd.assert_allclose(projected.states(), expected_states, rtol=1e-10)
        bkd.assert_allclose(projected.derivatives(), expected_derivs, rtol=1e-10)
        assert projected.nstates() == 2
