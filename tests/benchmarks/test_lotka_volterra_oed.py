"""Tests for LotkaVolterraOEDBenchmark."""

import numpy as np

from pyapprox.benchmarks.instances.oed.lotka_volterra import (
    LotkaVolterraOEDBenchmark,
)
from tests._helpers.markers import slow_test


class TestLotkaVolterraOEDBenchmark:

    def _create_benchmark(self, bkd):
        return LotkaVolterraOEDBenchmark(bkd)

    def test_construction(self, bkd):
        """Test benchmark can be constructed."""
        bench = self._create_benchmark(bkd)
        assert bench is not None

    def test_prior_nvars(self, bkd):
        """Test prior has 12 variables (3 growth + 9 competition)."""
        bench = self._create_benchmark(bkd)
        assert bench.prior().nvars() == 12

    def test_prior_rvs_shape(self, bkd):
        """Test prior sampling returns correct shape."""
        bench = self._create_benchmark(bkd)
        np.random.seed(42)
        samples = bench.prior().rvs(5)
        assert samples.shape == (12, 5)

    def test_observation_model_nqoi(self, bkd):
        """Test observation model: species 0 and 2 at 11 times = 22 QoI."""
        bench = self._create_benchmark(bkd)
        assert bench.obs_map().nqoi() == 22

    def test_prediction_model_nqoi(self, bkd):
        """Test prediction model: species 1 at 5 odd indices = 5 QoI."""
        bench = self._create_benchmark(bkd)
        assert bench.qoi_map().nqoi() == 5

    def test_solution_times_shape(self, bkd):
        """Test solution times: 11 points from 0 to 10."""
        bench = self._create_benchmark(bkd)
        times = bench.solution_times()
        assert times.shape[0] == 11

    def test_observation_times_shape(self, bkd):
        """Test observation times matches solution times."""
        bench = self._create_benchmark(bkd)
        obs_times = bench.observation_times()
        sol_times = bench.solution_times()
        bkd.assert_allclose(obs_times, sol_times)

    def test_prediction_times_shape(self, bkd):
        """Test prediction times: 5 odd-indexed time points."""
        bench = self._create_benchmark(bkd)
        pred_times = bench.prediction_times()
        assert pred_times.shape[0] == 5

    @slow_test
    def test_observation_model_evaluation(self, bkd):
        """Test observation model returns correct shape (22, nsamples)."""
        bench = self._create_benchmark(bkd)
        np.random.seed(42)
        sample = bench.prior().rvs(1)
        obs = bench.obs_map()(sample)
        assert obs.shape == (22, 1)
        # Values should be finite and positive (populations)
        obs_np = bkd.to_numpy(obs)
        assert np.all(np.isfinite(obs_np))

    @slow_test
    def test_prediction_model_evaluation(self, bkd):
        """Test prediction model returns correct shape (5, nsamples)."""
        bench = self._create_benchmark(bkd)
        np.random.seed(42)
        sample = bench.prior().rvs(1)
        pred = bench.qoi_map()(sample)
        assert pred.shape == (5, 1)
        pred_np = bkd.to_numpy(pred)
        assert np.all(np.isfinite(pred_np))

    @slow_test
    def test_batch_evaluation(self, bkd):
        """Test models handle multiple samples."""
        bench = self._create_benchmark(bkd)
        np.random.seed(42)
        samples = bench.prior().rvs(3)
        obs = bench.obs_map()(samples)
        pred = bench.qoi_map()(samples)
        assert obs.shape == (22, 3)
        assert pred.shape == (5, 3)
