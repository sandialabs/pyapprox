"""Tests for LotkaVolterraPredictionOEDProblem."""

import numpy as np

from pyapprox_benchmarks.expdesign.lotka_volterra import (
    LotkaVolterraPredictionOEDProblem,
)
from tests._helpers.markers import slow_test


class TestLotkaVolterraPredictionOEDProblem:

    def _create_oed_problem(self, bkd):
        return LotkaVolterraPredictionOEDProblem(bkd)

    def test_construction(self, bkd):
        oed_problem = self._create_oed_problem(bkd)
        assert oed_problem is not None

    def test_prior_nvars(self, bkd):
        oed_problem = self._create_oed_problem(bkd)
        assert oed_problem.inference_problem().prior().nvars() == 12

    def test_prior_rvs_shape(self, bkd):
        oed_problem = self._create_oed_problem(bkd)
        np.random.seed(42)
        samples = oed_problem.inference_problem().prior().rvs(5)
        assert samples.shape == (12, 5)

    def test_observation_model_nqoi(self, bkd):
        oed_problem = self._create_oed_problem(bkd)
        assert oed_problem.obs_map().nqoi() == 22

    def test_prediction_model_nqoi(self, bkd):
        oed_problem = self._create_oed_problem(bkd)
        assert oed_problem.qoi_map().nqoi() == 5

    def test_solution_times_shape(self, bkd):
        oed_problem = self._create_oed_problem(bkd)
        times = oed_problem.solution_times()
        assert times.shape[0] == 11

    def test_observation_times_shape(self, bkd):
        oed_problem = self._create_oed_problem(bkd)
        obs_times = oed_problem.observation_times()
        sol_times = oed_problem.solution_times()
        bkd.assert_allclose(obs_times, sol_times)

    def test_prediction_times_shape(self, bkd):
        oed_problem = self._create_oed_problem(bkd)
        pred_times = oed_problem.prediction_times()
        assert pred_times.shape[0] == 5

    @slow_test
    def test_observation_model_evaluation(self, bkd):
        oed_problem = self._create_oed_problem(bkd)
        np.random.seed(42)
        sample = oed_problem.inference_problem().prior().rvs(1)
        obs = oed_problem.obs_map()(sample)
        assert obs.shape == (22, 1)
        obs_np = bkd.to_numpy(obs)
        assert np.all(np.isfinite(obs_np))

    @slow_test
    def test_prediction_model_evaluation(self, bkd):
        oed_problem = self._create_oed_problem(bkd)
        np.random.seed(42)
        sample = oed_problem.inference_problem().prior().rvs(1)
        pred = oed_problem.qoi_map()(sample)
        assert pred.shape == (5, 1)
        pred_np = bkd.to_numpy(pred)
        assert np.all(np.isfinite(pred_np))

    @slow_test
    def test_batch_evaluation(self, bkd):
        oed_problem = self._create_oed_problem(bkd)
        np.random.seed(42)
        samples = oed_problem.inference_problem().prior().rvs(3)
        obs = oed_problem.obs_map()(samples)
        pred = oed_problem.qoi_map()(samples)
        assert obs.shape == (22, 3)
        assert pred.shape == (5, 3)
