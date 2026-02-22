"""Tests for LotkaVolterraOEDBenchmark."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests, slow_test  # noqa: F401

from pyapprox.typing.expdesign.benchmarks.lotka_volterra import (
    LotkaVolterraOEDBenchmark,
)


class TestLotkaVolterraOEDBenchmark(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _create_benchmark(self) -> LotkaVolterraOEDBenchmark[Array]:
        return LotkaVolterraOEDBenchmark(self._bkd)

    def test_construction(self):
        """Test benchmark can be constructed."""
        bench = self._create_benchmark()
        self.assertIsNotNone(bench)

    def test_prior_nvars(self):
        """Test prior has 12 variables (3 growth + 9 competition)."""
        bench = self._create_benchmark()
        self.assertEqual(bench.prior().nvars(), 12)

    def test_prior_rvs_shape(self):
        """Test prior sampling returns correct shape."""
        bench = self._create_benchmark()
        np.random.seed(42)
        samples = bench.prior().rvs(5)
        self.assertEqual(samples.shape, (12, 5))

    def test_observation_model_nqoi(self):
        """Test observation model: species 0 and 2 at 11 times = 22 QoI."""
        bench = self._create_benchmark()
        self.assertEqual(bench.observation_model().nqoi(), 22)

    def test_prediction_model_nqoi(self):
        """Test prediction model: species 1 at 5 odd indices = 5 QoI."""
        bench = self._create_benchmark()
        self.assertEqual(bench.prediction_model().nqoi(), 5)

    def test_solution_times_shape(self):
        """Test solution times: 11 points from 0 to 10."""
        bench = self._create_benchmark()
        times = bench.solution_times()
        self.assertEqual(times.shape[0], 11)

    def test_observation_times_shape(self):
        """Test observation times matches solution times."""
        bench = self._create_benchmark()
        obs_times = bench.observation_times()
        sol_times = bench.solution_times()
        self._bkd.assert_allclose(obs_times, sol_times)

    def test_prediction_times_shape(self):
        """Test prediction times: 5 odd-indexed time points."""
        bench = self._create_benchmark()
        pred_times = bench.prediction_times()
        self.assertEqual(pred_times.shape[0], 5)

    @slow_test
    def test_observation_model_evaluation(self):
        """Test observation model returns correct shape (22, nsamples)."""
        bench = self._create_benchmark()
        np.random.seed(42)
        sample = bench.prior().rvs(1)
        obs = bench.observation_model()(sample)
        self.assertEqual(obs.shape, (22, 1))
        # Values should be finite and positive (populations)
        obs_np = self._bkd.to_numpy(obs)
        self.assertTrue(np.all(np.isfinite(obs_np)))

    @slow_test
    def test_prediction_model_evaluation(self):
        """Test prediction model returns correct shape (5, nsamples)."""
        bench = self._create_benchmark()
        np.random.seed(42)
        sample = bench.prior().rvs(1)
        pred = bench.prediction_model()(sample)
        self.assertEqual(pred.shape, (5, 1))
        pred_np = self._bkd.to_numpy(pred)
        self.assertTrue(np.all(np.isfinite(pred_np)))

    @slow_test
    def test_batch_evaluation(self):
        """Test models handle multiple samples."""
        bench = self._create_benchmark()
        np.random.seed(42)
        samples = bench.prior().rvs(3)
        obs = bench.observation_model()(samples)
        pred = bench.prediction_model()(samples)
        self.assertEqual(obs.shape, (22, 3))
        self.assertEqual(pred.shape, (5, 3))


class TestLotkaVolterraOEDBenchmarkNumpy(
    TestLotkaVolterraOEDBenchmark[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLotkaVolterraOEDBenchmarkTorch(
    TestLotkaVolterraOEDBenchmark[torch.Tensor]
):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
