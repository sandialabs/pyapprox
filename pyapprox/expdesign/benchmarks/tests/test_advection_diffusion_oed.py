"""Tests for ObstructedAdvectionDiffusionOEDBenchmark.

NumPy only (Galerkin uses skfem which is NumPy-based).
Forward evaluations are slow, so model evaluation tests use @slow_test.
"""

import unittest

import numpy as np

from pyapprox.expdesign.benchmarks.advection_diffusion import (
    ObstructedAdvectionDiffusionOEDBenchmark,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.test_utils import load_tests, slow_test  # noqa: F401


class TestAdvectionDiffusionOEDBenchmark(unittest.TestCase):
    """Tests for ObstructedAdvectionDiffusionOEDBenchmark (NumPy only)."""

    def setUp(self):
        self._bkd = NumpyBkd()
        # Use minimal refinement and few KLE terms for speed
        self._nstokes_refine = 1
        self._nadvec_diff_refine = 1
        self._nkle_terms = 3
        self._nsensors = 5

    def _create_benchmark(self):
        return ObstructedAdvectionDiffusionOEDBenchmark(
            self._bkd,
            nstokes_refine=self._nstokes_refine,
            nadvec_diff_refine=self._nadvec_diff_refine,
            nkle_terms=self._nkle_terms,
            nsensors=self._nsensors,
        )

    def test_construction(self):
        """Test benchmark can be constructed."""
        bench = self._create_benchmark()
        self.assertIsNotNone(bench)

    def test_nparams(self):
        """Test correct number of parameters: nkle + 3."""
        bench = self._create_benchmark()
        self.assertEqual(bench.nparams(), self._nkle_terms + 3)

    def test_prior_nvars(self):
        """Test prior has correct number of variables."""
        bench = self._create_benchmark()
        self.assertEqual(bench.prior().nvars(), self._nkle_terms + 3)

    def test_prior_rvs_shape(self):
        """Test prior sampling returns correct shape."""
        bench = self._create_benchmark()
        np.random.seed(42)
        samples = bench.prior().rvs(3)
        self.assertEqual(samples.shape, (self._nkle_terms + 3, 3))

    def test_observation_model_nqoi(self):
        """Test observation model has nsensors QoI."""
        bench = self._create_benchmark()
        self.assertEqual(bench.observation_model().nqoi(), self._nsensors)

    def test_prediction_model_nqoi(self):
        """Test prediction model has 1 QoI."""
        bench = self._create_benchmark()
        self.assertEqual(bench.prediction_model().nqoi(), 1)

    def test_observation_locations_shape(self):
        """Test observation locations shape is (2, nsensors)."""
        bench = self._create_benchmark()
        locs = bench.observation_locations()
        self.assertEqual(locs.shape, (2, self._nsensors))

    def test_nobservations(self):
        """Test nobservations accessor."""
        bench = self._create_benchmark()
        self.assertEqual(bench.nobservations(), self._nsensors)

    def test_observation_locations_in_domain(self):
        """Test sensor locations are within [0,1]^2."""
        bench = self._create_benchmark()
        locs_np = self._bkd.to_numpy(bench.observation_locations())
        self.assertTrue(np.all(locs_np >= -1e-12))
        self.assertTrue(np.all(locs_np <= 1.0 + 1e-12))

    @slow_test
    def test_observation_model_evaluation(self):
        """Test observation model returns correct shape."""
        bench = self._create_benchmark()
        np.random.seed(42)
        sample = bench.prior().rvs(1)
        obs = bench.observation_model()(sample)
        self.assertEqual(obs.shape, (self._nsensors, 1))
        obs_np = self._bkd.to_numpy(obs)
        self.assertTrue(np.all(np.isfinite(obs_np)))

    @slow_test
    def test_prediction_model_evaluation(self):
        """Test prediction model returns correct shape."""
        bench = self._create_benchmark()
        np.random.seed(42)
        sample = bench.prior().rvs(1)
        pred = bench.prediction_model()(sample)
        self.assertEqual(pred.shape, (1, 1))
        pred_np = self._bkd.to_numpy(pred)
        self.assertTrue(np.all(np.isfinite(pred_np)))


if __name__ == "__main__":
    unittest.main()
