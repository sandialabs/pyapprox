"""Tests for ObstructedAdvectionDiffusionOEDBenchmark.

NumPy only (Galerkin uses skfem which is NumPy-based).
Forward evaluations are slow, so model evaluation tests use @slow_test.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.expdesign.benchmarks.advection_diffusion import (
    ObstructedAdvectionDiffusionOEDBenchmark,
)
from pyapprox.util.test_utils import slow_test


class TestAdvectionDiffusionOEDBenchmark:
    """Tests for ObstructedAdvectionDiffusionOEDBenchmark (NumPy only)."""

    def _setup_data(self, bkd):
        # Use minimal refinement and few KLE terms for speed
        self._nstokes_refine = 1
        self._nadvec_diff_refine = 1
        self._nkle_terms = 3
        self._nsensors = 5

    def _create_benchmark(self, bkd):
        return ObstructedAdvectionDiffusionOEDBenchmark(
            bkd,
            nstokes_refine=self._nstokes_refine,
            nadvec_diff_refine=self._nadvec_diff_refine,
            nkle_terms=self._nkle_terms,
            nsensors=self._nsensors,
        )

    def test_construction(self, bkd):
        """Test benchmark can be constructed."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        assert bench is not None

    def test_nparams(self, bkd):
        """Test correct number of parameters: nkle + 3."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        assert bench.nparams() == self._nkle_terms + 3

    def test_prior_nvars(self, bkd):
        """Test prior has correct number of variables."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        assert bench.prior().nvars() == self._nkle_terms + 3

    def test_prior_rvs_shape(self, bkd):
        """Test prior sampling returns correct shape."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        np.random.seed(42)
        samples = bench.prior().rvs(3)
        assert samples.shape == (self._nkle_terms + 3, 3)

    def test_observation_model_nqoi(self, bkd):
        """Test observation model has nsensors QoI."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        assert bench.observation_model().nqoi() == self._nsensors

    def test_prediction_model_nqoi(self, bkd):
        """Test prediction model has 1 QoI."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        assert bench.prediction_model().nqoi() == 1

    def test_observation_locations_shape(self, bkd):
        """Test observation locations shape is (2, nsensors)."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        locs = bench.observation_locations()
        assert locs.shape == (2, self._nsensors)

    def test_nobservations(self, bkd):
        """Test nobservations accessor."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        assert bench.nobservations() == self._nsensors

    def test_observation_locations_in_domain(self, bkd):
        """Test sensor locations are within [0,1]^2."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        locs_np = bkd.to_numpy(bench.observation_locations())
        assert np.all(locs_np >= -1e-12)
        assert np.all(locs_np <= 1.0 + 1e-12)

    @slow_test
    def test_observation_model_evaluation(self, bkd):
        """Test observation model returns correct shape."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        np.random.seed(42)
        sample = bench.prior().rvs(1)
        obs = bench.observation_model()(sample)
        assert obs.shape == (self._nsensors, 1)
        obs_np = bkd.to_numpy(obs)
        assert np.all(np.isfinite(obs_np))

    @slow_test
    def test_prediction_model_evaluation(self, bkd):
        """Test prediction model returns correct shape."""
        self._setup_data(bkd)
        bench = self._create_benchmark(bkd)
        np.random.seed(42)
        sample = bench.prior().rvs(1)
        pred = bench.prediction_model()(sample)
        assert pred.shape == (1, 1)
        pred_np = bkd.to_numpy(pred)
        assert np.all(np.isfinite(pred_np))
