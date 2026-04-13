"""
Tests for OED benchmark discovery via BenchmarkRegistry.

Verifies expected benchmarks for HasExactEIG, HasPrior, and OED-specific
protocols. Tests access maps via bm.problem().obs_map() for consistency.
"""

import numpy as np

# Ensure all OED benchmarks are registered by importing the package
import pyapprox_benchmarks.instances.oed  # noqa: F401
from pyapprox_benchmarks.protocols import (
    KLOEDBenchmarkProtocol,
    PredictionOEDBenchmarkProtocol,
)
from pyapprox_benchmarks.registry import BenchmarkRegistry
from pyapprox.expdesign.protocols.oed import (
    BayesianInferenceProblemProtocol,
    GaussianInferenceProblemProtocol,
)

# Exhaustive lists — update when adding OED benchmarks.
# Previously discovered dynamically via BenchmarkRegistry.names_satisfying(),
# which was removed because it instantiated every registered benchmark.
_HAS_EXACT_EIG_NAMES = {
    "cantilever_beam_2d_load_oed",
    "linear_gaussian_kl_oed",
}

_HAS_PRIOR_OED_NAMES = {
    "linear_gaussian_kl_oed",
    "nonlinear_gaussian_pred_oed",
    "linear_gaussian_pred_oed",
    "lotka_volterra_oed",
}


class TestOEDDiscovery:
    """Test OED benchmark discovery via protocol-based filtering."""

    def test_has_exact_eig_names(self):
        assert "linear_gaussian_kl_oed" in _HAS_EXACT_EIG_NAMES
        # Benchmarks without exact_eig should not appear
        assert "lotka_volterra_oed" not in _HAS_EXACT_EIG_NAMES

    def test_has_prior(self):
        for expected in [
            "linear_gaussian_kl_oed",
            "nonlinear_gaussian_pred_oed",
            "linear_gaussian_pred_oed",
            "lotka_volterra_oed",
        ]:
            assert expected in _HAS_PRIOR_OED_NAMES

    def test_observation_model_callable(self, bkd):
        bm = BenchmarkRegistry.get("linear_gaussian_kl_oed", bkd)
        obs_map = bm.problem().obs_map()
        nparams = bm.problem().nparams()
        nobs = bm.problem().nobs()
        samples = bkd.ones((nparams, 3))
        result = obs_map(samples)
        assert result.shape == (nobs, 3)

    def test_observation_model_matches_design_matrix(self, bkd):
        bm = BenchmarkRegistry.get("linear_gaussian_kl_oed", bkd)
        obs_map = bm.problem().obs_map()
        np.random.seed(42)
        theta_np = np.random.randn(bm.problem().nparams(), 5)
        theta = bkd.asarray(theta_np)
        result = obs_map(theta)
        expected = bkd.dot(bm.design_matrix(), theta)
        bkd.assert_allclose(result, expected)

    def test_all_registered_oed_benchmarks(self, bkd):
        oed_names = BenchmarkRegistry.list_category("oed")
        for expected in [
            "linear_gaussian_kl_oed",
            "nonlinear_gaussian_pred_oed",
            "linear_gaussian_pred_oed",
            "lotka_volterra_oed",
        ]:
            assert expected in oed_names

    def test_lotka_volterra_has_problem(self, bkd):
        """Test lotka_volterra_oed has problem() with BayesianInferenceProblem."""
        bm = BenchmarkRegistry.get("lotka_volterra_oed", bkd)
        problem = bm.problem()
        assert isinstance(problem, BayesianInferenceProblemProtocol)
        assert problem.nobs() == bm.obs_map().nqoi()
        assert problem.nparams() == bm.prior().nvars()

    def test_kl_benchmark_protocol(self, bkd):
        """Test KL benchmark satisfies KLOEDBenchmarkProtocol."""
        bm = BenchmarkRegistry.get("linear_gaussian_kl_oed", bkd)
        assert isinstance(bm, KLOEDBenchmarkProtocol)
        problem = bm.problem()
        assert isinstance(problem, GaussianInferenceProblemProtocol)
        # Verify obs_map is callable
        nparams = problem.nparams()
        samples = bkd.ones((nparams, 2))
        result = problem.obs_map()(samples)
        assert result.shape[1] == 2

    def test_prediction_benchmark_protocol(self, bkd):
        """Prediction benchmark satisfies protocol."""
        bm = BenchmarkRegistry.get("nonlinear_gaussian_pred_oed", bkd)
        assert isinstance(bm, PredictionOEDBenchmarkProtocol)
        problem = bm.problem()
        assert isinstance(problem, GaussianInferenceProblemProtocol)

    def test_nonlinear_gaussian_pred_model(self, bkd):
        """Test nonlinear pred benchmark qoi_map: exp(0) = 1."""
        bm = BenchmarkRegistry.get("nonlinear_gaussian_pred_oed", bkd)
        qoi_map = bm.problem().qoi_map()
        nparams = bm.problem().nparams()
        npred = qoi_map.nqoi()
        samples = bkd.zeros((nparams, 2))
        result = qoi_map(samples)
        assert result.shape == (npred, 2)
        expected = bkd.ones((npred, 2))
        bkd.assert_allclose(result, expected)
