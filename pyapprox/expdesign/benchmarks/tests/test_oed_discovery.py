"""
Tests for OED benchmark discovery via BenchmarkRegistry.

Verifies that names_satisfying() returns the expected benchmarks for
HasObservationModel, HasPredictionModel, HasExactEIG, and HasPrior
protocol combinations.
"""

import numpy as np
import pytest

# Ensure all OED benchmarks are registered by importing the package
import pyapprox.expdesign.benchmarks  # noqa: F401
from pyapprox.benchmarks.protocols import (
    HasExactEIG,
    HasObservationModel,
    HasPredictionModel,
    HasPrior,
)
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.util.test_utils import (
    slowest_test,
)


class TestOEDDiscovery:
    """Test OED benchmark discovery via protocol-based filtering."""

    @slowest_test
    def test_has_observation_model_names(self, bkd):
        names = BenchmarkRegistry.names_satisfying(
            HasObservationModel,
            bkd=bkd,
        )
        for expected in [
            "linear_gaussian_oed",
            "nonlinear_gaussian_oed",
            "linear_gaussian_pred_oed",
            "lotka_volterra_oed",
        ]:
            assert expected in names

    @slowest_test
    def test_has_observation_model_and_prior(self, bkd):
        names = BenchmarkRegistry.names_satisfying(
            HasObservationModel,
            HasPrior,
            bkd=bkd,
        )
        for expected in [
            "linear_gaussian_oed",
            "nonlinear_gaussian_oed",
            "linear_gaussian_pred_oed",
            "lotka_volterra_oed",
        ]:
            assert expected in names

    @slowest_test
    def test_has_prediction_model_names(self, bkd):
        names = BenchmarkRegistry.names_satisfying(
            HasObservationModel,
            HasPredictionModel,
            HasPrior,
            bkd=bkd,
        )
        for expected in [
            "nonlinear_gaussian_oed",
            "linear_gaussian_pred_oed",
            "lotka_volterra_oed",
        ]:
            assert expected in names
        assert "linear_gaussian_oed" not in names

    @slowest_test
    def test_has_exact_eig_names(self, bkd):
        names = BenchmarkRegistry.names_satisfying(
            HasExactEIG,
            bkd=bkd,
        )
        assert "linear_gaussian_oed" in names
        # Benchmarks without exact_eig should not appear
        assert "nonlinear_gaussian_oed" not in names
        assert "linear_gaussian_pred_oed" not in names
        assert "lotka_volterra_oed" not in names

    def test_observation_model_callable(self, bkd):
        bm = BenchmarkRegistry.get("linear_gaussian_oed", bkd)
        obs_model = bm.observation_model()
        nparams = bm.nparams()
        nobs = bm.nobs()
        samples = bkd.ones((nparams, 3))
        result = obs_model(samples)
        assert result.shape == (nobs, 3)

    def test_prediction_model_callable_nonlinear(self, bkd):
        bm = BenchmarkRegistry.get("nonlinear_gaussian_oed", bkd)
        pred_model = bm.prediction_model()
        nparams = bm.nparams()
        npred = bm.npred()
        samples = bkd.zeros((nparams, 2))
        result = pred_model(samples)
        assert result.shape == (npred, 2)
        # exp(0) = 1.0 for all entries
        expected = bkd.ones((npred, 2))
        bkd.assert_allclose(result, expected)

    def test_prediction_model_callable_linear(self, bkd):
        bm = BenchmarkRegistry.get("linear_gaussian_pred_oed", bkd)
        pred_model = bm.prediction_model()
        nparams = bm.nparams()
        npred = bm.npred()
        samples = bkd.zeros((nparams, 2))
        result = pred_model(samples)
        assert result.shape == (npred, 2)
        expected = bkd.zeros((npred, 2))
        bkd.assert_allclose(result, expected)

    def test_observation_model_matches_design_matrix(self, bkd):
        bm = BenchmarkRegistry.get("linear_gaussian_oed", bkd)
        obs_model = bm.observation_model()
        np.random.seed(42)
        theta_np = np.random.randn(bm.nparams(), 5)
        theta = bkd.asarray(theta_np)
        result = obs_model(theta)
        expected = bkd.dot(bm.design_matrix(), theta)
        bkd.assert_allclose(result, expected)

    def test_all_registered_oed_benchmarks(self, bkd):
        oed_names = BenchmarkRegistry.list_category("oed")
        for expected in [
            "linear_gaussian_oed",
            "nonlinear_gaussian_oed",
            "linear_gaussian_pred_oed",
            "lotka_volterra_oed",
        ]:
            assert expected in oed_names
