"""
Tests for OED benchmark discovery via BenchmarkRegistry.

Verifies that names_satisfying() returns the expected benchmarks for
HasObservationModel, HasPredictionModel, HasExactEIG, and HasPrior
protocol combinations.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests  # noqa: F401

from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.benchmarks.protocols import (
    HasObservationModel,
    HasPredictionModel,
    HasExactEIG,
    HasPrior,
)

# Ensure all OED benchmarks are registered by importing the package
import pyapprox.expdesign.benchmarks  # noqa: F401


class TestOEDDiscovery(Generic[Array], unittest.TestCase):
    """Test OED benchmark discovery via protocol-based filtering."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_has_observation_model_names(self):
        names = BenchmarkRegistry.names_satisfying(
            HasObservationModel, bkd=self._bkd,
        )
        for expected in [
            "linear_gaussian_oed",
            "nonlinear_gaussian_oed",
            "linear_gaussian_pred_oed",
            "lotka_volterra_oed",
        ]:
            self.assertIn(expected, names)

    def test_has_observation_model_and_prior(self):
        names = BenchmarkRegistry.names_satisfying(
            HasObservationModel, HasPrior, bkd=self._bkd,
        )
        for expected in [
            "linear_gaussian_oed",
            "nonlinear_gaussian_oed",
            "linear_gaussian_pred_oed",
            "lotka_volterra_oed",
        ]:
            self.assertIn(expected, names)

    def test_has_prediction_model_names(self):
        names = BenchmarkRegistry.names_satisfying(
            HasObservationModel, HasPredictionModel, HasPrior, bkd=self._bkd,
        )
        for expected in [
            "nonlinear_gaussian_oed",
            "linear_gaussian_pred_oed",
            "lotka_volterra_oed",
        ]:
            self.assertIn(expected, names)
        self.assertNotIn("linear_gaussian_oed", names)

    def test_has_exact_eig_names(self):
        names = BenchmarkRegistry.names_satisfying(
            HasExactEIG, bkd=self._bkd,
        )
        self.assertIn("linear_gaussian_oed", names)
        # Benchmarks without exact_eig should not appear
        self.assertNotIn("nonlinear_gaussian_oed", names)
        self.assertNotIn("linear_gaussian_pred_oed", names)
        self.assertNotIn("lotka_volterra_oed", names)

    def test_observation_model_callable(self):
        bm = BenchmarkRegistry.get("linear_gaussian_oed", self._bkd)
        obs_model = bm.observation_model()
        nparams = bm.nparams()
        nobs = bm.nobs()
        samples = self._bkd.ones((nparams, 3))
        result = obs_model(samples)
        self.assertEqual(result.shape, (nobs, 3))

    def test_prediction_model_callable_nonlinear(self):
        bm = BenchmarkRegistry.get("nonlinear_gaussian_oed", self._bkd)
        pred_model = bm.prediction_model()
        nparams = bm.nparams()
        npred = bm.npred()
        samples = self._bkd.zeros((nparams, 2))
        result = pred_model(samples)
        self.assertEqual(result.shape, (npred, 2))
        # exp(0) = 1.0 for all entries
        expected = self._bkd.ones((npred, 2))
        self._bkd.assert_allclose(result, expected)

    def test_prediction_model_callable_linear(self):
        bm = BenchmarkRegistry.get("linear_gaussian_pred_oed", self._bkd)
        pred_model = bm.prediction_model()
        nparams = bm.nparams()
        npred = bm.npred()
        samples = self._bkd.zeros((nparams, 2))
        result = pred_model(samples)
        self.assertEqual(result.shape, (npred, 2))
        expected = self._bkd.zeros((npred, 2))
        self._bkd.assert_allclose(result, expected)

    def test_observation_model_matches_design_matrix(self):
        bm = BenchmarkRegistry.get("linear_gaussian_oed", self._bkd)
        obs_model = bm.observation_model()
        np.random.seed(42)
        theta_np = np.random.randn(bm.nparams(), 5)
        theta = self._bkd.asarray(theta_np)
        result = obs_model(theta)
        expected = self._bkd.dot(bm.design_matrix(), theta)
        self._bkd.assert_allclose(result, expected)

    def test_all_registered_oed_benchmarks(self):
        oed_names = BenchmarkRegistry.list_category("oed")
        for expected in [
            "linear_gaussian_oed",
            "nonlinear_gaussian_oed",
            "linear_gaussian_pred_oed",
            "lotka_volterra_oed",
        ]:
            self.assertIn(expected, oed_names)


class TestOEDDiscoveryNumpy(TestOEDDiscovery[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestOEDDiscoveryTorch(TestOEDDiscovery[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
