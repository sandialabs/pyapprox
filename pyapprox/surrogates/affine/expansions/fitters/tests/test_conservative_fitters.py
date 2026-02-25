"""Tests for Conservative Fitters.

Tests focus on:
- Conservativeness: risk_measure(surrogate) >= risk_measure(data) for 500 trials
- Result types and shapes
- Single QoI validation
- Accessor methods
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

from pyapprox.surrogates.affine.expansions.fitters.conservative import (
    ConservativeLstSqFitter,
    ConservativeQuantileFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    DirectSolverResult,
)
from pyapprox.probability.risk import (
    SafetyMarginRiskMeasure,
    AverageValueAtRisk,
)

from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.probability import UniformMarginal


class TestConservativeFitters(Generic[Array], unittest.TestCase):
    """Base test class - NOT run directly."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_expansion(self, nvars: int, max_level: int, nqoi: int = 1):
        """Create test expansion."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    # --- ConservativeLstSqFitter Tests ---

    def test_lstsq_fit_returns_direct_solver_result(self) -> None:
        """ConservativeLstSqFitter returns DirectSolverResult."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(1, 30))

        fitter = ConservativeLstSqFitter(self._bkd, strength=1.0)
        result = fitter.fit(expansion, samples, values)

        self.assertIsInstance(result, DirectSolverResult)

    def test_lstsq_result_params_shape(self) -> None:
        """ConservativeLstSqFitter result params have correct shape."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(1, 30))

        fitter = ConservativeLstSqFitter(self._bkd, strength=1.0)
        result = fitter.fit(expansion, samples, values)

        self.assertEqual(result.params().shape, (expansion.nterms(), 1))

    def test_lstsq_handles_1d_values(self) -> None:
        """ConservativeLstSqFitter handles 1D values array."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values_1d = self._bkd.asarray(np.random.randn(50))

        fitter = ConservativeLstSqFitter(self._bkd, strength=1.0)
        result = fitter.fit(expansion, samples, values_1d)

        self.assertEqual(result.params().shape[1], 1)

    def test_lstsq_fitted_surrogate_evaluates(self) -> None:
        """ConservativeLstSqFitter fitted surrogate can evaluate."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(1, 30))

        fitter = ConservativeLstSqFitter(self._bkd, strength=1.0)
        result = fitter.fit(expansion, samples, values)

        # Evaluate at new points
        test_samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        predictions = result(test_samples)

        self.assertEqual(predictions.shape, (1, 10))

    def test_lstsq_multi_qoi_raises(self) -> None:
        """ConservativeLstSqFitter: nqoi > 1 raises ValueError."""
        expansion = self._create_expansion(nvars=2, max_level=3, nqoi=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(2, 30))

        fitter = ConservativeLstSqFitter(self._bkd, strength=1.0)
        with self.assertRaises(ValueError) as ctx:
            fitter.fit(expansion, samples, values)
        self.assertIn("nqoi=1", str(ctx.exception))

    def test_lstsq_accessors(self) -> None:
        """ConservativeLstSqFitter accessors return correct values."""
        fitter = ConservativeLstSqFitter(self._bkd, strength=2.5)
        self._bkd.assert_allclose(
            self._bkd.asarray([fitter.strength()]),
            self._bkd.asarray([2.5]),
        )

    def test_lstsq_conservative_500_trials(self) -> None:
        """ConservativeLstSqFitter: risk(surrogate) >= risk(data) for 500 trials.

        This is the main conservativeness test, replicating the legacy test.
        """
        bkd = self._bkd
        nsamples, nvars = 100, 2
        max_level = 2

        # Create expansion
        expansion = self._create_expansion(nvars=nvars, max_level=max_level)

        # Generate true coefficients for underlying function
        nterms = expansion.nterms()
        true_coefs = bkd.ones((nterms, 1))
        true_expansion = expansion.with_params(true_coefs)

        # Test parameters
        noise_std = 0.1
        ntrials = 500
        strength = 1.0

        # Create risk measure for comparison
        risk_measure = SafetyMarginRiskMeasure(bkd, strength)

        fitter = ConservativeLstSqFitter(bkd, strength=strength)

        for _ in range(ntrials):
            # Generate samples and noisy values
            samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
            noiseless_values = true_expansion(samples)  # (1, nsamples)
            noise = bkd.asarray(np.random.normal(0, noise_std, (1, nsamples)))
            train_values = noiseless_values + noise

            # Fit conservative surrogate
            result = fitter.fit(expansion, samples, train_values)

            # Compute risk of surrogate predictions
            predictions = result(samples)  # (1, nsamples)
            risk_measure.set_samples(predictions)
            surrogate_risk = float(risk_measure())

            # Compute risk of training data
            risk_measure.set_samples(train_values)
            data_risk = float(risk_measure())

            # Conservative condition must hold
            self.assertGreaterEqual(
                surrogate_risk,
                data_risk - 1e-10,  # Small tolerance for numerical precision
                f"Conservative condition violated: "
                f"surrogate_risk={surrogate_risk} < data_risk={data_risk}",
            )

    # --- ConservativeQuantileFitter Tests ---

    def test_quantile_fit_returns_direct_solver_result(self) -> None:
        """ConservativeQuantileFitter returns DirectSolverResult."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(1, 30))

        fitter = ConservativeQuantileFitter(self._bkd, quantile=0.5)
        result = fitter.fit(expansion, samples, values)

        self.assertIsInstance(result, DirectSolverResult)

    def test_quantile_result_params_shape(self) -> None:
        """ConservativeQuantileFitter result params have correct shape."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(1, 30))

        fitter = ConservativeQuantileFitter(self._bkd, quantile=0.5)
        result = fitter.fit(expansion, samples, values)

        self.assertEqual(result.params().shape, (expansion.nterms(), 1))

    def test_quantile_handles_1d_values(self) -> None:
        """ConservativeQuantileFitter handles 1D values array."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values_1d = self._bkd.asarray(np.random.randn(50))

        fitter = ConservativeQuantileFitter(self._bkd, quantile=0.5)
        result = fitter.fit(expansion, samples, values_1d)

        self.assertEqual(result.params().shape[1], 1)

    def test_quantile_fitted_surrogate_evaluates(self) -> None:
        """ConservativeQuantileFitter fitted surrogate can evaluate."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(1, 30))

        fitter = ConservativeQuantileFitter(self._bkd, quantile=0.5)
        result = fitter.fit(expansion, samples, values)

        # Evaluate at new points
        test_samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        predictions = result(test_samples)

        self.assertEqual(predictions.shape, (1, 10))

    def test_quantile_multi_qoi_raises(self) -> None:
        """ConservativeQuantileFitter: nqoi > 1 raises ValueError."""
        expansion = self._create_expansion(nvars=2, max_level=3, nqoi=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(2, 30))

        fitter = ConservativeQuantileFitter(self._bkd, quantile=0.5)
        with self.assertRaises(ValueError) as ctx:
            fitter.fit(expansion, samples, values)
        self.assertIn("nqoi=1", str(ctx.exception))

    def test_quantile_invalid_quantile_raises(self) -> None:
        """ConservativeQuantileFitter: quantile outside [0, 1) raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ConservativeQuantileFitter(self._bkd, quantile=1.0)
        self.assertIn("quantile", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            ConservativeQuantileFitter(self._bkd, quantile=-0.1)
        self.assertIn("quantile", str(ctx.exception).lower())

    def test_quantile_accessors(self) -> None:
        """ConservativeQuantileFitter accessors return correct values."""
        fitter = ConservativeQuantileFitter(self._bkd, quantile=0.75)
        self._bkd.assert_allclose(
            self._bkd.asarray([fitter.quantile()]),
            self._bkd.asarray([0.75]),
        )

    def test_quantile_conservative_500_trials(self) -> None:
        """ConservativeQuantileFitter: risk(surrogate) >= risk(data) for 500 trials.

        This is the main conservativeness test, replicating the legacy test.
        Uses AVaR (CVaR) at the specified quantile as the risk measure.
        """
        bkd = self._bkd
        nsamples, nvars = 100, 2
        max_level = 2

        # Create expansion
        expansion = self._create_expansion(nvars=nvars, max_level=max_level)

        # Generate true coefficients for underlying function
        nterms = expansion.nterms()
        true_coefs = bkd.ones((nterms, 1))
        true_expansion = expansion.with_params(true_coefs)

        # Test parameters
        noise_std = 0.1
        ntrials = 500
        quantile = 0.5

        # Create risk measure for comparison
        risk_measure = AverageValueAtRisk(bkd, quantile)

        fitter = ConservativeQuantileFitter(bkd, quantile=quantile)

        for _ in range(ntrials):
            # Generate samples and noisy values
            samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
            noiseless_values = true_expansion(samples)  # (1, nsamples)
            noise = bkd.asarray(np.random.normal(0, noise_std, (1, nsamples)))
            train_values = noiseless_values + noise

            # Fit conservative surrogate
            result = fitter.fit(expansion, samples, train_values)

            # Compute risk of surrogate predictions
            predictions = result(samples)  # (1, nsamples)
            risk_measure.set_samples(predictions)
            surrogate_risk = float(risk_measure())

            # Compute risk of training data
            risk_measure.set_samples(train_values)
            data_risk = float(risk_measure())

            # Conservative condition must hold
            self.assertGreaterEqual(
                surrogate_risk,
                data_risk - 1e-10,  # Small tolerance for numerical precision
                f"Conservative condition violated: "
                f"surrogate_risk={surrogate_risk} < data_risk={data_risk}",
            )

    def test_different_strengths_affect_conservativeness(self) -> None:
        """Higher strength parameter produces more conservative surrogates."""
        bkd = self._bkd
        expansion = self._create_expansion(nvars=2, max_level=3)

        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 100)))
        values = bkd.asarray(np.random.randn(1, 100))

        # Fit with different strengths
        fitter_low = ConservativeLstSqFitter(bkd, strength=0.5)
        fitter_high = ConservativeLstSqFitter(bkd, strength=2.0)

        result_low = fitter_low.fit(expansion, samples, values)
        result_high = fitter_high.fit(expansion, samples, values)

        # Higher strength should give higher constant coefficient (more conservative)
        # The constant term is at index 0
        coef_low = float(result_low.params()[0, 0])
        coef_high = float(result_high.params()[0, 0])

        self.assertGreater(coef_high, coef_low)

    def test_different_quantiles_affect_conservativeness(self) -> None:
        """Higher quantile produces more conservative surrogates."""
        bkd = self._bkd
        expansion = self._create_expansion(nvars=2, max_level=3)

        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 100)))
        values = bkd.asarray(np.random.randn(1, 100))

        # Fit with different quantiles
        fitter_low = ConservativeQuantileFitter(bkd, quantile=0.3)
        fitter_high = ConservativeQuantileFitter(bkd, quantile=0.8)

        result_low = fitter_low.fit(expansion, samples, values)
        result_high = fitter_high.fit(expansion, samples, values)

        # Higher quantile should give higher constant coefficient (more conservative)
        coef_low = float(result_low.params()[0, 0])
        coef_high = float(result_high.params()[0, 0])

        self.assertGreater(coef_high, coef_low)


class TestConservativeFittersNumpy(TestConservativeFitters[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestConservativeFittersTorch(TestConservativeFitters[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
