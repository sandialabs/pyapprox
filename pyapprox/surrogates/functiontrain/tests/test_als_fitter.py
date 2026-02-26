"""Tests for ALSFitter."""

import numpy as np
import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.functiontrain import (
    ALSFitter,
    FunctionTrain,
    create_additive_functiontrain,
)


class TestALSFitter:
    """Base class for ALSFitter tests."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_univariate_expansion(self, bkd, max_level, nqoi=1):
        """Create a univariate polynomial expansion."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def _create_additive_ft(self, bkd, nvars=3, max_level=2, nqoi=1):
        """Create an additive FunctionTrain for testing."""
        univariate_bases = [
            self._create_univariate_expansion(bkd, max_level, nqoi) for _ in range(nvars)
        ]
        return create_additive_functiontrain(univariate_bases, bkd, nqoi)

    def test_als_rejects_non_functiontrain(self, bkd) -> None:
        """Test that ALSFitter raises TypeError for non-FunctionTrain."""
        fitter = ALSFitter(bkd)

        # Try to fit a BasisExpansion instead of FunctionTrain
        expansion = self._create_univariate_expansion(bkd, 2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        values = bkd.asarray(np.random.randn(1, 10))

        with pytest.raises(TypeError) as ctx:
            fitter.fit(expansion, samples, values)

        assert "ALSFitter only works with FunctionTrain" in str(ctx.value)

    def test_als_fits_own_output(self, bkd) -> None:
        """Test ALS can fit FunctionTrain to its own output (exact recovery).

        Following the legacy test pattern: set coefficients, evaluate,
        then fit to that output. This ensures exact recovery is possible.
        """
        nvars = 3
        max_level = 2
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=max_level)

        # Set known coefficients on all trainable expansions
        nparams = ft.nparams()
        true_params = bkd.asarray(np.ones(nparams))
        ft_with_true_params = ft.with_params(true_params)

        # Generate training data from the FT itself
        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = ft_with_true_params(samples)

        # Randomize parameters before fitting
        random_params = bkd.asarray(np.random.randn(nparams))
        ft_randomized = ft.with_params(random_params)

        # Fit with ALS - should recover exact fit
        fitter = ALSFitter(bkd, max_sweeps=10, tol=1e-10)
        result = fitter.fit(ft_randomized, samples, values)

        # Check that fitted model predicts correctly
        fitted_ft = result.surrogate()
        predictions = fitted_ft(samples)

        bkd.assert_allclose(predictions, values, rtol=1e-6, atol=1e-6)

    def test_als_fits_additive_polynomial(self, bkd) -> None:
        """Test ALS fits additive polynomial functions accurately.

        Uses polynomials that are exactly representable by the model.
        """
        nvars = 3
        max_level = 2  # Quadratic polynomials
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=max_level)

        # Set specific coefficients to create a known additive function
        # Each univariate expansion has (max_level + 1) = 3 terms
        nparams = ft.nparams()
        # Create coefficients that give a non-trivial additive function
        params = bkd.asarray(np.random.randn(nparams))
        ft_target = ft.with_params(params)

        # Generate training data from this target
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = ft_target(samples)

        # Start from random initialization
        random_params = bkd.asarray(np.random.randn(nparams) * 0.1)
        ft_init = ft.with_params(random_params)

        fitter = ALSFitter(bkd, max_sweeps=20, tol=1e-12)
        result = fitter.fit(ft_init, samples, values)

        # Check that fitted model predicts correctly
        fitted_ft = result.surrogate()
        predictions = fitted_ft(samples)

        bkd.assert_allclose(predictions, values, rtol=1e-6, atol=1e-6)

    def test_als_converges_for_additive(self, bkd) -> None:
        """Test that ALS converges for additive FT.

        For additive FunctionTrain, the tensor structure allows rapid
        convergence.
        """
        nvars = 3
        max_level = 2
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=max_level)

        # Create target with known params
        nparams = ft.nparams()
        target_params = bkd.asarray(np.ones(nparams) * 2.0)
        ft_target = ft.with_params(target_params)

        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = ft_target(samples)

        # Random initialization
        random_params = bkd.asarray(np.random.randn(nparams))
        ft_init = ft.with_params(random_params)

        # ALS should converge quickly for additive structure
        fitter = ALSFitter(bkd, max_sweeps=5, tol=1e-12)
        result = fitter.fit(ft_init, samples, values)

        fitted_ft = result.surrogate()
        predictions = fitted_ft(samples)

        # Should get very good fit
        bkd.assert_allclose(predictions, values, rtol=1e-6, atol=1e-6)

    def test_als_immutability(self, bkd) -> None:
        """Test that ALS doesn't modify original FunctionTrain."""
        ft = self._create_additive_ft(bkd, nvars=3, max_level=1)

        # Get original parameters
        original_params = bkd.copy(ft._flatten_params())

        # Fit with some data
        nsamples = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (3, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        fitter = ALSFitter(bkd, max_sweeps=5)
        _ = fitter.fit(ft, samples, values)

        # Original should be unchanged
        bkd.assert_allclose(ft._flatten_params(), original_params, rtol=1e-12)

    def test_als_result_contains_surrogate(self, bkd) -> None:
        """Test that ALSFitterResult contains the fitted surrogate."""
        ft = self._create_additive_ft(bkd, nvars=2, max_level=1)

        nsamples = 20
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        fitter = ALSFitter(bkd, max_sweeps=3)
        result = fitter.fit(ft, samples, values)

        # Check result attributes
        assert isinstance(result.surrogate(), FunctionTrain)
        assert isinstance(result.n_sweeps(), int)
        assert isinstance(result.residual_history(), list)
        assert isinstance(result.converged(), bool)

    def test_als_residual_decreases(self, bkd) -> None:
        """Test that residual generally decreases over sweeps."""
        nvars = 3
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=2)

        # Generate noisy data
        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        true_values = samples[0:1, :] + samples[1:2, :] ** 2
        noise = bkd.asarray(np.random.randn(1, nsamples) * 0.01)
        values = true_values + noise

        fitter = ALSFitter(bkd, max_sweeps=10, tol=1e-12)
        result = fitter.fit(ft, samples, values)

        history = result.residual_history()

        # Initial residual should be larger than final
        assert history[0] > history[-1]

    def test_als_handles_1d_values(self, bkd) -> None:
        """Test that ALS handles 1D values array."""
        ft = self._create_additive_ft(bkd, nvars=2, max_level=1)

        nsamples = 20
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values_1d = bkd.asarray(np.random.randn(nsamples))  # 1D

        fitter = ALSFitter(bkd, max_sweeps=3)
        result = fitter.fit(ft, samples, values_1d)

        # Should work without error
        assert isinstance(result.surrogate(), FunctionTrain)

    def test_als_validates_dimension_mismatch(self, bkd) -> None:
        """Test that ALS validates input dimensions."""
        ft = self._create_additive_ft(bkd, nvars=3, max_level=1)

        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))  # Wrong nvars
        values = bkd.asarray(np.random.randn(1, 10))

        fitter = ALSFitter(bkd)

        with pytest.raises(ValueError):
            fitter.fit(ft, samples, values)

    def test_als_convergence_flag(self, bkd) -> None:
        """Test that convergence flag is set correctly."""
        ft = self._create_additive_ft(bkd, nvars=2, max_level=1)

        # Simple linear function that should converge quickly
        nsamples = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values = samples[0:1, :]  # Just x_0

        # With very tight tolerance and few sweeps - may not converge
        fitter_strict = ALSFitter(bkd, max_sweeps=1, tol=1e-20)
        fitter_strict.fit(ft, samples, values)

        # With loose tolerance - should converge
        fitter_loose = ALSFitter(bkd, max_sweeps=10, tol=1.0)
        result_loose = fitter_loose.fit(ft, samples, values)

        # Loose tolerance should converge
        assert result_loose.converged()
