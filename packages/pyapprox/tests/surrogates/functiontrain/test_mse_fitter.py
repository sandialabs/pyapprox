"""Tests for MSEFitter.

Tests validate the gradient-based MSE fitter for FunctionTrain surrogates.
Covers convergence, accuracy compared to ALS, and immutability.
"""

import numpy as np
import pytest

from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.functiontrain import (
    ALSFitter,
    FunctionTrain,
    MSEFitter,
    create_additive_functiontrain,
)


class TestMSEFitter:
    """Tests for MSEFitter."""

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
            self._create_univariate_expansion(bkd, max_level, nqoi)
            for _ in range(nvars)
        ]
        return create_additive_functiontrain(univariate_bases, bkd, nqoi)

    def test_mse_rejects_non_functiontrain(self, bkd) -> None:
        """Test that MSEFitter raises TypeError for non-FunctionTrain."""
        fitter = MSEFitter(bkd)

        # Try to fit a BasisExpansion instead of FunctionTrain
        expansion = self._create_univariate_expansion(bkd, 2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        values = bkd.asarray(np.random.randn(1, 10))

        with pytest.raises(TypeError) as ctx:
            fitter.fit(expansion, samples, values)

        assert "MSEFitter only works with FunctionTrain" in str(ctx.value)

    def test_mse_fits_own_output(self, bkd) -> None:
        """Test MSE can fit FunctionTrain to its own output (exact recovery)."""
        nvars = 3
        max_level = 1
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=max_level)

        # Set known coefficients
        nparams = ft.nparams()
        true_params = bkd.asarray(np.ones(nparams))
        ft_with_true_params = ft.with_params(true_params)

        # Generate training data from the FT itself
        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = ft_with_true_params(samples)

        # Randomize parameters before fitting
        random_params = bkd.asarray(np.random.randn(nparams) * 0.1)
        ft_randomized = ft.with_params(random_params)

        # Fit with MSE
        fitter = MSEFitter(bkd)
        result = fitter.fit(ft_randomized, samples, values)

        # Check that fitted model predicts correctly
        fitted_ft = result.surrogate()
        predictions = fitted_ft(samples)

        bkd.assert_allclose(predictions, values, rtol=1e-5, atol=1e-5)

    def test_mse_fits_additive_polynomial(self, bkd) -> None:
        """Test MSE fits additive polynomial functions accurately."""
        nvars = 3
        max_level = 1
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=max_level)

        # Set specific coefficients
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft_target = ft.with_params(params)

        # Generate training data
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = ft_target(samples)

        # Start from random initialization
        random_params = bkd.asarray(np.random.randn(nparams) * 0.1)
        ft_init = ft.with_params(random_params)

        fitter = MSEFitter(bkd)
        result = fitter.fit(ft_init, samples, values)

        # Check that fitted model predicts correctly
        fitted_ft = result.surrogate()
        predictions = fitted_ft(samples)

        bkd.assert_allclose(predictions, values, rtol=1e-5, atol=1e-5)

    def test_mse_immutability(self, bkd) -> None:
        """Test that MSE doesn't modify original FunctionTrain."""
        ft = self._create_additive_ft(bkd, nvars=3, max_level=1)

        # Get original parameters
        original_params = bkd.copy(ft._flatten_params())

        # Fit with some data
        nsamples = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (3, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        fitter = MSEFitter(bkd)
        _ = fitter.fit(ft, samples, values)

        # Original should be unchanged
        bkd.assert_allclose(ft._flatten_params(), original_params, rtol=1e-12)

    def test_mse_result_attributes(self, bkd) -> None:
        """Test that MSEFitterResult contains expected attributes."""
        ft = self._create_additive_ft(bkd, nvars=2, max_level=1)

        nsamples = 20
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        fitter = MSEFitter(bkd)
        result = fitter.fit(ft, samples, values)

        # Check result attributes
        assert isinstance(result.surrogate(), FunctionTrain)
        assert isinstance(result.final_loss(), float)
        assert isinstance(result.converged(), bool)
        # n_iterations is optimizer-specific, access via raw result
        raw_result = result.optimizer_result().get_raw_result()
        assert hasattr(raw_result, "nit")

    def test_mse_validates_dimension_mismatch(self, bkd) -> None:
        """Test that MSE validates input dimensions."""
        ft = self._create_additive_ft(bkd, nvars=3, max_level=1)

        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))  # Wrong nvars
        values = bkd.asarray(np.random.randn(1, 10))

        fitter = MSEFitter(bkd)

        with pytest.raises(ValueError):
            fitter.fit(ft, samples, values)

    def test_mse_handles_1d_values(self, bkd) -> None:
        """Test that MSE handles 1D values array."""
        ft = self._create_additive_ft(bkd, nvars=2, max_level=1)

        nsamples = 20
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values_1d = bkd.asarray(np.random.randn(nsamples))  # 1D

        fitter = MSEFitter(bkd)
        result = fitter.fit(ft, samples, values_1d)

        # Should work without error
        assert isinstance(result.surrogate(), FunctionTrain)

    def test_mse_custom_optimizer(self, bkd) -> None:
        """Test MSE with custom optimizer settings."""
        nvars = 2
        max_level = 1
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=max_level)

        nsamples = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        # Configure custom optimizer
        optimizer = ScipyTrustConstrOptimizer(verbosity=0, maxiter=100, gtol=1e-6)

        fitter = MSEFitter(bkd)
        fitter.set_optimizer(optimizer)

        # Verify optimizer is set
        assert fitter.optimizer() is not None

        result = fitter.fit(ft, samples, values)
        assert isinstance(result.surrogate(), FunctionTrain)

    def test_mse_loss_decreases(self, bkd) -> None:
        """Test that final loss is less than initial loss."""
        nvars = 2
        max_level = 1
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=max_level)

        nsamples = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        # Generate target from random params
        nparams = ft.nparams()
        target_params = bkd.asarray(np.random.randn(nparams))
        ft_target = ft.with_params(target_params)
        values = ft_target(samples)

        # Start far from solution
        init_params = bkd.asarray(np.random.randn(nparams) * 5.0)
        ft_init = ft.with_params(init_params)

        # Compute initial loss
        from pyapprox.surrogates.functiontrain import FunctionTrainMSELoss

        loss = FunctionTrainMSELoss(ft_init, samples, values, bkd)
        initial_loss = float(loss(init_params)[0, 0])

        # Fit
        fitter = MSEFitter(bkd)
        result = fitter.fit(ft_init, samples, values)

        # Final loss should be less than initial
        assert result.final_loss() < initial_loss

    def test_mse_comparable_to_als(self, bkd) -> None:
        """Test that MSE achieves similar accuracy to ALS on simple problems."""
        nvars = 3
        max_level = 1
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=max_level)

        # Set known parameters
        nparams = ft.nparams()
        true_params = bkd.asarray(np.ones(nparams) * 2.0)
        ft_target = ft.with_params(true_params)

        # Generate training data
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = ft_target(samples)

        # Random initialization
        random_params = bkd.asarray(np.random.randn(nparams) * 0.1)
        ft_init = ft.with_params(random_params)

        # Fit with ALS
        als_fitter = ALSFitter(bkd, max_sweeps=10, tol=1e-10)
        als_result = als_fitter.fit(ft_init, samples, values)
        als_predictions = als_result.surrogate()(samples)

        # Fit with MSE
        mse_fitter = MSEFitter(bkd)
        mse_result = mse_fitter.fit(ft_init, samples, values)
        mse_predictions = mse_result.surrogate()(samples)

        # Both should achieve good fit
        bkd.assert_allclose(als_predictions, values, rtol=1e-4, atol=1e-4)
        bkd.assert_allclose(mse_predictions, values, rtol=1e-4, atol=1e-4)
