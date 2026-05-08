"""Tests for EntropicFitter and EntropicLoss.

Tests focus on:
- EntropicLoss jacobian and hvp via DerivativeChecker
- EntropicFitter fitting and entropic risk statistic = 0 at solution
- nqoi=1 restriction
- Result types and shapes
"""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.expansions.fitters.entropic import (
    EntropicFitter,
    EntropicLoss,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    DirectSolverResult,
)
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d


class TestEntropicLoss:
    """Tests for EntropicLoss objective function."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_loss(self, bkd, nsamples=100, nterms=3, strength=1.0):
        """Create a test EntropicLoss instance."""
        basis_matrix = bkd.asarray(np.random.normal(0, 1, (nsamples, nterms)))
        # Make first column constant (like polynomial expansion)
        basis_matrix = bkd.copy(basis_matrix)
        basis_matrix[:, 0] = 1.0
        train_values = bkd.asarray(np.random.normal(0, 1, (nsamples, 1)))
        return EntropicLoss(basis_matrix, train_values, bkd, strength=strength)

    def test_jacobian_via_derivative_checker(self, bkd) -> None:
        """Jacobian passes DerivativeChecker."""
        loss = self._create_loss(bkd, nsamples=50, nterms=5)
        coefs = bkd.asarray(np.random.normal(0, 1, (5, 1)))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(coefs)

        # Jacobian is first in errors list
        ratio = checker.error_ratio(errors[0])
        bkd.assert_allclose(
            bkd.asarray([float(ratio)]),
            bkd.asarray([0.0]),
            atol=1e-6,
        )

    def test_hvp_via_derivative_checker(self, bkd) -> None:
        """HVP passes DerivativeChecker."""
        loss = self._create_loss(bkd, nsamples=50, nterms=5)
        coefs = bkd.asarray(np.random.normal(0, 1, (5, 1)))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(coefs)

        # HVP is second in errors list (when function has hvp)
        assert len(errors) == 2
        ratio = checker.error_ratio(errors[1])
        bkd.assert_allclose(
            bkd.asarray([float(ratio)]),
            bkd.asarray([0.0]),
            atol=1e-6,
        )

    def test_nqoi_is_one(self, bkd) -> None:
        """nqoi() returns 1."""
        loss = self._create_loss(bkd)
        assert loss.nqoi() == 1

    def test_nvars_matches_nterms(self, bkd) -> None:
        """nvars() matches number of basis terms."""
        loss = self._create_loss(bkd, nterms=7)
        assert loss.nvars() == 7

    def test_call_output_shape(self, bkd) -> None:
        """__call__ returns correct shape."""
        loss = self._create_loss(bkd, nterms=4)
        coefs = bkd.asarray(np.random.normal(0, 1, (4, 3)))
        result = loss(coefs)
        assert result.shape == (1, 3)

    def test_jacobian_output_shape(self, bkd) -> None:
        """jacobian returns correct shape."""
        loss = self._create_loss(bkd, nterms=5)
        coef = bkd.asarray(np.random.normal(0, 1, (5, 1)))
        jac = loss.jacobian(coef)
        assert jac.shape == (1, 5)

    def test_hvp_output_shape(self, bkd) -> None:
        """hvp returns correct shape."""
        loss = self._create_loss(bkd, nterms=5)
        coef = bkd.asarray(np.random.normal(0, 1, (5, 1)))
        vec = bkd.asarray(np.random.normal(0, 1, (5, 1)))
        hvp_result = loss.hvp(coef, vec)
        assert hvp_result.shape == (5, 1)

    def test_loss_nonnegative(self, bkd) -> None:
        """Entropic loss is always non-negative."""
        loss = self._create_loss(bkd)
        coefs = bkd.asarray(np.random.normal(0, 1, (3, 10)))
        values = loss(coefs)
        # Check all values are >= -1e-10 (small tolerance for numerical precision)
        min_val = bkd.min(values)
        assert float(min_val) >= -1e-10

    def test_weights_validation(self, bkd) -> None:
        """Invalid weights shape raises ValueError."""
        basis_matrix = bkd.asarray(np.random.normal(0, 1, (10, 3)))
        train_values = bkd.asarray(np.random.normal(0, 1, (10, 1)))
        wrong_weights = bkd.asarray(np.ones((5, 1)))  # Wrong shape

        with pytest.raises(ValueError, match="weights"):
            EntropicLoss(basis_matrix, train_values, bkd, wrong_weights)

    def test_train_values_validation(self, bkd) -> None:
        """Invalid train_values shape raises ValueError."""
        basis_matrix = bkd.asarray(np.random.normal(0, 1, (10, 3)))
        wrong_values = bkd.asarray(np.random.normal(0, 1, (5, 1)))  # Wrong shape

        with pytest.raises(ValueError, match="train_values"):
            EntropicLoss(basis_matrix, wrong_values, bkd)

    def test_strength_accessor(self, bkd) -> None:
        """strength() returns correct value."""
        loss = self._create_loss(bkd, strength=2.0)
        bkd.assert_allclose(
            bkd.asarray([loss.strength()]),
            bkd.asarray([2.0]),
        )

    def test_jacobian_with_beta_2(self, bkd) -> None:
        """Jacobian passes DerivativeChecker with beta=2."""
        loss = self._create_loss(bkd, nsamples=50, nterms=5, strength=2.0)
        coefs = bkd.asarray(np.random.normal(0, 0.5, (5, 1)))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(coefs)

        ratio = checker.error_ratio(errors[0])
        bkd.assert_allclose(
            bkd.asarray([float(ratio)]),
            bkd.asarray([0.0]),
            atol=1e-6,
        )

    def test_hvp_with_beta_2(self, bkd) -> None:
        """HVP passes DerivativeChecker with beta=2."""
        loss = self._create_loss(bkd, nsamples=50, nterms=5, strength=2.0)
        coefs = bkd.asarray(np.random.normal(0, 0.5, (5, 1)))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(coefs)

        assert len(errors) == 2
        ratio = checker.error_ratio(errors[1])
        bkd.assert_allclose(
            bkd.asarray([float(ratio)]),
            bkd.asarray([0.0]),
            atol=1e-6,
        )


class TestEntropicFitter:
    """Tests for EntropicFitter."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_expansion(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        """Create test expansion."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_fit_returns_direct_solver_result(self, bkd) -> None:
        """Fit returns DirectSolverResult."""
        expansion = self._create_expansion(bkd, nvars=1, max_level=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        values = bkd.asarray(np.random.randn(1, 50))

        fitter = EntropicFitter(bkd)
        result = fitter.fit(expansion, samples, values)

        assert isinstance(result, DirectSolverResult)

    def test_result_params_shape(self, bkd) -> None:
        """Result params have correct shape."""
        expansion = self._create_expansion(bkd, nvars=1, max_level=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        values = bkd.asarray(np.random.randn(1, 50))

        fitter = EntropicFitter(bkd)
        result = fitter.fit(expansion, samples, values)

        assert result.params().shape == (expansion.nterms(), 1)

    def test_handles_1d_values(self, bkd) -> None:
        """Fitter handles 1D values array."""
        expansion = self._create_expansion(bkd, nvars=1, max_level=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        values_1d = bkd.asarray(np.random.randn(50))

        fitter = EntropicFitter(bkd)
        result = fitter.fit(expansion, samples, values_1d)

        assert result.params().shape[1] == 1

    def test_multi_qoi_raises(self, bkd) -> None:
        """nqoi > 1 raises ValueError."""
        expansion = self._create_expansion(bkd, nvars=1, max_level=2, nqoi=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        values = bkd.asarray(np.random.randn(2, 50))

        fitter = EntropicFitter(bkd)
        with pytest.raises(ValueError, match="nqoi=1"):
            fitter.fit(expansion, samples, values)

    def test_fitted_surrogate_evaluates(self, bkd) -> None:
        """Fitted surrogate can evaluate at new points."""
        expansion = self._create_expansion(bkd, nvars=1, max_level=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        values = bkd.asarray(np.random.randn(1, 50))

        fitter = EntropicFitter(bkd)
        result = fitter.fit(expansion, samples, values)

        test_samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        predictions = result(test_samples)

        assert predictions.shape == (1, 10)

    def test_entropic_risk_statistic_zero_at_solution_beta_1(self, bkd) -> None:
        """At the optimum with beta=1, entropic risk of residuals equals zero.

        This is the key property from the Entropic Risk Quadrangle:
        the optimal solution makes the entropic risk measure equal zero.
        Uses the same approach as the legacy test_entropic_regression.
        """
        nvars = 1
        max_level = 2

        # Create expansion
        expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)

        # Generate training data with noise
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        # Create true polynomial and add noise
        true_coef = bkd.asarray(np.random.randn(expansion.nterms(), 1))
        true_expansion = expansion.with_params(true_coef)
        noise_std = 0.1
        noise = bkd.asarray(np.random.normal(0, noise_std, (1, nsamples)))
        values = true_expansion(samples) + noise

        # Fit with entropic fitter (beta=1)
        beta = 1.0
        fitter = EntropicFitter(bkd, strength=beta, gtol=1e-12)
        result = fitter.fit(expansion, samples, values)

        # Compute residuals
        fitted_values = result(samples)
        residuals = values - fitted_values  # (1, nsamples)

        # Compute entropic risk: log(E[exp(beta*r)]) / beta
        # At optimum this should be zero
        entropic_risk = bkd.log(bkd.mean(bkd.exp(beta * residuals))) / beta

        bkd.assert_allclose(
            bkd.reshape(entropic_risk, (1,)),
            bkd.zeros((1,)),
            atol=1e-4,
        )

    def test_entropic_risk_statistic_zero_at_solution_beta_2(self, bkd) -> None:
        """At the optimum with beta=2, entropic risk of residuals equals zero."""
        nvars = 1
        max_level = 2

        # Create expansion
        expansion = self._create_expansion(bkd, nvars=nvars, max_level=max_level)

        # Generate training data with noise
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        # Create true polynomial and add noise
        true_coef = bkd.asarray(np.random.randn(expansion.nterms(), 1))
        true_expansion = expansion.with_params(true_coef)
        noise_std = 0.1
        noise = bkd.asarray(np.random.normal(0, noise_std, (1, nsamples)))
        values = true_expansion(samples) + noise

        # Fit with entropic fitter (beta=2)
        beta = 2.0
        fitter = EntropicFitter(bkd, strength=beta, gtol=1e-12)
        result = fitter.fit(expansion, samples, values)

        # Compute residuals
        fitted_values = result(samples)
        residuals = values - fitted_values  # (1, nsamples)

        # Compute entropic risk: log(E[exp(beta*r)]) / beta
        # At optimum this should be zero
        entropic_risk = bkd.log(bkd.mean(bkd.exp(beta * residuals))) / beta

        bkd.assert_allclose(
            bkd.reshape(entropic_risk, (1,)),
            bkd.zeros((1,)),
            atol=1e-4,
        )
