"""Tests for EntropicFitter and EntropicLoss.

Tests focus on:
- EntropicLoss jacobian and hvp via DerivativeChecker
- EntropicFitter fitting and entropic risk statistic = 0 at solution
- nqoi=1 restriction
- Result types and shapes
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

from pyapprox.surrogates.affine.expansions.fitters.entropic import (
    EntropicLoss,
    EntropicFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    DirectSolverResult,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)

from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.probability import UniformMarginal


class TestEntropicLoss(Generic[Array], unittest.TestCase):
    """Tests for EntropicLoss objective function."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_loss(
        self,
        nsamples: int = 100,
        nterms: int = 3,
        strength: float = 1.0,
    ) -> EntropicLoss[Array]:
        """Create a test EntropicLoss instance."""
        bkd = self._bkd
        basis_matrix = bkd.asarray(np.random.normal(0, 1, (nsamples, nterms)))
        # Make first column constant (like polynomial expansion)
        basis_matrix = bkd.copy(basis_matrix)
        basis_matrix[:, 0] = 1.0
        train_values = bkd.asarray(np.random.normal(0, 1, (nsamples, 1)))
        return EntropicLoss(basis_matrix, train_values, bkd, strength=strength)

    def test_jacobian_via_derivative_checker(self) -> None:
        """Jacobian passes DerivativeChecker."""
        loss = self._create_loss(nsamples=50, nterms=5)
        coefs = self._bkd.asarray(np.random.normal(0, 1, (5, 1)))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(coefs)

        # Jacobian is first in errors list
        ratio = checker.error_ratio(errors[0])
        self._bkd.assert_allclose(
            self._bkd.asarray([float(ratio)]),
            self._bkd.asarray([0.0]),
            atol=1e-6,
        )

    def test_hvp_via_derivative_checker(self) -> None:
        """HVP passes DerivativeChecker."""
        loss = self._create_loss(nsamples=50, nterms=5)
        coefs = self._bkd.asarray(np.random.normal(0, 1, (5, 1)))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(coefs)

        # HVP is second in errors list (when function has hvp)
        self.assertEqual(len(errors), 2)
        ratio = checker.error_ratio(errors[1])
        self._bkd.assert_allclose(
            self._bkd.asarray([float(ratio)]),
            self._bkd.asarray([0.0]),
            atol=1e-6,
        )

    def test_nqoi_is_one(self) -> None:
        """nqoi() returns 1."""
        loss = self._create_loss()
        self.assertEqual(loss.nqoi(), 1)

    def test_nvars_matches_nterms(self) -> None:
        """nvars() matches number of basis terms."""
        loss = self._create_loss(nterms=7)
        self.assertEqual(loss.nvars(), 7)

    def test_call_output_shape(self) -> None:
        """__call__ returns correct shape."""
        loss = self._create_loss(nterms=4)
        coefs = self._bkd.asarray(np.random.normal(0, 1, (4, 3)))
        result = loss(coefs)
        self.assertEqual(result.shape, (1, 3))

    def test_jacobian_output_shape(self) -> None:
        """jacobian returns correct shape."""
        loss = self._create_loss(nterms=5)
        coef = self._bkd.asarray(np.random.normal(0, 1, (5, 1)))
        jac = loss.jacobian(coef)
        self.assertEqual(jac.shape, (1, 5))

    def test_hvp_output_shape(self) -> None:
        """hvp returns correct shape."""
        loss = self._create_loss(nterms=5)
        coef = self._bkd.asarray(np.random.normal(0, 1, (5, 1)))
        vec = self._bkd.asarray(np.random.normal(0, 1, (5, 1)))
        hvp_result = loss.hvp(coef, vec)
        self.assertEqual(hvp_result.shape, (5, 1))

    def test_loss_nonnegative(self) -> None:
        """Entropic loss is always non-negative."""
        loss = self._create_loss()
        coefs = self._bkd.asarray(np.random.normal(0, 1, (3, 10)))
        values = loss(coefs)
        # Check all values are >= -1e-10 (small tolerance for numerical precision)
        min_val = self._bkd.min(values)
        self.assertGreaterEqual(float(min_val), -1e-10)

    def test_weights_validation(self) -> None:
        """Invalid weights shape raises ValueError."""
        bkd = self._bkd
        basis_matrix = bkd.asarray(np.random.normal(0, 1, (10, 3)))
        train_values = bkd.asarray(np.random.normal(0, 1, (10, 1)))
        wrong_weights = bkd.asarray(np.ones((5, 1)))  # Wrong shape

        with self.assertRaises(ValueError) as ctx:
            EntropicLoss(basis_matrix, train_values, bkd, wrong_weights)
        self.assertIn("weights", str(ctx.exception))

    def test_train_values_validation(self) -> None:
        """Invalid train_values shape raises ValueError."""
        bkd = self._bkd
        basis_matrix = bkd.asarray(np.random.normal(0, 1, (10, 3)))
        wrong_values = bkd.asarray(np.random.normal(0, 1, (5, 1)))  # Wrong shape

        with self.assertRaises(ValueError) as ctx:
            EntropicLoss(basis_matrix, wrong_values, bkd)
        self.assertIn("train_values", str(ctx.exception))

    def test_strength_accessor(self) -> None:
        """strength() returns correct value."""
        loss = self._create_loss(strength=2.0)
        self._bkd.assert_allclose(
            self._bkd.asarray([loss.strength()]),
            self._bkd.asarray([2.0]),
        )

    def test_jacobian_with_beta_2(self) -> None:
        """Jacobian passes DerivativeChecker with beta=2."""
        loss = self._create_loss(nsamples=50, nterms=5, strength=2.0)
        coefs = self._bkd.asarray(np.random.normal(0, 0.5, (5, 1)))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(coefs)

        ratio = checker.error_ratio(errors[0])
        self._bkd.assert_allclose(
            self._bkd.asarray([float(ratio)]),
            self._bkd.asarray([0.0]),
            atol=1e-6,
        )

    def test_hvp_with_beta_2(self) -> None:
        """HVP passes DerivativeChecker with beta=2."""
        loss = self._create_loss(nsamples=50, nterms=5, strength=2.0)
        coefs = self._bkd.asarray(np.random.normal(0, 0.5, (5, 1)))

        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(coefs)

        self.assertEqual(len(errors), 2)
        ratio = checker.error_ratio(errors[1])
        self._bkd.assert_allclose(
            self._bkd.asarray([float(ratio)]),
            self._bkd.asarray([0.0]),
            atol=1e-6,
        )


class TestEntropicFitter(Generic[Array], unittest.TestCase):
    """Tests for EntropicFitter."""

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

    def test_fit_returns_direct_solver_result(self) -> None:
        """Fit returns DirectSolverResult."""
        expansion = self._create_expansion(nvars=1, max_level=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        values = self._bkd.asarray(np.random.randn(1, 50))

        fitter = EntropicFitter(self._bkd)
        result = fitter.fit(expansion, samples, values)

        self.assertIsInstance(result, DirectSolverResult)

    def test_result_params_shape(self) -> None:
        """Result params have correct shape."""
        expansion = self._create_expansion(nvars=1, max_level=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        values = self._bkd.asarray(np.random.randn(1, 50))

        fitter = EntropicFitter(self._bkd)
        result = fitter.fit(expansion, samples, values)

        self.assertEqual(result.params().shape, (expansion.nterms(), 1))

    def test_handles_1d_values(self) -> None:
        """Fitter handles 1D values array."""
        expansion = self._create_expansion(nvars=1, max_level=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        values_1d = self._bkd.asarray(np.random.randn(50))

        fitter = EntropicFitter(self._bkd)
        result = fitter.fit(expansion, samples, values_1d)

        self.assertEqual(result.params().shape[1], 1)

    def test_multi_qoi_raises(self) -> None:
        """nqoi > 1 raises ValueError."""
        expansion = self._create_expansion(nvars=1, max_level=2, nqoi=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        values = self._bkd.asarray(np.random.randn(2, 50))

        fitter = EntropicFitter(self._bkd)
        with self.assertRaises(ValueError) as ctx:
            fitter.fit(expansion, samples, values)
        self.assertIn("nqoi=1", str(ctx.exception))

    def test_fitted_surrogate_evaluates(self) -> None:
        """Fitted surrogate can evaluate at new points."""
        expansion = self._create_expansion(nvars=1, max_level=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        values = self._bkd.asarray(np.random.randn(1, 50))

        fitter = EntropicFitter(self._bkd)
        result = fitter.fit(expansion, samples, values)

        test_samples = self._bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        predictions = result(test_samples)

        self.assertEqual(predictions.shape, (1, 10))

    def test_entropic_risk_statistic_zero_at_solution_beta_1(self) -> None:
        """At the optimum with beta=1, entropic risk of residuals equals zero.

        This is the key property from the Entropic Risk Quadrangle:
        the optimal solution makes the entropic risk measure equal zero.
        Uses the same approach as the legacy test_entropic_regression.
        """
        bkd = self._bkd
        nvars = 1
        max_level = 2

        # Create expansion
        expansion = self._create_expansion(nvars=nvars, max_level=max_level)

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

    def test_entropic_risk_statistic_zero_at_solution_beta_2(self) -> None:
        """At the optimum with beta=2, entropic risk of residuals equals zero."""
        bkd = self._bkd
        nvars = 1
        max_level = 2

        # Create expansion
        expansion = self._create_expansion(nvars=nvars, max_level=max_level)

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


class TestEntropicLossNumpy(TestEntropicLoss[NDArray[Any]]):
    """NumPy backend tests for EntropicLoss."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestEntropicLossTorch(TestEntropicLoss[torch.Tensor]):
    """PyTorch backend tests for EntropicLoss."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestEntropicFitterNumpy(TestEntropicFitter[NDArray[Any]]):
    """NumPy backend tests for EntropicFitter."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestEntropicFitterTorch(TestEntropicFitter[torch.Tensor]):
    """PyTorch backend tests for EntropicFitter."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
