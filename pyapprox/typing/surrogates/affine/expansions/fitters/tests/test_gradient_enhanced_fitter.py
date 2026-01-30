"""Tests for GradientEnhancedPCEFitter.

Tests focus on:
- Returning DirectSolverResult
- Constraint satisfaction (function value interpolation)
- Coefficient recovery when gradients available
- Multi-QoI rejection
- Underdetermined system rejection
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.surrogates.affine.expansions.fitters.gradient_enhanced import (
    GradientEnhancedPCEFitter,
)
from pyapprox.typing.surrogates.affine.expansions.fitters.results import (
    DirectSolverResult,
)

from pyapprox.typing.surrogates.affine.univariate import create_bases_1d
from pyapprox.typing.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.typing.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.typing.surrogates.affine.expansions import BasisExpansion
from pyapprox.typing.probability import UniformMarginal


class TestGradientEnhancedPCEFitter(Generic[Array], unittest.TestCase):
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

    def test_fit_returns_direct_solver_result(self) -> None:
        """Fit returns DirectSolverResult."""
        bkd = self._bkd
        nvars, max_level = 1, 3

        # Create target expansion
        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.asarray(np.random.randn(nterms, 1))
        target_expansion = target_expansion.with_params(true_coef)

        # Generate samples (nsamples >= nterms for constraints)
        nsamples = nterms + 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        # Get function values and gradients from target
        values = target_expansion(samples)  # (1, nsamples)
        gradients = target_expansion.jacobian_batch(samples)[:, 0, :].T  # (nvars, nsamples)

        # Fit
        fitter = GradientEnhancedPCEFitter(bkd)
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values, gradients)

        self.assertIsInstance(result, DirectSolverResult)

    def test_result_params_shape(self) -> None:
        """Result params have correct shape."""
        bkd = self._bkd
        nvars, max_level = 1, 3

        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.asarray(np.random.randn(nterms, 1))
        target_expansion = target_expansion.with_params(true_coef)

        nsamples = nterms + 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)
        gradients = target_expansion.jacobian_batch(samples)[:, 0, :].T

        fitter = GradientEnhancedPCEFitter(bkd)
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values, gradients)

        self.assertEqual(result.params().shape, (nterms, 1))

    def test_handles_1d_values(self) -> None:
        """Fitter handles 1D values array."""
        bkd = self._bkd
        nvars, max_level = 1, 3

        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.asarray(np.random.randn(nterms, 1))
        target_expansion = target_expansion.with_params(true_coef)

        nsamples = nterms + 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values_2d = target_expansion(samples)
        values_1d = values_2d[0, :]  # flatten to 1D
        gradients = target_expansion.jacobian_batch(samples)[:, 0, :].T

        fitter = GradientEnhancedPCEFitter(bkd)
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values_1d, gradients)

        self.assertEqual(result.params().shape[1], 1)

    def test_constraint_satisfaction(self) -> None:
        """Fitted expansion interpolates function values exactly.

        Replicates legacy test: verify Phi @ coef = y (constraint satisfied).
        """
        bkd = self._bkd
        nvars, max_level = 1, 4

        # Create target expansion
        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.asarray(np.random.randn(nterms, 1))
        target_expansion = target_expansion.with_params(true_coef)

        # Generate samples (exactly nterms for determined constraints)
        nsamples = nterms
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        # Get function values and gradients from target
        values = target_expansion(samples)
        gradients = target_expansion.jacobian_batch(samples)[:, 0, :].T

        # Fit
        fitter = GradientEnhancedPCEFitter(bkd)
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values, gradients)

        # Check constraint satisfaction: Phi @ coef = y
        Phi = fit_expansion.basis_matrix(samples)
        constraint_error = bkd.norm(Phi @ result.params() - values.T)
        bkd.assert_allclose(
            bkd.asarray([float(constraint_error)]),
            bkd.asarray([0.0]),
            atol=1e-10,
        )

    def test_coefficient_recovery(self) -> None:
        """Recover exact coefficients when gradients are available.

        With sufficient samples and exact gradient information, should
        recover the true PCE coefficients exactly.
        """
        bkd = self._bkd
        nvars, max_level = 2, 3

        # Create known target expansion
        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.asarray(np.random.randn(nterms, 1))
        target_expansion = target_expansion.with_params(true_coef)

        # Generate samples (overdetermined)
        nsamples = nterms + 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        # Get exact function values and gradients
        values = target_expansion(samples)
        gradients = target_expansion.jacobian_batch(samples)[:, 0, :].T

        # Fit
        fitter = GradientEnhancedPCEFitter(bkd)
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values, gradients)

        # Should recover coefficients
        bkd.assert_allclose(result.params(), true_coef, rtol=1e-8)

    def test_fitted_surrogate_evaluates(self) -> None:
        """Fitted surrogate can evaluate at new points."""
        bkd = self._bkd
        nvars, max_level = 1, 3

        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.asarray(np.random.randn(nterms, 1))
        target_expansion = target_expansion.with_params(true_coef)

        nsamples = nterms + 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)
        gradients = target_expansion.jacobian_batch(samples)[:, 0, :].T

        fitter = GradientEnhancedPCEFitter(bkd)
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values, gradients)

        # Evaluate at new points
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, 10)))
        predictions = result(test_samples)

        self.assertEqual(predictions.shape, (1, 10))

    def test_multi_qoi_raises(self) -> None:
        """nqoi > 1 raises ValueError."""
        bkd = self._bkd
        nvars, max_level = 1, 3

        expansion = self._create_expansion(nvars=nvars, max_level=max_level, nqoi=2)
        nterms = expansion.nterms()

        nsamples = nterms + 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = bkd.asarray(np.random.randn(2, nsamples))
        gradients = bkd.asarray(np.random.randn(nvars, nsamples))

        fitter = GradientEnhancedPCEFitter(bkd)
        with self.assertRaises(ValueError) as ctx:
            fitter.fit(expansion, samples, values, gradients)
        self.assertIn("nqoi=1", str(ctx.exception))

    def test_underdetermined_raises(self) -> None:
        """nsamples < nterms raises ValueError."""
        bkd = self._bkd
        nvars, max_level = 1, 5

        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.asarray(np.random.randn(nterms, 1))
        target_expansion = target_expansion.with_params(true_coef)

        # Too few samples (nsamples < nterms)
        nsamples = nterms - 2
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)
        gradients = target_expansion.jacobian_batch(samples)[:, 0, :].T

        fitter = GradientEnhancedPCEFitter(bkd)
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        with self.assertRaises(ValueError) as ctx:
            fitter.fit(fit_expansion, samples, values, gradients)
        self.assertIn("samples", str(ctx.exception).lower())

    def test_multivariate_gradient_recovery(self) -> None:
        """Gradient-enhanced fitting works for multivariate polynomials.

        Tests that gradients in multiple dimensions are properly handled.
        """
        bkd = self._bkd
        nvars, max_level = 3, 2

        # Create known target expansion
        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.asarray(np.random.randn(nterms, 1))
        target_expansion = target_expansion.with_params(true_coef)

        # Generate samples
        nsamples = nterms + 5
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        # Get exact function values and gradients
        values = target_expansion(samples)
        # jacobian_batch returns (nsamples, nqoi, nvars), extract (nvars, nsamples)
        gradients = target_expansion.jacobian_batch(samples)[:, 0, :].T

        # Fit
        fitter = GradientEnhancedPCEFitter(bkd)
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values, gradients)

        # Verify at test points
        ntest = 20
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntest)))
        target_values = target_expansion(test_samples)
        fitted_values = result(test_samples)

        bkd.assert_allclose(fitted_values, target_values, rtol=1e-8)

    def test_gradient_matching(self) -> None:
        """Fitted expansion matches gradients at training points.

        Since gradients are part of the objective, the fitted expansion
        should match gradients closely.
        """
        bkd = self._bkd
        nvars, max_level = 2, 3

        # Create known target expansion
        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.asarray(np.random.randn(nterms, 1))
        target_expansion = target_expansion.with_params(true_coef)

        # Generate samples
        nsamples = nterms + 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        # Get exact function values and gradients
        values = target_expansion(samples)
        target_gradients = target_expansion.jacobian_batch(samples)[:, 0, :].T

        # Fit
        fitter = GradientEnhancedPCEFitter(bkd)
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        result = fitter.fit(fit_expansion, samples, values, target_gradients)

        # Check gradient matching at training points
        fitted_gradients = result.surrogate().jacobian_batch(samples)[:, 0, :].T

        bkd.assert_allclose(fitted_gradients, target_gradients, rtol=1e-8)


class TestGradientEnhancedPCEFitterNumpy(
    TestGradientEnhancedPCEFitter[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGradientEnhancedPCEFitterTorch(
    TestGradientEnhancedPCEFitter[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
