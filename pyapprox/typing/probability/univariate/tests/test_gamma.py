"""
Tests for GammaMarginal distribution.
"""

import unittest
from typing import Any, Generic, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests
from pyapprox.typing.probability.univariate import GammaMarginal
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.typing.surrogates.affine.univariate.globalpoly import (
    LegendrePolynomial1D,
    GaussQuadratureRule,
)


class GaussLegendreQuadrature01(Generic[Array]):
    """Gauss-Legendre quadrature on [0, 1] with Lebesgue measure for tests."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._legendre = LegendrePolynomial1D(bkd)
        self._quad_rule = GaussQuadratureRule(self._legendre, store=True)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __call__(self, npoints: int) -> Tuple[Array, Array]:
        points_11, weights_11 = self._quad_rule(npoints)
        # Map from [-1, 1] to [0, 1]
        # GaussQuadratureRule integrates uniform probability measure on [-1,1]
        # (weights sum to 1). For [0,1] we just shift points; weights stay same.
        points_01 = (points_11 + 1.0) / 2.0
        return points_01, weights_11


class TestGammaMarginal(Generic[Array], unittest.TestCase):
    """Tests for GammaMarginal."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._shape = 2.0
        self._scale = 1.5
        # Create quadrature rule on [0, 1] for CDF/invcdf tests
        self._quad_rule = GaussLegendreQuadrature01(self._bkd)
        self._dist = GammaMarginal(
            self._shape, self._scale, self._bkd, quadrature_rule=self._quad_rule
        )
        self._scipy_dist = stats.gamma(self._shape, scale=self._scale)

    def test_nvars(self) -> None:
        """Test nvars returns 1."""
        self.assertEqual(self._dist.nvars(), 1)

    def test_shape_parameters(self) -> None:
        """Test shape and scale parameter accessors."""
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([self._dist.shape()]),
                self._bkd.asarray([self._shape]),
                atol=1e-10,
            )
        )
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([self._dist.scale()]),
                self._bkd.asarray([self._scale]),
                atol=1e-10,
            )
        )
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([self._dist.rate()]),
                self._bkd.asarray([1.0 / self._scale]),
                atol=1e-10,
            )
        )

    def test_pdf_matches_scipy(self) -> None:
        """Test PDF matches scipy."""
        samples = self._bkd.asarray([[0.5, 1.0, 2.0, 3.0]])
        pdf_vals = self._dist(samples)
        expected = self._bkd.asarray([self._scipy_dist.pdf([0.5, 1.0, 2.0, 3.0])])
        self.assertTrue(self._bkd.allclose(pdf_vals, expected, rtol=1e-10))

    def test_logpdf_matches_scipy(self) -> None:
        """Test logpdf matches scipy."""
        samples = self._bkd.asarray([[0.5, 1.0, 2.0, 3.0]])
        logpdf_vals = self._dist.logpdf(samples)
        expected = self._bkd.asarray([self._scipy_dist.logpdf([0.5, 1.0, 2.0, 3.0])])
        self.assertTrue(self._bkd.allclose(logpdf_vals, expected, rtol=1e-10))

    def test_cdf_matches_scipy(self) -> None:
        """Test CDF matches scipy."""
        samples = self._bkd.asarray([[0.5, 1.0, 2.0, 3.0]])
        cdf_vals = self._dist.cdf(samples)
        expected = self._bkd.asarray([self._scipy_dist.cdf([0.5, 1.0, 2.0, 3.0])])
        self.assertTrue(self._bkd.allclose(cdf_vals, expected, rtol=1e-10))

    def test_invcdf_matches_scipy(self) -> None:
        """Test invcdf matches scipy ppf."""
        probs = self._bkd.asarray([[0.1, 0.5, 0.9]])
        invcdf_vals = self._dist.invcdf(probs)
        expected = self._bkd.asarray([self._scipy_dist.ppf([0.1, 0.5, 0.9])])
        self.assertTrue(self._bkd.allclose(invcdf_vals, expected, rtol=1e-10))

    def test_cdf_invcdf_inverse(self) -> None:
        """Test cdf(invcdf(p)) = p."""
        probs = self._bkd.asarray([[0.1, 0.5, 0.9]])
        recovered = self._dist.cdf(self._dist.invcdf(probs))
        self.assertTrue(self._bkd.allclose(recovered, probs, rtol=1e-6))

    def test_invcdf_cdf_inverse(self) -> None:
        """Test invcdf(cdf(x)) = x."""
        samples = self._bkd.asarray([[0.5, 1.0, 2.0]])
        recovered = self._dist.invcdf(self._dist.cdf(samples))
        self.assertTrue(self._bkd.allclose(recovered, samples, rtol=1e-6))

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        samples = self._dist.rvs(100)
        self.assertEqual(samples.shape, (1, 100))

    def test_rvs_positive(self) -> None:
        """Test rvs produces positive samples."""
        samples = self._dist.rvs(1000)
        samples_flat = self._bkd.flatten(samples)
        self.assertTrue(self._bkd.all_bool(samples_flat >= 0.0))

    def test_logpdf_exp_equals_pdf(self) -> None:
        """Test exp(logpdf) = pdf."""
        samples = self._bkd.asarray([[0.5, 1.0, 2.0]])
        pdf_vals = self._dist(samples)
        logpdf_vals = self._dist.logpdf(samples)
        self.assertTrue(
            self._bkd.allclose(self._bkd.exp(logpdf_vals), pdf_vals, rtol=1e-10)
        )

    def test_mean_variance(self) -> None:
        """Test mean and variance match scipy."""
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([self._dist.mean_value()]),
                self._bkd.asarray([self._scipy_dist.mean()]),
                atol=1e-10,
            )
        )
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([self._dist.variance()]),
                self._bkd.asarray([self._scipy_dist.var()]),
                atol=1e-10,
            )
        )
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([self._dist.std()]),
                self._bkd.asarray([self._scipy_dist.std()]),
                atol=1e-10,
            )
        )

    def test_is_bounded(self) -> None:
        """Test Gamma is not bounded (unbounded on right)."""
        self.assertFalse(self._dist.is_bounded())

    def test_bounds(self) -> None:
        """Test bounds are [0, inf]."""
        lower, upper = self._dist.bounds()
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([lower]),
                self._bkd.asarray([0.0]),
                atol=1e-10,
            )
        )
        self.assertEqual(upper, float("inf"))

    def test_interval(self) -> None:
        """Test interval contains correct probability."""
        alpha = 0.95
        interval = self._dist.interval(alpha)
        # interval returns shape (1, 2) with [lower, upper] bounds
        self.assertEqual(interval.shape, (1, 2))
        lower, upper = float(interval[0, 0]), float(interval[0, 1])
        self.assertGreater(lower, 0.0)
        self.assertLess(lower, upper)
        # Check probability content
        prob = self._scipy_dist.cdf(upper) - self._scipy_dist.cdf(lower)
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([prob]),
                self._bkd.asarray([alpha]),
                atol=1e-6,
            )
        )

    def test_invcdf_jacobian(self) -> None:
        """Test invcdf Jacobian = 1/pdf at quantile."""
        probs = self._bkd.asarray([[0.25, 0.5, 0.75]])
        jacobian = self._dist.invcdf_jacobian(probs)
        quantiles = self._dist.invcdf(probs)
        pdf_at_quantiles = self._dist(quantiles)
        self.assertTrue(
            self._bkd.allclose(jacobian, 1.0 / pdf_at_quantiles, rtol=1e-6)
        )

    def test_logpdf_jacobian_derivative_checker(self) -> None:
        """Test logpdf Jacobian using DerivativeChecker."""

        def fun(sample: Array) -> Array:
            # sample is (1, 1), logpdf returns (1, 1)
            logpdf = self._dist.logpdf(sample)
            return logpdf.T  # Return (nsamples, nqoi) = (1, 1)

        def jacobian(sample: Array) -> Array:
            # sample is (1, 1), logpdf_jacobian returns (1, 1)
            return self._dist.logpdf_jacobian(sample)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=1,
            fun=fun,
            jacobian=jacobian,
            bkd=self._bkd,
        )

        checker = DerivativeChecker(function_obj)
        # Use a sample away from 0 to avoid numerical issues
        sample = self._bkd.asarray([[1.5]])
        fd_eps = self._bkd.asarray([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
        errors = checker.check_derivatives(sample, fd_eps=fd_eps)
        self.assertLessEqual(float(checker.error_ratio(errors[0])), 1e-5)

    def test_equality(self) -> None:
        """Test equality comparison."""
        dist2 = GammaMarginal(self._shape, self._scale, self._bkd)
        dist3 = GammaMarginal(1.0, 1.0, self._bkd)
        self.assertEqual(self._dist, dist2)
        self.assertNotEqual(self._dist, dist3)

    def test_invalid_parameters_raise(self) -> None:
        """Test invalid parameters raise ValueError."""
        with self.assertRaises(ValueError):
            GammaMarginal(0.0, 1.0, self._bkd)  # shape <= 0
        with self.assertRaises(ValueError):
            GammaMarginal(-1.0, 1.0, self._bkd)  # shape < 0
        with self.assertRaises(ValueError):
            GammaMarginal(1.0, 0.0, self._bkd)  # scale <= 0
        with self.assertRaises(ValueError):
            GammaMarginal(1.0, -1.0, self._bkd)  # scale < 0

    def test_repr(self) -> None:
        """Test string representation."""
        repr_str = repr(self._dist)
        self.assertIn("GammaMarginal", repr_str)
        self.assertIn(str(self._shape), repr_str)
        self.assertIn(str(self._scale), repr_str)

    def test_exponential_distribution(self) -> None:
        """Test special case: shape=1 gives exponential distribution."""
        exp_dist = GammaMarginal(shape=1.0, scale=2.0, bkd=self._bkd)
        scipy_exp = stats.expon(scale=2.0)

        samples = self._bkd.asarray([[0.5, 1.0, 2.0]])
        pdf_vals = exp_dist(samples)
        expected = self._bkd.asarray([scipy_exp.pdf([0.5, 1.0, 2.0])])
        self.assertTrue(self._bkd.allclose(pdf_vals, expected, rtol=1e-10))

    def test_ppf_alias(self) -> None:
        """Test ppf is alias for invcdf."""
        probs = self._bkd.asarray([[0.25, 0.5, 0.75]])
        self.assertTrue(
            self._bkd.allclose(self._dist.ppf(probs), self._dist.invcdf(probs), rtol=1e-10)
        )


class TestGammaMarginalNumpy(TestGammaMarginal[NDArray[Any]]):
    """NumPy backend tests for GammaMarginal."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGammaMarginalTorch(TestGammaMarginal[torch.Tensor]):
    """PyTorch backend tests for GammaMarginal."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGammaMarginalAutograd(unittest.TestCase):
    """Test autograd compatibility for GammaMarginal."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        self._shape = 2.0
        self._scale = 1.5
        self._dist = GammaMarginal(self._shape, self._scale, self._bkd)

    def test_logpdf_gradient_wrt_samples(self) -> None:
        """Test autograd gradient of logpdf w.r.t. sample values."""
        samples = torch.tensor([[0.5, 1.0, 2.0]], requires_grad=True)
        logpdf = self._dist.logpdf(samples)

        # Sum to get scalar for backward
        loss = logpdf.sum()
        loss.backward()

        # Analytical gradient: d/dx log f(x) = (k-1)/x - rate
        rate = 1.0 / self._scale
        expected_grad = (self._shape - 1.0) / samples.detach() - rate

        self.assertTrue(
            torch.allclose(samples.grad, expected_grad, rtol=1e-6)
        )

    def test_pdf_gradient_wrt_samples(self) -> None:
        """Test autograd gradient of pdf w.r.t. sample values."""
        samples = torch.tensor([[0.5, 1.0, 2.0]], requires_grad=True)
        pdf = self._dist(samples)

        loss = pdf.sum()
        loss.backward()

        # Analytical gradient: d/dx pdf = pdf * d/dx log(pdf)
        rate = 1.0 / self._scale
        pdf_vals = self._dist(samples.detach())
        logpdf_grad = (self._shape - 1.0) / samples.detach() - rate
        expected_grad = pdf_vals * logpdf_grad

        self.assertTrue(
            torch.allclose(samples.grad, expected_grad, rtol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
