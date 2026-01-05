"""
Tests for BetaMarginal distribution.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
from scipy import stats
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests
from pyapprox.typing.probability.univariate import BetaMarginal
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)


class TestBetaMarginal(Generic[Array], unittest.TestCase):
    """Tests for BetaMarginal."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._alpha = 2.0
        self._beta = 5.0
        self._dist = BetaMarginal(self._alpha, self._beta, self._bkd)
        self._scipy_dist = stats.beta(self._alpha, self._beta)

    def test_nvars(self) -> None:
        """Test nvars returns 1."""
        self.assertEqual(self._dist.nvars(), 1)

    def test_shape_parameters(self) -> None:
        """Test shape parameter accessors."""
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([self._dist.alpha]),
                self._bkd.asarray([self._alpha]),
                atol=1e-10,
            )
        )
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([self._dist.beta]),
                self._bkd.asarray([self._beta]),
                atol=1e-10,
            )
        )

    def test_pdf_matches_scipy(self) -> None:
        """Test PDF matches scipy."""
        samples = self._bkd.asarray([0.1, 0.3, 0.5, 0.7])
        pdf_vals = self._dist(samples)
        expected = self._bkd.asarray(self._scipy_dist.pdf([0.1, 0.3, 0.5, 0.7]))
        self.assertTrue(self._bkd.allclose(pdf_vals, expected, rtol=1e-10))

    def test_logpdf_matches_scipy(self) -> None:
        """Test logpdf matches scipy."""
        samples = self._bkd.asarray([0.1, 0.3, 0.5, 0.7])
        logpdf_vals = self._dist.logpdf(samples)
        expected = self._bkd.asarray(self._scipy_dist.logpdf([0.1, 0.3, 0.5, 0.7]))
        self.assertTrue(self._bkd.allclose(logpdf_vals, expected, rtol=1e-10))

    def test_cdf_matches_scipy(self) -> None:
        """Test CDF matches scipy."""
        samples = self._bkd.asarray([0.1, 0.3, 0.5, 0.7])
        cdf_vals = self._dist.cdf(samples)
        expected = self._bkd.asarray(self._scipy_dist.cdf([0.1, 0.3, 0.5, 0.7]))
        self.assertTrue(self._bkd.allclose(cdf_vals, expected, rtol=1e-10))

    def test_invcdf_matches_scipy(self) -> None:
        """Test invcdf matches scipy ppf."""
        probs = self._bkd.asarray([0.1, 0.5, 0.9])
        invcdf_vals = self._dist.invcdf(probs)
        expected = self._bkd.asarray(self._scipy_dist.ppf([0.1, 0.5, 0.9]))
        self.assertTrue(self._bkd.allclose(invcdf_vals, expected, rtol=1e-10))

    def test_cdf_invcdf_inverse(self) -> None:
        """Test cdf(invcdf(p)) = p."""
        probs = self._bkd.asarray([0.1, 0.5, 0.9])
        recovered = self._dist.cdf(self._dist.invcdf(probs))
        self.assertTrue(self._bkd.allclose(recovered, probs, rtol=1e-6))

    def test_invcdf_cdf_inverse(self) -> None:
        """Test invcdf(cdf(x)) = x."""
        samples = self._bkd.asarray([0.2, 0.4, 0.6])
        recovered = self._dist.invcdf(self._dist.cdf(samples))
        self.assertTrue(self._bkd.allclose(recovered, samples, rtol=1e-6))

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        samples = self._dist.rvs(100)
        self.assertEqual(samples.shape, (1, 100))

    def test_rvs_in_bounds(self) -> None:
        """Test rvs produces samples in [0, 1]."""
        samples = self._dist.rvs(1000)
        samples_flat = self._bkd.flatten(samples)
        self.assertTrue(self._bkd.all_bool(samples_flat >= 0.0))
        self.assertTrue(self._bkd.all_bool(samples_flat <= 1.0))

    def test_logpdf_exp_equals_pdf(self) -> None:
        """Test exp(logpdf) = pdf."""
        samples = self._bkd.asarray([0.2, 0.4, 0.6])
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
        """Test Beta is bounded."""
        self.assertTrue(self._dist.is_bounded())

    def test_bounds(self) -> None:
        """Test bounds are [0, 1]."""
        lower, upper = self._dist.bounds()
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([lower]),
                self._bkd.asarray([0.0]),
                atol=1e-10,
            )
        )
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([upper]),
                self._bkd.asarray([1.0]),
                atol=1e-10,
            )
        )

    def test_interval(self) -> None:
        """Test interval contains correct probability."""
        alpha = 0.95
        interval = self._dist.interval(alpha)
        self.assertEqual(len(interval), 2)
        lower, upper = float(interval[0]), float(interval[1])
        self.assertLess(lower, 0.5)
        self.assertGreater(upper, 0.5)
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
        probs = self._bkd.asarray([0.25, 0.5, 0.75])
        jacobian = self._dist.invcdf_jacobian(probs)
        quantiles = self._dist.invcdf(probs)
        pdf_at_quantiles = self._dist(quantiles)
        self.assertTrue(
            self._bkd.allclose(jacobian, 1.0 / pdf_at_quantiles, rtol=1e-6)
        )

    def test_logpdf_jacobian_derivative_checker(self) -> None:
        """Test logpdf Jacobian using DerivativeChecker."""

        def fun(sample: Array) -> Array:
            x = self._bkd.flatten(sample)
            logpdf = self._dist.logpdf(x)
            return logpdf[:, None]

        def jacobian(sample: Array) -> Array:
            x = self._bkd.flatten(sample)
            return self._dist.logpdf_jacobian(x)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=1,
            fun=fun,
            jacobian=jacobian,
            bkd=self._bkd,
        )

        checker = DerivativeChecker(function_obj)
        sample = self._bkd.asarray([[0.3]])
        # Use smaller fd_eps to stay within [0, 1] bounds
        fd_eps = self._bkd.asarray([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
        errors = checker.check_derivatives(sample, fd_eps=fd_eps)
        self.assertLessEqual(float(checker.error_ratio(errors[0])), 1e-6)

    def test_equality(self) -> None:
        """Test equality comparison."""
        dist2 = BetaMarginal(self._alpha, self._beta, self._bkd)
        dist3 = BetaMarginal(1.0, 1.0, self._bkd)
        self.assertEqual(self._dist, dist2)
        self.assertNotEqual(self._dist, dist3)

    def test_invalid_parameters_raise(self) -> None:
        """Test invalid parameters raise ValueError."""
        with self.assertRaises(ValueError):
            BetaMarginal(0.0, 1.0, self._bkd)
        with self.assertRaises(ValueError):
            BetaMarginal(1.0, -1.0, self._bkd)


class TestBetaMarginalNumpy(TestBetaMarginal[NDArray[Any]]):
    """NumPy backend tests for BetaMarginal."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBetaMarginalTorch(TestBetaMarginal[torch.Tensor]):
    """PyTorch backend tests for BetaMarginal."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
