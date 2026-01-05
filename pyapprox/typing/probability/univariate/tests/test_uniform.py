"""
Tests for UniformMarginal distribution.
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
from pyapprox.typing.probability.univariate import UniformMarginal
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)


class TestUniformMarginal(Generic[Array], unittest.TestCase):
    """Tests for UniformMarginal."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._lower = -1.0
        self._upper = 2.0
        self._dist = UniformMarginal(self._lower, self._upper, self._bkd)
        self._scipy_dist = stats.uniform(loc=self._lower, scale=self._upper - self._lower)

    def test_nvars(self) -> None:
        """Test nvars returns 1."""
        self.assertEqual(self._dist.nvars(), 1)

    def test_bounds_accessors(self) -> None:
        """Test lower and upper accessors."""
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([self._dist.lower]),
                self._bkd.asarray([self._lower]),
                atol=1e-10,
            )
        )
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([self._dist.upper]),
                self._bkd.asarray([self._upper]),
                atol=1e-10,
            )
        )

    def test_pdf_constant(self) -> None:
        """Test PDF is constant within bounds."""
        samples = self._bkd.asarray([0.0, 0.5, 1.0, 1.5])
        pdf_vals = self._dist(samples)
        expected_val = 1.0 / (self._upper - self._lower)
        expected = self._bkd.ones_like(samples) * expected_val
        self.assertTrue(self._bkd.allclose(pdf_vals, expected, rtol=1e-10))

    def test_pdf_matches_scipy(self) -> None:
        """Test PDF matches scipy."""
        samples = self._bkd.asarray([0.0, 0.5, 1.0])
        pdf_vals = self._dist(samples)
        expected = self._bkd.asarray(self._scipy_dist.pdf([0.0, 0.5, 1.0]))
        self.assertTrue(self._bkd.allclose(pdf_vals, expected, rtol=1e-10))

    def test_logpdf_matches_scipy(self) -> None:
        """Test logpdf matches scipy."""
        samples = self._bkd.asarray([0.0, 0.5, 1.0])
        logpdf_vals = self._dist.logpdf(samples)
        expected = self._bkd.asarray(self._scipy_dist.logpdf([0.0, 0.5, 1.0]))
        self.assertTrue(self._bkd.allclose(logpdf_vals, expected, rtol=1e-10))

    def test_cdf_matches_scipy(self) -> None:
        """Test CDF matches scipy."""
        samples = self._bkd.asarray([-0.5, 0.0, 0.5, 1.0, 1.5])
        cdf_vals = self._dist.cdf(samples)
        expected = self._bkd.asarray(self._scipy_dist.cdf([-0.5, 0.0, 0.5, 1.0, 1.5]))
        self.assertTrue(self._bkd.allclose(cdf_vals, expected, rtol=1e-10))

    def test_invcdf_matches_scipy(self) -> None:
        """Test invcdf matches scipy ppf."""
        probs = self._bkd.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
        invcdf_vals = self._dist.invcdf(probs)
        expected = self._bkd.asarray(self._scipy_dist.ppf([0.0, 0.25, 0.5, 0.75, 1.0]))
        self.assertTrue(self._bkd.allclose(invcdf_vals, expected, rtol=1e-10))

    def test_cdf_invcdf_inverse(self) -> None:
        """Test cdf(invcdf(p)) = p."""
        probs = self._bkd.asarray([0.1, 0.5, 0.9])
        recovered = self._dist.cdf(self._dist.invcdf(probs))
        self.assertTrue(self._bkd.allclose(recovered, probs, rtol=1e-10))

    def test_invcdf_cdf_inverse(self) -> None:
        """Test invcdf(cdf(x)) = x."""
        samples = self._bkd.asarray([0.0, 0.5, 1.0])
        recovered = self._dist.invcdf(self._dist.cdf(samples))
        self.assertTrue(self._bkd.allclose(recovered, samples, rtol=1e-10))

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        samples = self._dist.rvs(100)
        self.assertEqual(samples.shape, (1, 100))

    def test_rvs_in_bounds(self) -> None:
        """Test rvs produces samples in [lower, upper]."""
        samples = self._dist.rvs(1000)
        samples_flat = self._bkd.flatten(samples)
        self.assertTrue(self._bkd.all_bool(samples_flat >= self._lower))
        self.assertTrue(self._bkd.all_bool(samples_flat <= self._upper))

    def test_logpdf_exp_equals_pdf(self) -> None:
        """Test exp(logpdf) = pdf."""
        samples = self._bkd.asarray([0.0, 0.5, 1.0])
        pdf_vals = self._dist(samples)
        logpdf_vals = self._dist.logpdf(samples)
        self.assertTrue(
            self._bkd.allclose(self._bkd.exp(logpdf_vals), pdf_vals, rtol=1e-10)
        )

    def test_mean_variance(self) -> None:
        """Test mean and variance match analytical formulas."""
        expected_mean = (self._lower + self._upper) / 2.0
        expected_var = (self._upper - self._lower) ** 2 / 12.0
        expected_std = np.sqrt(expected_var)

        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([self._dist.mean_value()]),
                self._bkd.asarray([expected_mean]),
                atol=1e-10,
            )
        )
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([self._dist.variance()]),
                self._bkd.asarray([expected_var]),
                atol=1e-10,
            )
        )
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([self._dist.std()]),
                self._bkd.asarray([expected_std]),
                atol=1e-10,
            )
        )

    def test_mean_variance_match_scipy(self) -> None:
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
        """Test Uniform is bounded."""
        self.assertTrue(self._dist.is_bounded())

    def test_bounds(self) -> None:
        """Test bounds are [lower, upper]."""
        lower, upper = self._dist.bounds()
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([lower]),
                self._bkd.asarray([self._lower]),
                atol=1e-10,
            )
        )
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([upper]),
                self._bkd.asarray([self._upper]),
                atol=1e-10,
            )
        )

    def test_interval(self) -> None:
        """Test interval contains correct probability."""
        alpha = 0.95
        interval = self._dist.interval(alpha)
        self.assertEqual(len(interval), 2)
        lower, upper = float(interval[0]), float(interval[1])
        self.assertGreater(lower, self._lower)
        self.assertLess(upper, self._upper)
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
        """Test invcdf Jacobian = width (constant for uniform)."""
        probs = self._bkd.asarray([0.25, 0.5, 0.75])
        jacobian = self._dist.invcdf_jacobian(probs)
        expected_width = self._upper - self._lower
        expected = self._bkd.ones_like(probs) * expected_width
        self.assertTrue(self._bkd.allclose(jacobian, expected, rtol=1e-10))

    def test_logpdf_jacobian_zero(self) -> None:
        """Test logpdf Jacobian is zero (constant PDF)."""
        samples = self._bkd.asarray([0.0, 0.5, 1.0])
        jacobian = self._dist.logpdf_jacobian(samples)
        expected = self._bkd.zeros((1, 3))
        self.assertTrue(self._bkd.allclose(jacobian, expected, atol=1e-10))

    def test_pdf_jacobian_zero(self) -> None:
        """Test pdf Jacobian is zero (constant PDF)."""
        samples = self._bkd.asarray([0.0, 0.5, 1.0])
        jacobian = self._dist.pdf_jacobian(samples)
        expected = self._bkd.zeros((1, 3))
        self.assertTrue(self._bkd.allclose(jacobian, expected, atol=1e-10))

    def test_equality(self) -> None:
        """Test equality comparison."""
        dist2 = UniformMarginal(self._lower, self._upper, self._bkd)
        dist3 = UniformMarginal(0.0, 1.0, self._bkd)
        self.assertEqual(self._dist, dist2)
        self.assertNotEqual(self._dist, dist3)

    def test_invalid_parameters_raise(self) -> None:
        """Test invalid parameters raise ValueError."""
        with self.assertRaises(ValueError):
            UniformMarginal(1.0, 1.0, self._bkd)  # lower == upper
        with self.assertRaises(ValueError):
            UniformMarginal(2.0, 1.0, self._bkd)  # lower > upper

    def test_repr(self) -> None:
        """Test string representation."""
        repr_str = repr(self._dist)
        self.assertIn("UniformMarginal", repr_str)
        self.assertIn(str(self._lower), repr_str)
        self.assertIn(str(self._upper), repr_str)

    def test_standard_uniform(self) -> None:
        """Test standard uniform [0, 1]."""
        dist = UniformMarginal(0.0, 1.0, self._bkd)
        samples = self._bkd.asarray([0.0, 0.25, 0.5, 0.75, 1.0])

        # PDF should be 1.0 everywhere
        pdf_vals = dist(samples)
        self.assertTrue(
            self._bkd.allclose(pdf_vals, self._bkd.ones_like(samples), rtol=1e-10)
        )

        # CDF should equal the sample values
        cdf_vals = dist.cdf(samples)
        self.assertTrue(self._bkd.allclose(cdf_vals, samples, rtol=1e-10))

        # invcdf should equal the probability values
        invcdf_vals = dist.invcdf(samples)
        self.assertTrue(self._bkd.allclose(invcdf_vals, samples, rtol=1e-10))

    def test_ppf_alias(self) -> None:
        """Test ppf is alias for invcdf."""
        probs = self._bkd.asarray([0.25, 0.5, 0.75])
        self.assertTrue(
            self._bkd.allclose(self._dist.ppf(probs), self._dist.invcdf(probs), rtol=1e-10)
        )


class TestUniformMarginalNumpy(TestUniformMarginal[NDArray[Any]]):
    """NumPy backend tests for UniformMarginal."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestUniformMarginalTorch(TestUniformMarginal[torch.Tensor]):
    """PyTorch backend tests for UniformMarginal."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
