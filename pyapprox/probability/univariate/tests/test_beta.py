"""
Tests for BetaMarginal distribution.
"""

import unittest
from typing import Any, Generic, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats
import torch
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests
from pyapprox.probability.univariate import BetaMarginal
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.surrogates.affine.univariate.globalpoly import (
    LegendrePolynomial1D,
    GaussQuadratureRule,
)

# Test configurations: (name, alpha, beta, lb, ub)
BETA_CONFIGS = [
    ("canonical_2_5", 2.0, 5.0, 0.0, 1.0),
    ("canonical_5_2", 5.0, 2.0, 0.0, 1.0),
    ("symmetric_3_3", 3.0, 3.0, 0.0, 1.0),
    ("scaled_2_5_02", 2.0, 5.0, 0.0, 2.0),
    ("scaled_5_2_25", 5.0, 2.0, 2.0, 5.0),
    ("negative_3_3", 3.0, 3.0, -1.0, 3.0),
    ("wide_2_2", 2.0, 2.0, -10.0, 10.0),
]


class GaussLegendreQuadratureScaled(Generic[Array]):
    """Gauss-Legendre quadrature on [lb, ub] with Lebesgue measure for tests."""

    def __init__(
        self, bkd: Backend[Array], lb: float = 0.0, ub: float = 1.0
    ) -> None:
        self._bkd = bkd
        self._lb = lb
        self._ub = ub
        self._scale = ub - lb
        self._legendre = LegendrePolynomial1D(bkd)
        self._quad_rule = GaussQuadratureRule(self._legendre, store=True)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __call__(self, npoints: int) -> Tuple[Array, Array]:
        points_11, weights_11 = self._quad_rule(npoints)
        # Map from [-1, 1] to [lb, ub]
        # GaussQuadratureRule integrates uniform probability measure on [-1,1]
        # (weights sum to 1). For [lb, ub] we shift/scale points; weights stay same.
        points_scaled = self._lb + (points_11 + 1.0) / 2.0 * self._scale
        return points_scaled, weights_11


class TestBetaMarginal(Generic[Array], unittest.TestCase):
    """Tests for BetaMarginal."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._alpha = 2.0
        self._beta = 5.0
        # Create quadrature rule on [0, 1] for CDF/invcdf tests
        self._quad_rule = GaussLegendreQuadratureScaled(self._bkd)
        self._dist = BetaMarginal(
            self._alpha, self._beta, self._bkd, quadrature_rule=self._quad_rule
        )
        self._scipy_dist = stats.beta(self._alpha, self._beta)

    def test_nvars(self) -> None:
        """Test nvars returns 1."""
        self.assertEqual(self._dist.nvars(), 1)

    def test_shape_parameters(self) -> None:
        """Test shape parameter accessors."""
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([self._dist.alpha()]),
                self._bkd.asarray([self._alpha]),
                atol=1e-10,
            )
        )
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([self._dist.beta()]),
                self._bkd.asarray([self._beta]),
                atol=1e-10,
            )
        )

    def test_pdf_matches_scipy(self) -> None:
        """Test PDF matches scipy."""
        samples = self._bkd.asarray([[0.1, 0.3, 0.5, 0.7]])
        pdf_vals = self._dist(samples)
        expected = self._bkd.asarray([self._scipy_dist.pdf([0.1, 0.3, 0.5, 0.7])])
        self.assertTrue(self._bkd.allclose(pdf_vals, expected, rtol=1e-10))

    def test_logpdf_matches_scipy(self) -> None:
        """Test logpdf matches scipy."""
        samples = self._bkd.asarray([[0.1, 0.3, 0.5, 0.7]])
        logpdf_vals = self._dist.logpdf(samples)
        expected = self._bkd.asarray([self._scipy_dist.logpdf([0.1, 0.3, 0.5, 0.7])])
        self.assertTrue(self._bkd.allclose(logpdf_vals, expected, rtol=1e-10))

    def test_cdf_matches_scipy(self) -> None:
        """Test CDF matches scipy."""
        samples = self._bkd.asarray([[0.1, 0.3, 0.5, 0.7]])
        cdf_vals = self._dist.cdf(samples)
        expected = self._bkd.asarray([self._scipy_dist.cdf([0.1, 0.3, 0.5, 0.7])])
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
        samples = self._bkd.asarray([[0.2, 0.4, 0.6]])
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
        samples = self._bkd.asarray([[0.2, 0.4, 0.6]])
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
        # interval returns shape (1, 2) with [lower, upper] bounds
        self.assertEqual(interval.shape, (1, 2))
        lower, upper = float(interval[0, 0]), float(interval[0, 1])
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


class TestBetaMarginalParametrized(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Parametrized tests for BetaMarginal with various shapes and bounds."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def _create_dist(
        self, alpha: float, beta: float, lb: float, ub: float
    ) -> Tuple[BetaMarginal, Any]:
        """Create BetaMarginal and scipy reference distribution."""
        bkd = self.bkd()
        scale = ub - lb
        # BetaMarginal expects a quadrature rule on [0, 1] for internal CDF
        # computation. The class handles the transformation to [lb, ub] internally.
        quad_rule = GaussLegendreQuadratureScaled(bkd, 0.0, 1.0)
        dist = BetaMarginal(
            alpha, beta, bkd, lb=lb, ub=ub, quadrature_rule=quad_rule
        )
        scipy_dist = stats.beta(alpha, beta, loc=lb, scale=scale)
        return dist, scipy_dist

    @parametrize(
        "name,alpha,beta,lb,ub",
        BETA_CONFIGS,
    )
    def test_bounds_accessors(
        self, name: str, alpha: float, beta: float, lb: float, ub: float
    ) -> None:
        """Test bounds accessor methods for various configurations."""
        bkd = self.bkd()
        dist, _ = self._create_dist(alpha, beta, lb, ub)
        bkd.assert_allclose(
            bkd.asarray([dist.lower()]), bkd.asarray([lb]), atol=1e-10
        )
        bkd.assert_allclose(
            bkd.asarray([dist.upper()]), bkd.asarray([ub]), atol=1e-10
        )
        lower, upper = dist.bounds()
        bkd.assert_allclose(
            bkd.asarray([lower, upper]), bkd.asarray([lb, ub]), atol=1e-10
        )

    @parametrize(
        "name,alpha,beta,lb,ub",
        BETA_CONFIGS,
    )
    def test_pdf_matches_scipy(
        self, name: str, alpha: float, beta: float, lb: float, ub: float
    ) -> None:
        """Test PDF matches scipy for various configurations."""
        bkd = self.bkd()
        dist, scipy_dist = self._create_dist(alpha, beta, lb, ub)
        # Generate sample points within [lb, ub]
        scale = ub - lb
        test_points = [lb + 0.1 * scale, lb + 0.3 * scale, lb + 0.5 * scale,
                       lb + 0.7 * scale, lb + 0.9 * scale]
        samples = bkd.asarray([test_points])
        pdf_vals = dist(samples)
        expected = bkd.asarray([scipy_dist.pdf(test_points)])
        bkd.assert_allclose(pdf_vals, expected, rtol=1e-10)

    @parametrize(
        "name,alpha,beta,lb,ub",
        BETA_CONFIGS,
    )
    def test_logpdf_matches_scipy(
        self, name: str, alpha: float, beta: float, lb: float, ub: float
    ) -> None:
        """Test logpdf matches scipy for various configurations."""
        bkd = self.bkd()
        dist, scipy_dist = self._create_dist(alpha, beta, lb, ub)
        scale = ub - lb
        test_points = [lb + 0.1 * scale, lb + 0.3 * scale, lb + 0.5 * scale,
                       lb + 0.7 * scale, lb + 0.9 * scale]
        samples = bkd.asarray([test_points])
        logpdf_vals = dist.logpdf(samples)
        expected = bkd.asarray([scipy_dist.logpdf(test_points)])
        bkd.assert_allclose(logpdf_vals, expected, rtol=1e-10)

    @parametrize(
        "name,alpha,beta,lb,ub",
        BETA_CONFIGS,
    )
    def test_cdf_matches_scipy(
        self, name: str, alpha: float, beta: float, lb: float, ub: float
    ) -> None:
        """Test CDF matches scipy for various configurations."""
        bkd = self.bkd()
        dist, scipy_dist = self._create_dist(alpha, beta, lb, ub)
        scale = ub - lb
        test_points = [lb + 0.1 * scale, lb + 0.3 * scale, lb + 0.5 * scale,
                       lb + 0.7 * scale, lb + 0.9 * scale]
        samples = bkd.asarray([test_points])
        cdf_vals = dist.cdf(samples)
        expected = bkd.asarray([scipy_dist.cdf(test_points)])
        bkd.assert_allclose(cdf_vals, expected, rtol=1e-10)

    @parametrize(
        "name,alpha,beta,lb,ub",
        BETA_CONFIGS,
    )
    def test_invcdf_matches_scipy(
        self, name: str, alpha: float, beta: float, lb: float, ub: float
    ) -> None:
        """Test invcdf matches scipy ppf for various configurations."""
        bkd = self.bkd()
        dist, scipy_dist = self._create_dist(alpha, beta, lb, ub)
        probs = bkd.asarray([[0.1, 0.5, 0.9]])
        invcdf_vals = dist.invcdf(probs)
        expected = bkd.asarray([scipy_dist.ppf([0.1, 0.5, 0.9])])
        # Use atol for values near zero (e.g., symmetric distributions)
        bkd.assert_allclose(invcdf_vals, expected, rtol=1e-10, atol=1e-12)

    @parametrize(
        "name,alpha,beta,lb,ub",
        BETA_CONFIGS,
    )
    def test_cdf_invcdf_inverse(
        self, name: str, alpha: float, beta: float, lb: float, ub: float
    ) -> None:
        """Test cdf(invcdf(p)) = p for various configurations."""
        bkd = self.bkd()
        dist, _ = self._create_dist(alpha, beta, lb, ub)
        probs = bkd.asarray([[0.1, 0.5, 0.9]])
        recovered = dist.cdf(dist.invcdf(probs))
        bkd.assert_allclose(recovered, probs, rtol=1e-6)

    @parametrize(
        "name,alpha,beta,lb,ub",
        BETA_CONFIGS,
    )
    def test_mean_value_matches_scipy(
        self, name: str, alpha: float, beta: float, lb: float, ub: float
    ) -> None:
        """Test mean_value matches scipy for various configurations."""
        bkd = self.bkd()
        dist, scipy_dist = self._create_dist(alpha, beta, lb, ub)
        bkd.assert_allclose(
            bkd.asarray([dist.mean_value()]),
            bkd.asarray([scipy_dist.mean()]),
            atol=1e-10,
        )

    @parametrize(
        "name,alpha,beta,lb,ub",
        BETA_CONFIGS,
    )
    def test_variance_matches_scipy(
        self, name: str, alpha: float, beta: float, lb: float, ub: float
    ) -> None:
        """Test variance matches scipy for various configurations."""
        bkd = self.bkd()
        dist, scipy_dist = self._create_dist(alpha, beta, lb, ub)
        bkd.assert_allclose(
            bkd.asarray([dist.variance()]),
            bkd.asarray([scipy_dist.var()]),
            atol=1e-10,
        )

    @parametrize(
        "name,alpha,beta,lb,ub",
        BETA_CONFIGS,
    )
    def test_rvs_in_bounds(
        self, name: str, alpha: float, beta: float, lb: float, ub: float
    ) -> None:
        """Test rvs produces samples in [lb, ub]."""
        np.random.seed(42)
        bkd = self.bkd()
        dist, _ = self._create_dist(alpha, beta, lb, ub)
        samples = dist.rvs(1000)
        samples_flat = bkd.flatten(samples)
        self.assertTrue(bkd.all_bool(samples_flat >= lb))
        self.assertTrue(bkd.all_bool(samples_flat <= ub))

    @parametrize(
        "name,alpha,beta,lb,ub",
        BETA_CONFIGS,
    )
    def test_logpdf_jacobian(
        self, name: str, alpha: float, beta: float, lb: float, ub: float
    ) -> None:
        """Test logpdf Jacobian using DerivativeChecker."""
        bkd = self.bkd()
        dist, _ = self._create_dist(alpha, beta, lb, ub)

        def fun(sample: Array) -> Array:
            return dist.logpdf(sample).T

        def jacobian(sample: Array) -> Array:
            return dist.logpdf_jacobian(sample)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=1, fun=fun, jacobian=jacobian, bkd=bkd
        )

        checker = DerivativeChecker(function_obj)
        # Sample in the middle of the domain
        scale = ub - lb
        # Sample at 50% of domain to stay away from boundaries
        sample = bkd.asarray([[lb + 0.5 * scale]])
        # Scale fd_eps relative to the domain size for wide bounds
        base_eps = bkd.asarray([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
        fd_eps = base_eps * max(1.0, scale / 10.0)
        errors = checker.check_derivatives(sample, fd_eps=fd_eps)
        # Use relaxed tolerance for numerical derivatives per CLAUDE.md
        self.assertLessEqual(float(checker.error_ratio(errors[0])), 1e-5)


class TestBetaMarginalParametrizedNumpy(
    TestBetaMarginalParametrized[NDArray[Any]]
):
    """NumPy backend parametrized tests for BetaMarginal."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBetaMarginalParametrizedTorch(
    TestBetaMarginalParametrized[torch.Tensor]
):
    """PyTorch backend parametrized tests for BetaMarginal."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestBetaMarginalBoundsValidation(unittest.TestCase):
    """Tests for bounds validation in BetaMarginal."""

    def test_invalid_bounds_lb_equals_ub(self) -> None:
        """Test that lb == ub raises ValueError."""
        bkd = NumpyBkd()
        with self.assertRaises(ValueError):
            BetaMarginal(2.0, 5.0, bkd, lb=1.0, ub=1.0)

    def test_invalid_bounds_lb_greater_ub(self) -> None:
        """Test that lb > ub raises ValueError."""
        bkd = NumpyBkd()
        with self.assertRaises(ValueError):
            BetaMarginal(2.0, 5.0, bkd, lb=2.0, ub=1.0)

    def test_equality_includes_bounds(self) -> None:
        """Test that equality comparison includes bounds."""
        bkd = NumpyBkd()
        dist1 = BetaMarginal(2.0, 5.0, bkd, lb=0.0, ub=2.0)
        dist2 = BetaMarginal(2.0, 5.0, bkd, lb=0.0, ub=2.0)
        dist3 = BetaMarginal(2.0, 5.0, bkd, lb=0.0, ub=1.0)
        self.assertEqual(dist1, dist2)
        self.assertNotEqual(dist1, dist3)


if __name__ == "__main__":
    unittest.main()
