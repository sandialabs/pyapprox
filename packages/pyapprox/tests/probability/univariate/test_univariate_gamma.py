"""
Tests for GammaMarginal distribution.
"""

from typing import Callable, Tuple

import pytest
from scipy import stats

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.probability.univariate import GammaMarginal
from pyapprox.surrogates.affine.univariate.globalpoly import (
    GaussQuadratureRule,
    LegendrePolynomial1D,
)
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array


class GaussLegendreQuadrature01:
    """Gauss-Legendre quadrature on [0, 1] with Lebesgue measure for tests."""

    def __init__(self, bkd) -> None:
        self._bkd = bkd
        self._legendre = LegendrePolynomial1D(bkd)
        self._quad_rule = GaussQuadratureRule(self._legendre, store=True)

    def bkd(self):
        return self._bkd

    def __call__(self, npoints: int) -> Tuple:
        points_11, weights_11 = self._quad_rule(npoints)
        # Map from [-1, 1] to [0, 1]
        # GaussQuadratureRule integrates uniform probability measure on [-1,1]
        # (weights sum to 1). For [0,1] we just shift points; weights stay same.
        points_01 = (points_11 + 1.0) / 2.0
        return points_01, weights_11


class TestGammaMarginal:
    """Tests for GammaMarginal."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd) -> None:
        self._bkd = bkd
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
        assert self._dist.nvars() == 1

    def test_shape_parameters(self) -> None:
        """Test shape and scale parameter accessors."""
        assert self._bkd.allclose(
            self._bkd.asarray([self._dist.shape()]),
            self._bkd.asarray([self._shape]),
            atol=1e-10,
        )
        assert self._bkd.allclose(
            self._bkd.asarray([self._dist.scale()]),
            self._bkd.asarray([self._scale]),
            atol=1e-10,
        )
        assert self._bkd.allclose(
            self._bkd.asarray([self._dist.rate()]),
            self._bkd.asarray([1.0 / self._scale]),
            atol=1e-10,
        )

    def test_pdf_matches_scipy(self) -> None:
        """Test PDF matches scipy."""
        samples = self._bkd.asarray([[0.5, 1.0, 2.0, 3.0]])
        pdf_vals = self._dist(samples)
        expected = self._bkd.asarray([self._scipy_dist.pdf([0.5, 1.0, 2.0, 3.0])])
        assert self._bkd.allclose(pdf_vals, expected, rtol=1e-10)

    def test_logpdf_matches_scipy(self) -> None:
        """Test logpdf matches scipy."""
        samples = self._bkd.asarray([[0.5, 1.0, 2.0, 3.0]])
        logpdf_vals = self._dist.logpdf(samples)
        expected = self._bkd.asarray([self._scipy_dist.logpdf([0.5, 1.0, 2.0, 3.0])])
        assert self._bkd.allclose(logpdf_vals, expected, rtol=1e-10)

    def test_cdf_matches_scipy(self) -> None:
        """Test CDF matches scipy."""
        samples = self._bkd.asarray([[0.5, 1.0, 2.0, 3.0]])
        cdf_vals = self._dist.cdf(samples)
        expected = self._bkd.asarray([self._scipy_dist.cdf([0.5, 1.0, 2.0, 3.0])])
        assert self._bkd.allclose(cdf_vals, expected, rtol=1e-10)

    def test_invcdf_matches_scipy(self) -> None:
        """Test invcdf matches scipy ppf."""
        probs = self._bkd.asarray([[0.1, 0.5, 0.9]])
        invcdf_vals = self._dist.invcdf(probs)
        expected = self._bkd.asarray([self._scipy_dist.ppf([0.1, 0.5, 0.9])])
        assert self._bkd.allclose(invcdf_vals, expected, rtol=1e-10)

    def test_cdf_invcdf_inverse(self) -> None:
        """Test cdf(invcdf(p)) = p."""
        probs = self._bkd.asarray([[0.1, 0.5, 0.9]])
        recovered = self._dist.cdf(self._dist.invcdf(probs))
        assert self._bkd.allclose(recovered, probs, rtol=1e-6)

    def test_invcdf_cdf_inverse(self) -> None:
        """Test invcdf(cdf(x)) = x."""
        samples = self._bkd.asarray([[0.5, 1.0, 2.0]])
        recovered = self._dist.invcdf(self._dist.cdf(samples))
        assert self._bkd.allclose(recovered, samples, rtol=1e-6)

    def test_rvs_shape(self) -> None:
        """Test rvs returns correct shape."""
        samples = self._dist.rvs(100)
        assert samples.shape == (1, 100)

    def test_rvs_positive(self) -> None:
        """Test rvs produces positive samples."""
        samples = self._dist.rvs(1000)
        samples_flat = self._bkd.flatten(samples)
        assert self._bkd.all_bool(samples_flat >= 0.0)

    def test_logpdf_exp_equals_pdf(self) -> None:
        """Test exp(logpdf) = pdf."""
        samples = self._bkd.asarray([[0.5, 1.0, 2.0]])
        pdf_vals = self._dist(samples)
        logpdf_vals = self._dist.logpdf(samples)
        assert self._bkd.allclose(self._bkd.exp(logpdf_vals), pdf_vals, rtol=1e-10)

    def test_mean_variance(self) -> None:
        """Test mean and variance match scipy."""
        assert self._bkd.allclose(
            self._bkd.asarray([self._dist.mean_value()]),
            self._bkd.asarray([self._scipy_dist.mean()]),
            atol=1e-10,
        )
        assert self._bkd.allclose(
            self._bkd.asarray([self._dist.variance()]),
            self._bkd.asarray([self._scipy_dist.var()]),
            atol=1e-10,
        )
        assert self._bkd.allclose(
            self._bkd.asarray([self._dist.std()]),
            self._bkd.asarray([self._scipy_dist.std()]),
            atol=1e-10,
        )

    def test_is_bounded(self) -> None:
        """Test Gamma is not bounded (unbounded on right)."""
        assert not self._dist.is_bounded()

    def test_bounds(self) -> None:
        """Test bounds are [0, inf]."""
        lower, upper = self._dist.bounds()
        assert self._bkd.allclose(
            self._bkd.asarray([lower]),
            self._bkd.asarray([0.0]),
            atol=1e-10,
        )
        assert upper == float("inf")

    def test_interval(self) -> None:
        """Test interval contains correct probability."""
        alpha = 0.95
        interval = self._dist.interval(alpha)
        # interval returns shape (1, 2) with [lower, upper] bounds
        assert interval.shape == (1, 2)
        lower, upper = float(interval[0, 0]), float(interval[0, 1])
        assert lower > 0.0
        assert lower < upper
        # Check probability content
        prob = self._scipy_dist.cdf(upper) - self._scipy_dist.cdf(lower)
        assert self._bkd.allclose(
            self._bkd.asarray([prob]),
            self._bkd.asarray([alpha]),
            atol=1e-6,
        )

    def test_invcdf_jacobian(self) -> None:
        """Test invcdf Jacobian = 1/pdf at quantile."""
        probs = self._bkd.asarray([[0.25, 0.5, 0.75]])
        jacobian = self._dist.invcdf_jacobian(probs)
        quantiles = self._dist.invcdf(probs)
        pdf_at_quantiles = self._dist(quantiles)
        assert self._bkd.allclose(jacobian, 1.0 / pdf_at_quantiles, rtol=1e-6)

    def test_logpdf_jacobian_derivative_checker(self) -> None:
        """Test logpdf Jacobian using DerivativeChecker."""

        def fun(sample):
            # sample is (1, 1), logpdf returns (1, 1)
            logpdf = self._dist.logpdf(sample)
            return logpdf.T  # Return (nsamples, nqoi) = (1, 1)

        def jacobian(sample):
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
        assert float(checker.error_ratio(errors[0])) <= 1e-5

    def test_equality(self) -> None:
        """Test equality comparison."""
        dist2 = GammaMarginal(self._shape, self._scale, self._bkd)
        dist3 = GammaMarginal(1.0, 1.0, self._bkd)
        assert self._dist == dist2
        assert self._dist != dist3

    def test_invalid_parameters_raise(self) -> None:
        """Test invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            GammaMarginal(0.0, 1.0, self._bkd)  # shape <= 0
        with pytest.raises(ValueError):
            GammaMarginal(-1.0, 1.0, self._bkd)  # shape < 0
        with pytest.raises(ValueError):
            GammaMarginal(1.0, 0.0, self._bkd)  # scale <= 0
        with pytest.raises(ValueError):
            GammaMarginal(1.0, -1.0, self._bkd)  # scale < 0

    def test_repr(self) -> None:
        """Test string representation."""
        repr_str = repr(self._dist)
        assert "GammaMarginal" in repr_str
        assert str(self._shape) in repr_str
        assert str(self._scale) in repr_str

    def test_exponential_distribution(self) -> None:
        """Test special case: shape=1 gives exponential distribution."""
        exp_dist = GammaMarginal(shape=1.0, scale=2.0, bkd=self._bkd)
        scipy_exp = stats.expon(scale=2.0)

        samples = self._bkd.asarray([[0.5, 1.0, 2.0]])
        pdf_vals = exp_dist(samples)
        expected = self._bkd.asarray([scipy_exp.pdf([0.5, 1.0, 2.0])])
        assert self._bkd.allclose(pdf_vals, expected, rtol=1e-10)

    def test_ppf_alias(self) -> None:
        """Test ppf is alias for invcdf."""
        probs = self._bkd.asarray([[0.25, 0.5, 0.75]])
        assert self._bkd.allclose(
            self._dist.ppf(probs), self._dist.invcdf(probs), rtol=1e-10
        )


def _grad_of_sum(
    bkd: AutodiffBackend[Array], fun: Callable[[Array], Array], samples: Array
) -> Array:
    """Autodiff gradient of ``sum(fun(samples))`` for ``samples`` of shape
    ``(1, nsamples)``, shaped like ``samples``."""
    nsamples = samples.shape[1]

    def total(flat: Array) -> Array:
        return bkd.reshape(bkd.sum(fun(bkd.reshape(flat, (1, nsamples)))), (1,))

    return bkd.reshape(bkd.jacobian(total, bkd.flatten(samples)), (1, nsamples))


class TestGammaMarginalAutograd:
    """Autodiff derivatives of GammaMarginal w.r.t. its argument."""

    SHAPE = 2.0
    SCALE = 1.5

    def test_logpdf_gradient_wrt_samples(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """d/dx log f(x) = (k-1)/x - rate."""
        bkd = autodiff_bkd
        dist = GammaMarginal(self.SHAPE, self.SCALE, bkd)
        samples = bkd.asarray([[0.5, 1.0, 2.0]])
        rate = 1.0 / self.SCALE
        expected = (self.SHAPE - 1.0) / samples - rate
        bkd.assert_allclose(
            _grad_of_sum(bkd, dist.logpdf, samples), expected, rtol=1e-6
        )

    def test_pdf_gradient_wrt_samples(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """d/dx pdf = pdf * d/dx log(pdf)."""
        bkd = autodiff_bkd
        dist = GammaMarginal(self.SHAPE, self.SCALE, bkd)
        samples = bkd.asarray([[0.5, 1.0, 2.0]])
        rate = 1.0 / self.SCALE
        expected = dist(samples) * ((self.SHAPE - 1.0) / samples - rate)
        bkd.assert_allclose(_grad_of_sum(bkd, dist, samples), expected, rtol=1e-6)
