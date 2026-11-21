"""
Tests that Gaussian distributions keep the computation graph intact.

The analytical derivatives of logpdf, pdf, cdf and invcdf are compared with
the backend's autodiff Jacobian and Hessian-vector product, and the autodiff
gradients are validated against finite differences with DerivativeChecker.
"""

import math
from typing import Callable

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.probability.gaussian import (
    DenseCholeskyMultivariateGaussian,
    DiagonalMultivariateGaussian,
)
from pyapprox.probability.univariate import GaussianMarginal
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array

# TODO: Should be move to the submodule it is testing


def _grad_of_sum(
    bkd: AutodiffBackend[Array], fun: Callable[[Array], Array], samples: Array
) -> Array:
    """Autodiff gradient of ``sum(fun(samples))``, shaped like ``samples``."""
    shape = tuple(samples.shape)

    def total(flat: Array) -> Array:
        return bkd.reshape(bkd.sum(fun(bkd.reshape(flat, shape))), (1,))

    return bkd.reshape(bkd.jacobian(total, bkd.flatten(samples)), shape)


def _fd_error_ratio(
    bkd: AutodiffBackend[Array], fun: Callable[[Array], Array], samples: Array
) -> float:
    """DerivativeChecker error ratio of the autodiff gradient of
    ``sum(fun(samples))``."""
    shape = tuple(samples.shape)
    nvars = math.prod(shape)

    def total(x: Array) -> Array:
        return bkd.reshape(bkd.sum(fun(bkd.reshape(x, shape))), (1, 1))

    def jacobian(x: Array) -> Array:
        return bkd.reshape(_grad_of_sum(bkd, fun, bkd.reshape(x, shape)), (1, nvars))

    checker = DerivativeChecker(
        FunctionWithJacobianFromCallable(
            nqoi=1, nvars=nvars, fun=total, jacobian=jacobian, bkd=bkd
        )
    )
    errors = checker.check_derivatives(bkd.reshape(samples, (nvars, 1)), verbosity=0)
    return float(bkd.to_numpy(checker.error_ratio(errors[0])))


class TestGaussianMarginalAutograd:
    """Autodiff derivatives of GaussianMarginal w.r.t. its argument."""

    MEAN = 2.0
    STDEV = 0.5

    def _dist(self, bkd: AutodiffBackend[Array]) -> GaussianMarginal[Array]:
        return GaussianMarginal(self.MEAN, self.STDEV, bkd)

    def test_logpdf_gradient_wrt_samples(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """d/dx log(pdf) = -(x - mu) / sigma^2."""
        bkd = autodiff_bkd
        dist = self._dist(bkd)
        samples = bkd.asarray([[1.0, 2.0, 3.0]])  # (1, nsamples)
        expected = -(samples - self.MEAN) / (self.STDEV**2)
        bkd.assert_allclose(
            _grad_of_sum(bkd, dist.logpdf, samples), expected, rtol=1e-6
        )

    def test_pdf_gradient_wrt_samples(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """d/dx pdf = pdf * (-(x - mu) / sigma^2)."""
        bkd = autodiff_bkd
        dist = self._dist(bkd)
        samples = bkd.asarray([[1.5, 2.0, 2.5]])
        expected = dist(samples) * (-(samples - self.MEAN) / (self.STDEV**2))
        bkd.assert_allclose(_grad_of_sum(bkd, dist, samples), expected, rtol=1e-6)

    def test_cdf_gradient_wrt_samples(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """d/dx CDF = pdf."""
        bkd = autodiff_bkd
        dist = self._dist(bkd)
        samples = bkd.asarray([[1.5, 2.0, 2.5]])
        bkd.assert_allclose(
            _grad_of_sum(bkd, dist.cdf, samples), dist(samples), rtol=1e-6
        )

    def test_invcdf_gradient_wrt_probs(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """d/dp invcdf = 1 / pdf(invcdf(p))."""
        bkd = autodiff_bkd
        dist = self._dist(bkd)
        probs = bkd.asarray([[0.25, 0.5, 0.75]])
        expected = 1.0 / dist(dist.invcdf(probs))
        bkd.assert_allclose(_grad_of_sum(bkd, dist.invcdf, probs), expected, rtol=1e-5)

    def test_logpdf_second_derivative(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """d^2/dx^2 log(pdf) = -1/sigma^2; the Hessian of the summed logpdf is
        diagonal, so its product with ones gives the second derivatives."""
        bkd = autodiff_bkd
        dist = self._dist(bkd)
        samples = bkd.asarray([[1.5, 2.0, 2.5]])
        nsamples = samples.shape[1]

        def total(flat: Array) -> Array:
            return bkd.sum(dist.logpdf(bkd.reshape(flat, (1, nsamples))))

        second = bkd.hvp(total, bkd.flatten(samples), bkd.ones((nsamples,)))
        expected = bkd.full((nsamples,), -1.0 / (self.STDEV**2))
        bkd.assert_allclose(second, expected, rtol=1e-6)


class TestGaussianMarginalAutogradFiniteDiff:
    """Autodiff gradients of GaussianMarginal match finite differences."""

    def test_logpdf_gradient_finite_diff(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        bkd = autodiff_bkd
        dist = GaussianMarginal(1.0, 2.0, bkd)
        samples = bkd.asarray([[0.5, 1.0, 1.5]])  # (1, nsamples)
        assert _fd_error_ratio(bkd, dist.logpdf, samples) <= 1e-6


def _dense_gaussian(
    bkd: AutodiffBackend[Array],
) -> DenseCholeskyMultivariateGaussian[Array]:
    mean = bkd.asarray([[1.0], [2.0], [3.0]])
    cov = bkd.asarray(
        [
            [2.0, 0.5, 0.1],
            [0.5, 1.5, 0.3],
            [0.1, 0.3, 1.0],
        ]
    )
    return DenseCholeskyMultivariateGaussian(mean, cov, bkd)


class TestDenseCholeskyMultivariateGaussianAutograd:
    """Autodiff derivatives of DenseCholeskyMultivariateGaussian."""

    def test_logpdf_gradient_wrt_samples(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """Autodiff gradient matches logpdf_gradient."""
        bkd = autodiff_bkd
        dist = _dense_gaussian(bkd)
        samples = bkd.asarray([[0.5, 1.5], [1.0, 2.5], [2.0, 3.5]])
        bkd.assert_allclose(
            _grad_of_sum(bkd, dist.logpdf, samples),
            dist.logpdf_gradient(samples),
            rtol=1e-6,
        )

    def test_logpdf_gradient_finite_diff(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        bkd = autodiff_bkd
        dist = _dense_gaussian(bkd)
        samples = bkd.asarray([[0.5], [1.0], [2.0]])
        assert _fd_error_ratio(bkd, dist.logpdf, samples) <= 1e-6

    def test_pdf_gradient_wrt_samples(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """d/dx pdf = pdf * grad_log_pdf."""
        bkd = autodiff_bkd
        dist = _dense_gaussian(bkd)
        samples = bkd.asarray([[1.0], [2.0], [3.0]])
        expected = dist.pdf(samples) * dist.logpdf_gradient(samples)
        bkd.assert_allclose(_grad_of_sum(bkd, dist.pdf, samples), expected, rtol=1e-6)


def _diagonal_gaussian(
    bkd: AutodiffBackend[Array],
) -> DiagonalMultivariateGaussian[Array]:
    mean = bkd.asarray([[0.0], [1.0], [2.0]])
    variance = bkd.asarray([1.0, 2.0, 0.5])
    return DiagonalMultivariateGaussian(mean, variance, bkd)


class TestDiagonalMultivariateGaussianAutograd:
    """Autodiff derivatives of DiagonalMultivariateGaussian."""

    def test_logpdf_gradient_wrt_samples(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """Autodiff gradient matches logpdf_gradient."""
        bkd = autodiff_bkd
        dist = _diagonal_gaussian(bkd)
        samples = bkd.asarray([[0.5, -0.5], [1.5, 0.5], [2.5, 1.5]])
        bkd.assert_allclose(
            _grad_of_sum(bkd, dist.logpdf, samples),
            dist.logpdf_gradient(samples),
            rtol=1e-6,
        )

    def test_logpdf_gradient_finite_diff(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        bkd = autodiff_bkd
        dist = _diagonal_gaussian(bkd)
        samples = bkd.asarray([[0.5], [1.0], [2.0]])
        assert _fd_error_ratio(bkd, dist.logpdf, samples) <= 1e-6
