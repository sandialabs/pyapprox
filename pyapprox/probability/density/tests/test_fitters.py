"""Tests for density fitters."""

import math

import numpy as np
from scipy import stats

from pyapprox.probability.density._fitters import (
    DensityFitterProtocol,
    KDEFitter,
    LinearDensityFitter,
)
from pyapprox.probability.density.kernel_density_basis import (
    KernelDensityBasis,
)
from pyapprox.surrogates.affine.basis.kernel_basis import KernelBasis
from pyapprox.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)


class TestDensityFitters:

    def test_linear_fitter_solves_system(self, bkd) -> None:
        """LinearDensityFitter solves M*d = b correctly."""
        M = bkd.asarray([[2.0, 1.0], [1.0, 3.0]])
        b = bkd.asarray([5.0, 7.0])
        fitter = LinearDensityFitter(bkd)
        d = fitter.fit(M, b)
        # Verify M*d = b
        Md = bkd.dot(M, d)
        bkd.assert_allclose(Md, b, rtol=1e-12)

    def test_linear_fitter_protocol_conformance(self, bkd) -> None:
        """LinearDensityFitter satisfies DensityFitterProtocol."""
        fitter = LinearDensityFitter(bkd)
        assert isinstance(fitter, DensityFitterProtocol)

    def test_kde_fitter_returns_load_vector(self, bkd) -> None:
        """KDEFitter returns the load vector directly."""
        M = bkd.asarray([[2.0, 1.0], [1.0, 3.0]])
        b = bkd.asarray([0.3, 0.7])
        fitter = KDEFitter(bkd)
        d = fitter.fit(M, b)
        bkd.assert_allclose(d, b, rtol=1e-14)

    def test_kde_fitter_protocol_conformance(self, bkd) -> None:
        """KDEFitter satisfies DensityFitterProtocol."""
        fitter = KDEFitter(bkd)
        assert isinstance(fitter, DensityFitterProtocol)

    def test_kde_fitter_ignores_mass_matrix(self, bkd) -> None:
        """KDEFitter returns the same result regardless of mass matrix."""
        b = bkd.asarray([1.0, 2.0, 3.0])
        M1 = bkd.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        M2 = bkd.asarray([[5.0, 2.0, 1.0], [2.0, 6.0, 3.0], [1.0, 3.0, 7.0]])
        fitter = KDEFitter(bkd)
        d1 = fitter.fit(M1, b)
        d2 = fitter.fit(M2, b)
        bkd.assert_allclose(d1, d2, rtol=1e-14)

    def test_kde_matches_scipy_gaussian_kde(self, bkd) -> None:
        """KDEFitter with SE kernel at data points matches scipy KDE."""
        np.random.seed(42)
        N = 50
        data = np.random.randn(N)

        # Build scipy KDE with default (Scott's rule) bandwidth
        kde_scipy = stats.gaussian_kde(data)

        # Extract the actual kernel sigma from scipy
        # scipy: covariance = np.cov(data) * factor^2
        # For 1D: sigma = sqrt(covariance[0,0])
        sigma = float(np.sqrt(kde_scipy.covariance[0, 0]))

        # Our SE kernel with l = sigma, centers at data points
        kernel = SquaredExponentialKernel(
            bkd.asarray([sigma]),
            (0.01, 100.0),
            1,
            bkd,
        )
        centers = bkd.reshape(bkd.asarray(data), (1, -1))
        kb = KernelBasis(kernel, centers)
        basis = KernelDensityBasis(kb)

        # KDE coefficients: d_i = 1 / (N * sigma * sqrt(2*pi))
        # Our kernel: K(y, x_i) = exp(-0.5*(y-x_i)^2/sigma^2) (unnormalized)
        # Scipy kernel: (1/(sigma*sqrt(2*pi))) * exp(-0.5*(y-x_i)^2/sigma^2)
        # So d_i * K(y, x_i) = (1/N) * scipy_kernel(y, x_i)
        kde_coeff = 1.0 / (N * sigma * math.sqrt(2.0 * math.pi))
        desired_coeffs = bkd.full((N,), kde_coeff)

        # Use KDEFitter to set coefficients directly
        fitter = KDEFitter(bkd)
        M = basis.mass_matrix()
        d = fitter.fit(M, desired_coeffs)

        # Evaluate density: f(y) = sum_i d_i * K(y, x_i)
        y_test = bkd.reshape(bkd.linspace(-3.0, 3.0, 100), (1, -1))
        Phi = basis.evaluate(y_test)  # (N, 100)
        d_row = bkd.reshape(d, (1, -1))  # (1, N)
        f_approx = bkd.dot(d_row, Phi)  # (1, 100)

        # scipy reference
        y_test_np = bkd.to_numpy(y_test[0])
        f_scipy_np = kde_scipy(y_test_np)
        f_scipy = bkd.reshape(bkd.asarray(f_scipy_np), (1, -1))

        bkd.assert_allclose(f_approx, f_scipy, rtol=1e-10)
