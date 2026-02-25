"""Tests for density fitters."""

import math
import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import stats

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401

from pyapprox.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)
from pyapprox.surrogates.affine.basis.kernel_basis import KernelBasis
from pyapprox.probability.density.kernel_density_basis import (
    KernelDensityBasis,
)
from pyapprox.probability.density._fitters import (
    DensityFitterProtocol,
    KDEFitter,
    LinearDensityFitter,
)


class TestDensityFitters(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_linear_fitter_solves_system(self) -> None:
        """LinearDensityFitter solves M*d = b correctly."""
        bkd = self._bkd
        M = bkd.asarray([[2.0, 1.0], [1.0, 3.0]])
        b = bkd.asarray([5.0, 7.0])
        fitter = LinearDensityFitter(bkd)
        d = fitter.fit(M, b)
        # Verify M*d = b
        Md = bkd.dot(M, d)
        bkd.assert_allclose(Md, b, rtol=1e-12)

    def test_linear_fitter_protocol_conformance(self) -> None:
        """LinearDensityFitter satisfies DensityFitterProtocol."""
        bkd = self._bkd
        fitter = LinearDensityFitter(bkd)
        assert isinstance(fitter, DensityFitterProtocol)

    def test_kde_fitter_returns_load_vector(self) -> None:
        """KDEFitter returns the load vector directly."""
        bkd = self._bkd
        M = bkd.asarray([[2.0, 1.0], [1.0, 3.0]])
        b = bkd.asarray([0.3, 0.7])
        fitter = KDEFitter(bkd)
        d = fitter.fit(M, b)
        bkd.assert_allclose(d, b, rtol=1e-14)

    def test_kde_fitter_protocol_conformance(self) -> None:
        """KDEFitter satisfies DensityFitterProtocol."""
        bkd = self._bkd
        fitter = KDEFitter(bkd)
        assert isinstance(fitter, DensityFitterProtocol)

    def test_kde_fitter_ignores_mass_matrix(self) -> None:
        """KDEFitter returns the same result regardless of mass matrix."""
        bkd = self._bkd
        b = bkd.asarray([1.0, 2.0, 3.0])
        M1 = bkd.asarray([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        M2 = bkd.asarray([[5.0, 2.0, 1.0],
                           [2.0, 6.0, 3.0],
                           [1.0, 3.0, 7.0]])
        fitter = KDEFitter(bkd)
        d1 = fitter.fit(M1, b)
        d2 = fitter.fit(M2, b)
        bkd.assert_allclose(d1, d2, rtol=1e-14)

    def test_kde_matches_scipy_gaussian_kde(self) -> None:
        """KDEFitter with SE kernel at data points matches scipy KDE."""
        bkd = self._bkd
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
            bkd.asarray([sigma]), (0.01, 100.0), 1, bkd,
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


class TestDensityFittersNumpy(TestDensityFitters[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDensityFittersTorch(TestDensityFitters[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
