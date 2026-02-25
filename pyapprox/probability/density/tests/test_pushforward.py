"""Tests for PushforwardDensity: end-to-end density estimation via projection.

Tests include:
- Linear Gaussian pushforward against analytical N(mu, sigma^2) PDF
- Linear Uniform pushforward against analytical uniform density
- Polynomial PCE pushforward against UnivariatePCEDensity ground truth
- Normalization (integral to 1)
- Moment matching (E[Y^k] from density vs quadrature)
- L1 convergence with increasing nbasis
- FunctionProtocol conformance
- Kernel basis density estimation
- Quadratic vs linear basis accuracy comparison
"""

import math
import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import stats

from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.probability import GaussianMarginal
from pyapprox.probability.density._fitters import (
    KDEFitter,
    LinearDensityFitter,
)
from pyapprox.probability.density.kernel_density_basis import (
    KernelDensityBasis,
)
from pyapprox.probability.density.piecewise_density_basis import (
    PiecewiseDensityBasis,
)
from pyapprox.probability.density.pushforward import (
    PushforwardDensity,
)
from pyapprox.surrogates.affine.basis.kernel_basis import KernelBasis
from pyapprox.surrogates.affine.expansions import (
    create_pce_from_marginals,
)
from pyapprox.surrogates.affine.expansions.pce_density import (
    UnivariatePCEDensity,
    composite_gauss_legendre,
)
from pyapprox.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestPushforwardDensity(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _gaussian_quadrature(self, mu: float, sigma: float, nquad: int):
        """Gauss-Hermite quadrature for N(mu, sigma^2).

        Returns (y_values, weights) where y_values = g(xi) for a linear
        pushforward g(xi) = mu + sigma*xi, xi ~ N(0,1).
        """
        bkd = self._bkd
        xi_np, w_np = np.polynomial.hermite_e.hermegauss(nquad)
        y_np = mu + sigma * xi_np
        w_np = w_np / math.sqrt(2.0 * math.pi)
        y_values = bkd.reshape(bkd.asarray(y_np), (1, -1))
        weights = bkd.asarray(w_np)
        return y_values, weights

    def _uniform_quadrature(self, a: float, b: float, nquad: int):
        """Gauss-Legendre quadrature for Uniform(a, b).

        Weights include the 1/(b-a) factor from the uniform density.
        """
        bkd = self._bkd
        xi_np, w_np = np.polynomial.legendre.leggauss(nquad)
        # Map from [-1,1] to [a,b]: x = (b-a)/2 * xi + (a+b)/2
        half_range = (b - a) / 2.0
        mid = (a + b) / 2.0
        y_np = half_range * xi_np + mid
        # Weights: include uniform density 1/(b-a) and Jacobian (b-a)/2
        # So effective weight = w_np * (b-a)/2 * 1/(b-a) = w_np / 2
        w_np = w_np / 2.0
        y_values = bkd.reshape(bkd.asarray(y_np), (1, -1))
        weights = bkd.asarray(w_np)
        return y_values, weights

    def test_linear_gaussian(self) -> None:
        """g(xi) = 2 + 3*xi, xi ~ N(0,1) -> Y ~ N(2, 9)."""
        bkd = self._bkd
        mu, sigma = 2.0, 3.0
        nquad = 80

        # Quadrature for standard normal, then pushforward y = mu + sigma*xi
        y_values, weights = self._gaussian_quadrature(mu, sigma, nquad)

        # Set basis domain from data range
        y_np = bkd.to_numpy(y_values[0])
        y_min, y_max = float(y_np.min()), float(y_np.max())
        basis = PiecewiseDensityBasis(y_min, y_max, 81, degree=1, bkd=bkd)

        density = PushforwardDensity(y_values, weights, basis)

        # Evaluate at test points within effective support
        y_test = bkd.reshape(
            bkd.linspace(mu - 3.0 * sigma, mu + 3.0 * sigma, 50), (1, -1)
        )
        f_approx = density(y_test)

        # Analytical: N(mu, sigma^2)
        y_test_np = bkd.to_numpy(y_test[0])
        f_true = bkd.reshape(
            bkd.asarray(stats.norm.pdf(y_test_np, loc=mu, scale=sigma)),
            (1, -1),
        )

        bkd.assert_allclose(f_approx, f_true, atol=0.005, rtol=0.05)

    def test_linear_uniform(self) -> None:
        """g(xi) = 1 + 2*xi, xi ~ U(-1,1) -> Y ~ U(-1, 3), density = 0.25."""
        bkd = self._bkd
        a_coef, b_coef = 1.0, 2.0
        nquad = 100

        # Quadrature on [-1,1] with uniform density 1/2
        y_quad, w_quad = self._uniform_quadrature(-1.0, 1.0, nquad)

        # Pushforward: y = a + b*xi
        xi_vals = bkd.to_numpy(y_quad[0])
        y_np = a_coef + b_coef * xi_vals
        y_values = bkd.reshape(bkd.asarray(y_np), (1, -1))

        # Set basis domain from data range
        y_min_data = float(y_np.min())
        y_max_data = float(y_np.max())
        basis = PiecewiseDensityBasis(
            y_min_data,
            y_max_data,
            41,
            degree=1,
            bkd=bkd,
        )

        density = PushforwardDensity(y_values, w_quad, basis)

        # Evaluate at interior points of output range [-1, 3]
        y_test = bkd.reshape(bkd.linspace(-0.5, 2.5, 30), (1, -1))
        f_approx = density(y_test)

        # Analytical density of Y ~ U(-1, 3) is 1/4 = 0.25
        f_true = bkd.full((1, 30), 0.25)
        bkd.assert_allclose(f_approx, f_true, atol=0.05)

    def test_polynomial_vs_baseline(self) -> None:
        """Degree-5 PCE with random coefficients: compare to UnivariatePCEDensity."""
        bkd = self._bkd
        marginal = GaussianMarginal(0.0, 1.0, bkd)
        pce = create_pce_from_marginals([marginal], max_level=5, bkd=bkd)

        np.random.seed(42)
        coef_np = np.random.randn(pce.nterms(), 1) * 0.5
        coef_np[0, 0] = 1.0  # non-zero mean
        pce.set_coefficients(bkd.asarray(coef_np))

        # Ground truth via companion matrix root-finding
        ground_truth = UnivariatePCEDensity(pce, marginal)

        # Generate quadrature in xi-space (standard normal)
        nquad = 80
        xi_np, w_np = np.polynomial.hermite_e.hermegauss(nquad)
        w_np = w_np / math.sqrt(2.0 * math.pi)

        # Pushforward: y = pce(xi)
        xi_2d = bkd.reshape(bkd.asarray(xi_np), (1, -1))
        y_values = pce(xi_2d)  # (1, nquad)
        weights = bkd.asarray(w_np)

        # Set basis domain from y_values range
        y_np = bkd.to_numpy(y_values[0])
        y_min, y_max = float(y_np.min()), float(y_np.max())
        basis = PiecewiseDensityBasis(y_min, y_max, 81, degree=1, bkd=bkd)

        density = PushforwardDensity(y_values, weights, basis)

        # Compare on a grid inside the effective support
        y_grid = np.linspace(y_min + 0.5, y_max - 0.5, 40)
        y_test = bkd.reshape(bkd.asarray(y_grid), (1, -1))
        f_approx = density(y_test)
        f_true = ground_truth.pdf(y_test)

        bkd.assert_allclose(f_approx, f_true, atol=0.02, rtol=0.15)

    def test_normalization(self) -> None:
        """Integral of estimated density should be approximately 1."""
        bkd = self._bkd
        y_values, weights = self._gaussian_quadrature(0.0, 1.0, 80)

        y_np = bkd.to_numpy(y_values[0])
        y_min, y_max = float(y_np.min()), float(y_np.max())
        basis = PiecewiseDensityBasis(y_min, y_max, 81, degree=1, bkd=bkd)

        density = PushforwardDensity(y_values, weights, basis)

        def integrand(y_np_arr: np.ndarray) -> np.ndarray:
            y_arr = bkd.reshape(bkd.asarray(y_np_arr), (1, -1))
            f = density(y_arr)
            return bkd.to_numpy(f[0])

        integral = composite_gauss_legendre(
            integrand,
            y_min,
            y_max,
            n_intervals=500,
            n_points=5,
        )
        bkd.assert_allclose(
            bkd.asarray([integral]),
            bkd.asarray([1.0]),
            atol=0.01,
        )

    def test_moment_matching(self) -> None:
        """E[Y^k] from density integral matches xi-space quadrature."""
        bkd = self._bkd
        mu, sigma = 1.0, 2.0
        nquad = 80

        y_values, weights = self._gaussian_quadrature(mu, sigma, nquad)

        y_np = bkd.to_numpy(y_values[0])
        y_min, y_max = float(y_np.min()), float(y_np.max())
        basis = PiecewiseDensityBasis(y_min, y_max, 81, degree=1, bkd=bkd)

        density = PushforwardDensity(y_values, weights, basis)

        # Compute reference moments via xi-space quadrature
        # E[Y^k] = sum_q w_q * y_q^k
        for kk in range(1, 5):
            moment_quad = float(bkd.to_numpy(bkd.sum(weights * y_values[0] ** kk)))

            # Compute from density integral: int y^k f(y) dy
            def integrand(y_np_arr: np.ndarray, k=kk) -> np.ndarray:
                y_arr = bkd.reshape(bkd.asarray(y_np_arr), (1, -1))
                f = density(y_arr)
                return bkd.to_numpy(f[0]) * y_np_arr**k

            moment_dens = composite_gauss_legendre(
                integrand,
                y_min,
                y_max,
                n_intervals=500,
                n_points=5,
            )
            bkd.assert_allclose(
                bkd.asarray([moment_dens]),
                bkd.asarray([moment_quad]),
                atol=0.05,
                rtol=0.05,
            )

    def test_l1_convergence_nbasis(self) -> None:
        """L1 error vs ground truth decreases with increasing nbasis."""
        bkd = self._bkd
        marginal = GaussianMarginal(0.0, 1.0, bkd)
        pce = create_pce_from_marginals([marginal], max_level=3, bkd=bkd)

        # Cubic PCE with dominant linear term
        coef = bkd.zeros((pce.nterms(), 1))
        coef[0, 0] = 1.0
        coef[1, 0] = 2.0
        coef[2, 0] = 0.3
        coef[3, 0] = 0.1
        pce.set_coefficients(coef)

        ground_truth = UnivariatePCEDensity(pce, marginal)

        # Quadrature in xi-space
        nquad = 100
        xi_np, w_np = np.polynomial.hermite_e.hermegauss(nquad)
        w_np = w_np / math.sqrt(2.0 * math.pi)
        xi_2d = bkd.reshape(bkd.asarray(xi_np), (1, -1))
        y_values = pce(xi_2d)
        weights = bkd.asarray(w_np)

        # Use get_range for a well-matched domain
        y_range = ground_truth.get_range()
        margin = 0.5
        y_min, y_max = y_range[0] - margin, y_range[1] + margin

        # Common test grid inside effective support
        y_grid = np.linspace(y_min + 0.3, y_max - 0.3, 200)
        y_test = bkd.reshape(bkd.asarray(y_grid), (1, -1))
        f_true_np = bkd.to_numpy(ground_truth.pdf(y_test)[0])
        dy = (y_grid[-1] - y_grid[0]) / len(y_grid)

        l1_errors = []
        nbasis_list = [7, 11, 21]
        for nb in nbasis_list:
            basis = PiecewiseDensityBasis(y_min, y_max, nb, degree=1, bkd=bkd)
            dens = PushforwardDensity(y_values, weights, basis)
            f_approx_np = bkd.to_numpy(dens(y_test)[0])
            l1 = float(np.sum(np.abs(f_approx_np - f_true_np)) * dy)
            l1_errors.append(l1)

        # Each increase in nbasis should reduce error
        for ii in range(1, len(l1_errors)):
            self.assertLess(
                l1_errors[ii],
                l1_errors[ii - 1],
                f"L1 error did not decrease: nbasis={nbasis_list[ii]} "
                f"({l1_errors[ii]:.6f}) >= nbasis={nbasis_list[ii - 1]} "
                f"({l1_errors[ii - 1]:.6f})",
            )

    def test_function_protocol(self) -> None:
        """PushforwardDensity should satisfy FunctionProtocol."""
        bkd = self._bkd
        y_values, weights = self._gaussian_quadrature(0.0, 1.0, 40)
        basis = PiecewiseDensityBasis(-5.0, 5.0, 21, degree=1, bkd=bkd)
        density = PushforwardDensity(y_values, weights, basis)
        assert isinstance(density, FunctionProtocol)
        bkd.assert_allclose(
            bkd.asarray([density.nvars()]),
            bkd.asarray([1]),
        )
        bkd.assert_allclose(
            bkd.asarray([density.nqoi()]),
            bkd.asarray([1]),
        )

    def test_output_shape(self) -> None:
        """Output shape should be (1, npts)."""
        bkd = self._bkd
        y_values, weights = self._gaussian_quadrature(0.0, 1.0, 40)
        basis = PiecewiseDensityBasis(-5.0, 5.0, 21, degree=1, bkd=bkd)
        density = PushforwardDensity(y_values, weights, basis)

        npts = 17
        y_test = bkd.reshape(bkd.linspace(-3.0, 3.0, npts), (1, -1))
        result = density(y_test)
        bkd.assert_allclose(
            bkd.asarray([result.shape[0], result.shape[1]]),
            bkd.asarray([1, npts]),
        )

    def test_kernel_basis_density(self) -> None:
        """PushforwardDensity with KernelDensityBasis approximates N(0,1)."""
        bkd = self._bkd
        nquad = 80
        y_values, weights = self._gaussian_quadrature(0.0, 1.0, nquad)

        kernel = SquaredExponentialKernel(
            bkd.asarray([0.5]),
            (0.01, 100.0),
            1,
            bkd,
        )
        centers = bkd.reshape(bkd.linspace(-4.0, 4.0, 25), (1, -1))
        kb = KernelBasis(kernel, centers)
        basis = KernelDensityBasis(kb)

        density = PushforwardDensity(y_values, weights, basis)

        y_test = bkd.reshape(bkd.linspace(-3.0, 3.0, 50), (1, -1))
        f_approx = density(y_test)

        y_test_np = bkd.to_numpy(y_test[0])
        f_true = bkd.reshape(
            bkd.asarray(stats.norm.pdf(y_test_np)),
            (1, -1),
        )
        bkd.assert_allclose(f_approx, f_true, atol=0.01, rtol=0.05)

    def test_coefficients_accessor(self) -> None:
        """coefficients() returns correct shape."""
        bkd = self._bkd
        y_values, weights = self._gaussian_quadrature(0.0, 1.0, 40)
        nbasis = 15
        basis = PiecewiseDensityBasis(-5.0, 5.0, nbasis, degree=1, bkd=bkd)
        density = PushforwardDensity(y_values, weights, basis)
        bkd.assert_allclose(
            bkd.asarray([density.coefficients().shape[0]]),
            bkd.asarray([nbasis]),
        )

    def test_basis_accessor(self) -> None:
        """basis() returns the same basis object."""
        bkd = self._bkd
        y_values, weights = self._gaussian_quadrature(0.0, 1.0, 40)
        basis = PiecewiseDensityBasis(-5.0, 5.0, 15, degree=1, bkd=bkd)
        density = PushforwardDensity(y_values, weights, basis)
        assert density.basis() is basis

    def test_invalid_basis_type(self) -> None:
        """Should reject non-DensityBasisProtocol objects."""
        y_values, weights = self._gaussian_quadrature(0.0, 1.0, 40)
        with self.assertRaises(TypeError):
            PushforwardDensity(
                y_values,
                weights,
                "not a basis",  # type: ignore
            )

    def test_default_fitter_is_linear(self) -> None:
        """Default fitter should be LinearDensityFitter."""
        bkd = self._bkd
        y_values, weights = self._gaussian_quadrature(0.0, 1.0, 40)
        basis = PiecewiseDensityBasis(-5.0, 5.0, 21, degree=1, bkd=bkd)
        density = PushforwardDensity(y_values, weights, basis)
        assert isinstance(density.fitter(), LinearDensityFitter)

    def test_explicit_linear_fitter_same_as_default(self) -> None:
        """Passing LinearDensityFitter explicitly gives same coefficients."""
        bkd = self._bkd
        y_values, weights = self._gaussian_quadrature(0.0, 1.0, 40)
        basis = PiecewiseDensityBasis(-5.0, 5.0, 21, degree=1, bkd=bkd)
        d_default = PushforwardDensity(y_values, weights, basis)
        d_explicit = PushforwardDensity(
            y_values,
            weights,
            basis,
            fitter=LinearDensityFitter(bkd),
        )
        bkd.assert_allclose(
            d_explicit.coefficients(),
            d_default.coefficients(),
            rtol=1e-14,
        )

    def test_ise_score_consistent(self) -> None:
        """ise_score should equal d^T M d - 2 * d^T b."""
        bkd = self._bkd
        y_values, weights = self._gaussian_quadrature(0.0, 1.0, 60)
        basis = PiecewiseDensityBasis(-5.0, 5.0, 21, degree=1, bkd=bkd)
        density = PushforwardDensity(y_values, weights, basis)

        d = density.coefficients()
        M = basis.mass_matrix()
        Phi = basis.evaluate(y_values)
        b = bkd.dot(Phi, weights)

        Md = bkd.dot(M, d)
        expected = bkd.sum(d * Md) - 2.0 * bkd.sum(d * b)
        bkd.assert_allclose(
            bkd.reshape(density.ise_score(), (1,)),
            bkd.reshape(expected, (1,)),
            rtol=1e-10,
        )

    def test_ise_score_linear_equals_neg_btMinvb(self) -> None:
        """For L2 projection, ise_score = -b^T M^{-1} b."""
        bkd = self._bkd
        y_values, weights = self._gaussian_quadrature(0.0, 1.0, 60)
        basis = PiecewiseDensityBasis(-5.0, 5.0, 21, degree=1, bkd=bkd)
        density = PushforwardDensity(y_values, weights, basis)

        # For M*d = b: d^T M d - 2*d^T b = b^T M^{-1} b - 2*b^T M^{-1} b
        #                                 = -b^T M^{-1} b
        M = basis.mass_matrix()
        Phi = basis.evaluate(y_values)
        b = bkd.dot(Phi, weights)
        M_inv_b = bkd.solve(M, b)
        neg_btMinvb = -bkd.sum(b * M_inv_b)

        bkd.assert_allclose(
            bkd.reshape(density.ise_score(), (1,)),
            bkd.reshape(neg_btMinvb, (1,)),
            rtol=1e-10,
        )

    def test_kde_fitter_through_pushforward(self) -> None:
        """KDEFitter produces density with coefficients equal to load vector."""
        bkd = self._bkd
        y_values, weights = self._gaussian_quadrature(0.0, 1.0, 40)
        basis = PiecewiseDensityBasis(-5.0, 5.0, 21, degree=1, bkd=bkd)

        fitter = KDEFitter(bkd)
        density = PushforwardDensity(y_values, weights, basis, fitter=fitter)

        # Coefficients should be the load vector b = Phi @ weights
        Phi = basis.evaluate(y_values)
        b = bkd.dot(Phi, weights)
        bkd.assert_allclose(density.coefficients(), b, rtol=1e-14)

        # Should still be callable
        y_test = bkd.reshape(bkd.linspace(-3.0, 3.0, 10), (1, -1))
        result = density(y_test)
        bkd.assert_allclose(
            bkd.asarray([result.shape[0], result.shape[1]]),
            bkd.asarray([1, 10]),
        )

    def test_invalid_fitter_type(self) -> None:
        """Should reject non-DensityFitterProtocol objects."""
        bkd = self._bkd
        y_values, weights = self._gaussian_quadrature(0.0, 1.0, 40)
        basis = PiecewiseDensityBasis(-5.0, 5.0, 21, degree=1, bkd=bkd)
        with self.assertRaises(TypeError):
            PushforwardDensity(
                y_values,
                weights,
                basis,
                fitter="not a fitter",  # type: ignore
            )


class TestPushforwardDensityNumpy(TestPushforwardDensity[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPushforwardDensityTorch(TestPushforwardDensity[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
