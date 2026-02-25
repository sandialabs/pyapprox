"""Tests for ProjectionDensityFitter."""

import math
import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.probability.density.kernel_density_basis import (
    KernelDensityBasis,
)
from pyapprox.probability.density.piecewise_density_basis import (
    PiecewiseDensityBasis,
)
from pyapprox.probability.density.projection import (
    ISEOptimizingFitter,
    ProjectionDensityFitter,
)
from pyapprox.surrogates.affine.basis.kernel_basis import KernelBasis
from pyapprox.surrogates.affine.expansions.pce_density import (
    composite_gauss_legendre,
)
from pyapprox.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestProjectionDensityFitter(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _gaussian_pdf(self, y: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """Standard Gaussian PDF for reference."""
        return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(
            -0.5 * ((y - mu) / sigma) ** 2
        )

    def _make_gaussian_quadrature(self, mu: float, sigma: float, nquad: int):
        """Generate Gauss-Hermite quadrature for N(mu, sigma^2).

        Returns (y_values, weights) where y_values has shape (1, nquad).
        """
        bkd = self._bkd
        # Gauss-Hermite nodes/weights for standard normal
        xi_np, w_np = np.polynomial.hermite_e.hermegauss(nquad)
        # Transform: y = mu + sigma * xi
        y_np = mu + sigma * xi_np
        # Weights include the exp(-x^2/2) / sqrt(2pi) normalization
        # hermegauss uses weight function exp(-x^2/2), so
        # int f(x) exp(-x^2/2) dx ≈ sum w_i f(x_i)
        # We want int f(y) p(y) dy where p(y) = N(mu, sigma^2)
        # = int f(mu + sigma*xi) * (1/sqrt(2pi)) * exp(-xi^2/2) dxi
        # ≈ sum (w_i / sqrt(2pi)) * f(mu + sigma*xi_i)
        w_np = w_np / math.sqrt(2.0 * math.pi)
        y_values = bkd.reshape(bkd.asarray(y_np), (1, -1))
        weights = bkd.asarray(w_np)
        return y_values, weights

    def test_fit_coefficients_shape_piecewise(self) -> None:
        """Fit returns correct shape with piecewise basis."""
        bkd = self._bkd
        basis = PiecewiseDensityBasis(-3.0, 3.0, 21, degree=1, bkd=bkd)
        fitter = ProjectionDensityFitter(basis)
        y_values, weights = self._make_gaussian_quadrature(0.0, 1.0, 50)
        d = fitter.fit(y_values, weights)
        bkd.assert_allclose(
            bkd.asarray([d.shape[0]]),
            bkd.asarray([21]),
        )

    def test_fit_coefficients_shape_kernel(self) -> None:
        """Fit returns correct shape with kernel basis."""
        bkd = self._bkd
        kernel = SquaredExponentialKernel(
            bkd.asarray([0.5]),
            (0.01, 100.0),
            1,
            bkd,
        )
        centers = bkd.reshape(bkd.linspace(-3.0, 3.0, 15), (1, -1))
        kb = KernelBasis(kernel, centers)
        basis = KernelDensityBasis(kb)
        fitter = ProjectionDensityFitter(basis)
        y_values, weights = self._make_gaussian_quadrature(0.0, 1.0, 50)
        d = fitter.fit(y_values, weights)
        bkd.assert_allclose(
            bkd.asarray([d.shape[0]]),
            bkd.asarray([15]),
        )

    def test_gaussian_density_piecewise_linear(self) -> None:
        """Projected density approximates N(0,1) with piecewise linear basis."""
        bkd = self._bkd
        y_values, weights = self._make_gaussian_quadrature(0.0, 1.0, 80)
        # Set basis domain from data range
        y_np = bkd.to_numpy(y_values[0])
        y_min, y_max = float(y_np.min()), float(y_np.max())
        basis = PiecewiseDensityBasis(y_min, y_max, 81, degree=1, bkd=bkd)
        fitter = ProjectionDensityFitter(basis)
        d = fitter.fit(y_values, weights)

        # Evaluate density at test points inside effective support
        y_test = bkd.reshape(bkd.linspace(-3.0, 3.0, 50), (1, -1))
        Phi = basis.evaluate(y_test)  # (nbasis, npts)
        f_approx = bkd.dot(bkd.reshape(d, (1, -1)), Phi)[0]  # (npts,)

        # Compare to true N(0,1) pdf
        y_test_np = bkd.to_numpy(y_test[0])
        f_true_np = self._gaussian_pdf(y_test_np, 0.0, 1.0)
        f_true = bkd.asarray(f_true_np)

        bkd.assert_allclose(f_approx, f_true, atol=0.01, rtol=0.05)

    def test_gaussian_density_piecewise_quadratic(self) -> None:
        """Quadratic basis should approximate N(0,1) more accurately."""
        bkd = self._bkd
        y_values, weights = self._make_gaussian_quadrature(0.0, 1.0, 80)
        y_np = bkd.to_numpy(y_values[0])
        y_min, y_max = float(y_np.min()), float(y_np.max())
        basis = PiecewiseDensityBasis(y_min, y_max, 81, degree=2, bkd=bkd)
        fitter = ProjectionDensityFitter(basis)
        d = fitter.fit(y_values, weights)

        y_test = bkd.reshape(bkd.linspace(-3.0, 3.0, 50), (1, -1))
        Phi = basis.evaluate(y_test)
        f_approx = bkd.dot(bkd.reshape(d, (1, -1)), Phi)[0]

        y_test_np = bkd.to_numpy(y_test[0])
        f_true_np = self._gaussian_pdf(y_test_np, 0.0, 1.0)
        f_true = bkd.asarray(f_true_np)

        bkd.assert_allclose(f_approx, f_true, atol=0.04, rtol=0.1)

    def test_gaussian_density_kernel(self) -> None:
        """Kernel basis density approximates N(0,1)."""
        bkd = self._bkd
        kernel = SquaredExponentialKernel(
            bkd.asarray([0.5]),
            (0.01, 100.0),
            1,
            bkd,
        )
        centers = bkd.reshape(bkd.linspace(-4.0, 4.0, 25), (1, -1))
        kb = KernelBasis(kernel, centers)
        basis = KernelDensityBasis(kb)
        fitter = ProjectionDensityFitter(basis)
        y_values, weights = self._make_gaussian_quadrature(0.0, 1.0, 80)
        d = fitter.fit(y_values, weights)

        y_test = bkd.reshape(bkd.linspace(-3.0, 3.0, 50), (1, -1))
        Phi = basis.evaluate(y_test)
        f_approx = bkd.dot(bkd.reshape(d, (1, -1)), Phi)[0]

        y_test_np = bkd.to_numpy(y_test[0])
        f_true_np = self._gaussian_pdf(y_test_np, 0.0, 1.0)
        f_true = bkd.asarray(f_true_np)

        bkd.assert_allclose(f_approx, f_true, atol=0.01, rtol=0.05)

    def test_normalization_piecewise(self) -> None:
        """Projected density integrates to 1."""
        bkd = self._bkd
        basis = PiecewiseDensityBasis(-5.0, 5.0, 31, degree=1, bkd=bkd)
        fitter = ProjectionDensityFitter(basis)
        y_values, weights = self._make_gaussian_quadrature(0.0, 1.0, 60)
        d = fitter.fit(y_values, weights)

        y_min, y_max = basis.domain()

        def integrand(y_np: np.ndarray) -> np.ndarray:
            y_arr = bkd.reshape(bkd.asarray(y_np), (1, -1))
            Phi = basis.evaluate(y_arr)
            f = bkd.dot(bkd.reshape(d, (1, -1)), Phi)[0]
            return bkd.to_numpy(f)

        integral = composite_gauss_legendre(
            integrand, y_min, y_max, n_intervals=500, n_points=5
        )
        bkd.assert_allclose(
            bkd.asarray([integral]),
            bkd.asarray([1.0]),
            atol=0.01,
        )

    def test_ise_criterion_positive(self) -> None:
        """ISE criterion b^T M^{-1} b should be positive."""
        bkd = self._bkd
        basis = PiecewiseDensityBasis(-5.0, 5.0, 21, degree=1, bkd=bkd)
        fitter = ProjectionDensityFitter(basis)
        y_values, weights = self._make_gaussian_quadrature(0.0, 1.0, 50)
        ise = fitter.ise_criterion(y_values, weights)
        assert float(bkd.to_numpy(ise)) > 0.0

    def test_ise_criterion_kernel(self) -> None:
        """ISE criterion should be positive for kernel basis."""
        bkd = self._bkd
        kernel = SquaredExponentialKernel(
            bkd.asarray([0.5]),
            (0.01, 100.0),
            1,
            bkd,
        )
        centers = bkd.reshape(bkd.linspace(-3.0, 3.0, 15), (1, -1))
        kb = KernelBasis(kernel, centers)
        basis = KernelDensityBasis(kb)
        fitter = ProjectionDensityFitter(basis)
        y_values, weights = self._make_gaussian_quadrature(0.0, 1.0, 50)
        ise = fitter.ise_criterion(y_values, weights)
        assert float(bkd.to_numpy(ise)) > 0.0

    def test_ise_equals_dtMd(self) -> None:
        """ISE criterion b^T M^{-1} b should equal d^T M d."""
        bkd = self._bkd
        basis = PiecewiseDensityBasis(-5.0, 5.0, 21, degree=1, bkd=bkd)
        fitter = ProjectionDensityFitter(basis)
        y_values, weights = self._make_gaussian_quadrature(0.0, 1.0, 50)

        d = fitter.fit(y_values, weights)
        ise = fitter.ise_criterion(y_values, weights)

        # d^T M d = d^T b = b^T M^{-1} b (since M d = b)
        M = basis.mass_matrix()
        dtMd = bkd.sum(d * bkd.dot(M, bkd.reshape(d, (-1, 1)))[:, 0])
        bkd.assert_allclose(
            bkd.reshape(ise, (1,)),
            bkd.reshape(dtMd, (1,)),
            rtol=1e-8,
        )

    def test_invalid_basis_type(self) -> None:
        """Should reject non-DensityBasisProtocol objects."""
        with self.assertRaises(TypeError):
            ProjectionDensityFitter("not a basis")  # type: ignore

    def test_uniform_density_piecewise(self) -> None:
        """Project uniform density on [-1, 1]."""
        bkd = self._bkd

        # Uniform quadrature: use Gauss-Legendre on [-1, 1]
        nquad = 100
        xi_np, w_np = np.polynomial.legendre.leggauss(nquad)
        # Gauss-Legendre integrates over [-1,1] with weight 1
        # For uniform on [-1,1], density = 0.5, so weights = w / 2
        y_values = bkd.reshape(bkd.asarray(xi_np), (1, -1))
        weights = bkd.asarray(w_np / 2.0)

        # Set domain from data range
        y_np = bkd.to_numpy(y_values[0])
        y_min, y_max = float(y_np.min()), float(y_np.max())
        basis = PiecewiseDensityBasis(y_min, y_max, 21, degree=1, bkd=bkd)
        fitter = ProjectionDensityFitter(basis)
        d = fitter.fit(y_values, weights)

        # Evaluate at test points well inside [-1, 1]
        y_test = bkd.reshape(bkd.linspace(-0.8, 0.8, 30), (1, -1))
        Phi = basis.evaluate(y_test)
        f_approx = bkd.dot(bkd.reshape(d, (1, -1)), Phi)[0]

        f_true = bkd.full((30,), 0.5)
        bkd.assert_allclose(f_approx, f_true, atol=0.05)


class TestProjectionDensityFitterNumpy(TestProjectionDensityFitter[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestProjectionDensityFitterTorch(TestProjectionDensityFitter[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestISEOptimizingFitter(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _gaussian_quadrature(self, nquad: int = 80):
        """Gauss-Hermite quadrature for N(0,1)."""
        bkd = self._bkd
        xi_np, w_np = np.polynomial.hermite_e.hermegauss(nquad)
        w_np = w_np / math.sqrt(2.0 * math.pi)
        y_values = bkd.reshape(bkd.asarray(xi_np), (1, -1))
        weights = bkd.asarray(w_np)
        return y_values, weights

    def _make_kernel_basis(
        self,
        lenscale: float = 0.1,
        ncenters: int = 20,
        y_min: float = -4.0,
        y_max: float = 4.0,
    ) -> KernelDensityBasis:
        bkd = self._bkd
        # Upper bound: average spacing between centers
        domain_len = y_max - y_min
        max_lenscale = domain_len / ncenters
        kernel = SquaredExponentialKernel(
            bkd.asarray([lenscale]),
            (0.05, max_lenscale),
            1,
            bkd,
        )
        centers = bkd.reshape(bkd.linspace(y_min, y_max, ncenters), (1, -1))
        kb = KernelBasis(kernel, centers)
        return KernelDensityBasis(kb)

    def test_ise_optimization(self) -> None:
        """Optimal ISE score should be higher than non-optimal starting point."""
        bkd = self._bkd
        y_values, weights = self._gaussian_quadrature()

        # Start from a clearly non-optimal length scale
        basis = self._make_kernel_basis(lenscale=0.1)
        fitter_proj = ProjectionDensityFitter(basis)
        ise_before = float(bkd.to_numpy(fitter_proj.ise_criterion(y_values, weights)))

        ise_fitter = ISEOptimizingFitter(basis)
        ise_fitter.fit(y_values, weights)

        ise_after = float(bkd.to_numpy(fitter_proj.ise_criterion(y_values, weights)))

        self.assertGreater(
            ise_after,
            ise_before,
            f"ISE did not improve: {ise_after:.6f} <= {ise_before:.6f}",
        )

    def test_ise_vs_fixed_kernel(self) -> None:
        """ISE-optimized density should be closer to N(0,1) than fixed bad kernel."""
        bkd = self._bkd
        y_values, weights = self._gaussian_quadrature()

        from scipy import stats

        # Fixed bad kernel
        basis_fixed = self._make_kernel_basis(lenscale=0.1)
        fitter_fixed = ProjectionDensityFitter(basis_fixed)
        d_fixed = fitter_fixed.fit(y_values, weights)

        # Optimized kernel
        basis_opt = self._make_kernel_basis(lenscale=0.1)
        ise_fitter = ISEOptimizingFitter(basis_opt)
        d_opt = ise_fitter.fit(y_values, weights)

        # Compare L2 error against N(0,1) on test grid
        y_test = bkd.reshape(bkd.linspace(-3.0, 3.0, 100), (1, -1))
        y_test_np = bkd.to_numpy(y_test[0])
        f_true_np = stats.norm.pdf(y_test_np)

        Phi_fixed = basis_fixed.evaluate(y_test)
        f_fixed_np = bkd.to_numpy(bkd.dot(bkd.reshape(d_fixed, (1, -1)), Phi_fixed)[0])
        l2_fixed = float(np.sqrt(np.sum((f_fixed_np - f_true_np) ** 2)))

        Phi_opt = basis_opt.evaluate(y_test)
        f_opt_np = bkd.to_numpy(bkd.dot(bkd.reshape(d_opt, (1, -1)), Phi_opt)[0])
        l2_opt = float(np.sqrt(np.sum((f_opt_np - f_true_np) ** 2)))

        self.assertLess(
            l2_opt,
            l2_fixed,
            f"Optimized L2 ({l2_opt:.6f}) not less than fixed L2 ({l2_fixed:.6f})",
        )

    def test_ise_with_custom_optimizer(self) -> None:
        """ISEOptimizingFitter should accept a custom optimizer."""
        bkd = self._bkd
        y_values, weights = self._gaussian_quadrature()

        from pyapprox.optimization.minimize.scipy.trust_constr import (
            ScipyTrustConstrOptimizer,
        )

        optimizer = ScipyTrustConstrOptimizer(verbosity=0, maxiter=100)

        basis = self._make_kernel_basis(lenscale=0.1)
        ise_fitter = ISEOptimizingFitter(basis, optimizer=optimizer)
        d = ise_fitter.fit(y_values, weights)

        bkd.assert_allclose(
            bkd.asarray([d.shape[0]]),
            bkd.asarray([20]),
        )

    def test_ise_invalid_basis_type(self) -> None:
        """Should reject non-KernelDensityBasis objects."""
        bkd = self._bkd
        basis = PiecewiseDensityBasis(-5.0, 5.0, 21, degree=1, bkd=bkd)
        with self.assertRaises(TypeError):
            ISEOptimizingFitter(basis)  # type: ignore


class TestISEOptimizingFitterNumpy(TestISEOptimizingFitter[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestISEOptimizingFitterTorch(TestISEOptimizingFitter[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
