"""Tests for UnivariatePCEDensity: exact PDF of 1D PCE output.

Tests include:
- Monomial conversion correctness
- Linear PCE with Gaussian (analytical Gaussian PDF reference)
- Linear PCE with Uniform (analytical uniform PDF reference)
- Quadratic PCE with chi-squared reference
- Moment computation
- Convergence tests with g(x) = x*cos(x) for both Gaussian and Uniform
"""

import math

import numpy as np
import pytest
from scipy import stats

from pyapprox.probability import GaussianMarginal, UniformMarginal
from pyapprox.surrogates.affine.expansions import (
    create_pce_from_marginals,
)
from pyapprox.surrogates.affine.expansions.pce_density import (
    UnivariatePCEDensity,
)
from pyapprox.surrogates.affine.univariate.globalpoly import (
    HermitePolynomial1D,
    LegendrePolynomial1D,
)
from pyapprox.surrogates.affine.univariate.globalpoly.monomial_conversion import (
    convert_orthonormal_to_monomials_1d,
)
from pyapprox.util.test_utils import slow_test


class TestMonomialConversion:
    """Test conversion from orthonormal basis to monomial coefficients."""

    def test_hermite_conversion(self, bkd):
        """Verify Hermite-to-monomial conversion by pointwise evaluation."""
        nterms = 6
        poly = HermitePolynomial1D(bkd)
        poly.set_nterms(nterms)
        rcoefs = poly.recursion_coefficients()

        M = convert_orthonormal_to_monomials_1d(rcoefs, bkd)

        # Evaluate at random points
        np.random.seed(42)
        x_np = np.random.randn(20)
        x_2d = bkd.reshape(bkd.asarray(x_np), (1, -1))

        # Evaluate via three-term recurrence
        basis_vals = poly(x_2d)  # (nsamples, nterms)

        # Evaluate via monomial form
        for k in range(nterms):
            mono_k = bkd.to_numpy(M[k, :])
            vals_mono = np.polynomial.polynomial.polyval(x_np, mono_k)
            vals_recur = bkd.to_numpy(basis_vals[:, k])
            bkd.assert_allclose(
                bkd.asarray(vals_mono),
                bkd.asarray(vals_recur),
                rtol=1e-12,
            )

    def test_legendre_conversion(self, bkd):
        """Verify Legendre-to-monomial conversion by pointwise evaluation."""
        nterms = 6
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(nterms)
        rcoefs = poly.recursion_coefficients()

        M = convert_orthonormal_to_monomials_1d(rcoefs, bkd)

        np.random.seed(42)
        x_np = np.random.uniform(-1, 1, 20)
        x_2d = bkd.reshape(bkd.asarray(x_np), (1, -1))

        basis_vals = poly(x_2d)

        for k in range(nterms):
            mono_k = bkd.to_numpy(M[k, :])
            vals_mono = np.polynomial.polynomial.polyval(x_np, mono_k)
            vals_recur = bkd.to_numpy(basis_vals[:, k])
            bkd.assert_allclose(
                bkd.asarray(vals_mono),
                bkd.asarray(vals_recur),
                rtol=1e-12,
            )


class TestLinearPCEDensity:
    """Test PDF of linear PCE Y = a + b*xi against analytical references."""

    def test_linear_gaussian(self, bkd):
        """Y = 2 + 3*xi, xi ~ N(0,1). f_Y(y) = phi((y-2)/3) / 3."""
        a, b = 2.0, 3.0

        marginal = GaussianMarginal(0.0, 1.0, bkd)
        pce = create_pce_from_marginals([marginal], max_level=1, bkd=bkd)

        # Set coefficients: c_0 * psi_0 + c_1 * psi_1
        # For standard normal Hermite basis: psi_0 = 1, psi_1 = x
        # So Y = c_0 + c_1 * xi
        coef = bkd.zeros((pce.nterms(), 1))
        coef[0, 0] = a  # constant term
        coef[1, 0] = b  # linear term
        pce.set_coefficients(coef)

        density = UnivariatePCEDensity(pce, marginal)

        # Evaluate at test points
        y_np = np.linspace(a - 4 * abs(b), a + 4 * abs(b), 50)
        y_vals = bkd.reshape(bkd.asarray(y_np), (1, -1))
        pdf_computed = density.pdf(y_vals)

        # Analytical: f_Y(y) = phi((y-a)/b) / |b|
        pdf_expected = stats.norm.pdf(y_np, loc=a, scale=abs(b))

        bkd.assert_allclose(
            pdf_computed,
            bkd.reshape(bkd.asarray(pdf_expected), (1, -1)),
            rtol=1e-12,
        )

    def test_linear_uniform(self, bkd):
        """Y = a + b*xi, xi ~ Uniform(-1,1). f_Y(y) = 1/(2|b|) on support."""
        a, b = 1.0, 2.0

        marginal = UniformMarginal(-1.0, 1.0, bkd)
        pce = create_pce_from_marginals([marginal], max_level=1, bkd=bkd)

        # For orthonormal Legendre: psi_0=1, psi_1=sqrt(3)*x
        # To get Y = a + b*x: c_0 = a, c_1 = b / sqrt(3)
        coef = bkd.zeros((pce.nterms(), 1))
        coef[0, 0] = a
        coef[1, 0] = b / math.sqrt(3.0)
        pce.set_coefficients(coef)

        density = UnivariatePCEDensity(pce, marginal)

        # Evaluate at interior points (away from boundaries)
        y_np = np.linspace(a - abs(b) + 0.1, a + abs(b) - 0.1, 30)
        y_vals = bkd.reshape(bkd.asarray(y_np), (1, -1))
        pdf_computed = density.pdf(y_vals)

        # Analytical: f_Y(y) = 1 / (2 * |b|) = 1/4
        pdf_expected = np.full_like(y_np, 1.0 / (2.0 * abs(b)))

        bkd.assert_allclose(
            pdf_computed,
            bkd.reshape(bkd.asarray(pdf_expected), (1, -1)),
            rtol=1e-10,
        )


class TestQuadraticPCEDensity:
    """Test PDF of quadratic PCE against chi-squared reference."""

    def test_quadratic_chi_squared(self, bkd):
        """Y = psi_2(xi) for standard Hermite. Y+1 = xi^2 ~ chi2(1)."""
        marginal = GaussianMarginal(0.0, 1.0, bkd)
        pce = create_pce_from_marginals([marginal], max_level=2, bkd=bkd)

        # Set coefficient for 2nd basis function only (psi_2)
        # For probabilists' Hermite: psi_2(x) = (x^2 - 1) / sqrt(2)
        coef = bkd.zeros((pce.nterms(), 1))
        coef[2, 0] = 1.0
        pce.set_coefficients(coef)

        density = UnivariatePCEDensity(pce, marginal)

        # Evaluate at interior points (avoid singularity near y=-1/sqrt(2))
        # psi_2 min is -1/sqrt(2) at xi=0
        psi2_min = -1.0 / math.sqrt(2)
        y_np = np.linspace(psi2_min + 0.5, psi2_min + 8.0, 40)
        y_vals = bkd.reshape(bkd.asarray(y_np), (1, -1))
        pdf_computed = density.pdf(y_vals)

        # Reference: psi_2(x) = (x^2 - 1) / sqrt(2)
        # So y = (x^2 - 1) / sqrt(2), meaning x^2 = sqrt(2)*y + 1
        # f_Y(y) = sum_i phi(xi_i) / |psi_2'(xi_i)|
        # where psi_2'(x) = 2x / sqrt(2) = x * sqrt(2)
        # For y such that xi^2 = sqrt(2)*y + 1 > 0:
        # Two roots: xi = +/- sqrt(sqrt(2)*y + 1)
        # f_Y(y) = phi(xi_+) / |xi_+ * sqrt(2)| + phi(xi_-) / |xi_- * sqrt(2)|
        #        = 2 * phi(xi_+) / (xi_+ * sqrt(2))
        pdf_expected = np.zeros_like(y_np)
        for ii, y in enumerate(y_np):
            xi_sq = math.sqrt(2) * y + 1.0
            if xi_sq > 0:
                xi = math.sqrt(xi_sq)
                phi_val = stats.norm.pdf(xi)
                deriv_abs = xi * math.sqrt(2)
                pdf_expected[ii] = 2.0 * phi_val / deriv_abs

        bkd.assert_allclose(
            pdf_computed,
            bkd.reshape(bkd.asarray(pdf_expected), (1, -1)),
            rtol=1e-8,
        )


class TestMomentsExact:
    """Test exact moment computation via quadrature."""

    def test_moments_match_pce_statistics(self, bkd):
        """Verify moments_exact matches PCE mean and variance."""
        marginal = GaussianMarginal(0.0, 1.0, bkd)
        pce = create_pce_from_marginals([marginal], max_level=5, bkd=bkd)

        # Set some non-trivial coefficients
        np.random.seed(42)
        coef_np = np.random.randn(pce.nterms(), 1)
        pce.set_coefficients(bkd.asarray(coef_np))

        density = UnivariatePCEDensity(pce, marginal)
        moments = density.moments_exact(2)

        # E[Y] should match pce.mean()
        pce_mean = pce.mean()[0]
        bkd.assert_allclose(
            bkd.reshape(moments[0], (1,)),
            bkd.reshape(pce_mean, (1,)),
            rtol=1e-10,
        )

        # E[Y^2] - E[Y]^2 should match pce.variance()
        pce_var = pce.variance()[0]
        var_from_moments = moments[1] - moments[0] ** 2
        bkd.assert_allclose(
            bkd.reshape(var_from_moments, (1,)),
            bkd.reshape(pce_var, (1,)),
            rtol=1e-10,
        )


def _build_pce_for_function(func, marginal, max_level, bkd):
    """Helper: build PCE by spectral projection for a given function."""
    pce = create_pce_from_marginals([marginal], max_level=max_level, bkd=bkd)
    basis = pce.get_basis()

    n_quad = max(100, 3 * max_level)
    quad_pts, quad_wts = basis.univariate_quadrature(0, n_quad)
    quad_wts_flat = bkd.flatten(quad_wts)

    # Evaluate function at quadrature points
    pts_np = bkd.to_numpy(quad_pts)
    vals_np = func(pts_np[0])
    values = bkd.reshape(bkd.asarray(vals_np), (1, -1))

    pce.fit_via_projection(quad_pts, values, quad_wts_flat)
    return pce


class TestConvergenceGaussian:
    """Convergence test: g(x)=x*cos(x), xi ~ N(0,1), Hermite basis."""

    def _target_func(self, x):
        return x * np.cos(x)

    def test_moment_matching(self, bkd):
        """Verify moments from quadrature match PCE statistics at P=15."""
        P = 15

        marginal = GaussianMarginal(0.0, 1.0, bkd)
        pce = _build_pce_for_function(self._target_func, marginal, P, bkd)

        # Pre-check: PCE approximation accuracy
        np.random.seed(123)
        test_xi = np.random.randn(100)
        test_pts = bkd.reshape(bkd.asarray(test_xi), (1, -1))
        pce_vals = bkd.to_numpy(pce(test_pts)[0])
        true_vals = self._target_func(test_xi)
        max_approx_err = np.max(np.abs(pce_vals - true_vals))
        assert max_approx_err < 1e-5, f"PCE approximation error too large: {max_approx_err}"

        density = UnivariatePCEDensity(pce, marginal)

        # Exact moments via quadrature in xi-space must match PCE stats
        moments = density.moments_exact(2)
        # Mean of x*cos(x) for standard normal is ~0 (odd function),
        # so use atol for near-zero comparison
        bkd.assert_allclose(
            bkd.reshape(moments[0], (1,)),
            bkd.reshape(pce.mean()[0], (1,)),
            atol=1e-10,
        )
        var_from_moments = moments[1] - moments[0] ** 2
        bkd.assert_allclose(
            bkd.reshape(var_from_moments, (1,)),
            bkd.reshape(pce.variance()[0], (1,)),
            rtol=1e-10,
        )

    @slow_test
    def test_l1_convergence(self, bkd):
        """Verify PDF converges in L1 as PCE order increases."""
        marginal = GaussianMarginal(0.0, 1.0, bkd)

        # Evaluate successive PDFs on a common grid and check L1
        # convergence between consecutive orders
        orders = [10, 15, 20, 25]
        y_grid = np.linspace(-3.5, 3.5, 500)
        y_2d = bkd.reshape(bkd.asarray(y_grid), (1, -1))
        dy = (y_grid[-1] - y_grid[0]) / len(y_grid)

        prev_pdf = None
        l1_diffs = []
        for P in orders:
            pce_P = _build_pce_for_function(self._target_func, marginal, P, bkd)
            density_P = UnivariatePCEDensity(pce_P, marginal)
            pdf_P = bkd.to_numpy(density_P.pdf(y_2d)[0])
            if prev_pdf is not None:
                l1_diff = np.sum(np.abs(pdf_P - prev_pdf)) * dy
                l1_diffs.append(l1_diff)
            prev_pdf = pdf_P

        # L1 differences between consecutive orders should decrease
        # (skip check once both diffs are at machine precision)
        for ii in range(1, len(l1_diffs)):
            if l1_diffs[ii - 1] < 1e-10:
                continue
            assert l1_diffs[ii] < l1_diffs[ii - 1], (
                f"L1 diff not decreasing: "
                f"P={orders[ii]}->{orders[ii + 1]} ({l1_diffs[ii]:.6f}) >= "
                f"P={orders[ii - 1]}->{orders[ii]} ({l1_diffs[ii - 1]:.6f})"
            )
        # Final consecutive difference should be small
        assert l1_diffs[-1] < 0.01, f"Final L1 diff too large: {l1_diffs[-1]}"


class TestConvergenceUniform:
    """Convergence test: g(x)=x*cos(x), xi ~ Uniform(-1,1), Legendre basis."""

    def _target_func(self, x):
        return x * np.cos(x)

    def test_moment_matching(self, bkd):
        """Verify moments from quadrature match PCE statistics at P=15."""
        P = 15

        marginal = UniformMarginal(-1.0, 1.0, bkd)
        pce = _build_pce_for_function(self._target_func, marginal, P, bkd)

        # Pre-check: PCE approximation accuracy
        np.random.seed(123)
        test_xi = np.random.uniform(-1, 1, 100)
        test_pts = bkd.reshape(bkd.asarray(test_xi), (1, -1))
        pce_vals = bkd.to_numpy(pce(test_pts)[0])
        true_vals = self._target_func(test_xi)
        max_approx_err = np.max(np.abs(pce_vals - true_vals))
        assert max_approx_err < 1e-5, f"PCE approximation error too large: {max_approx_err}"

        density = UnivariatePCEDensity(pce, marginal)

        # Exact moments via quadrature in xi-space must match PCE stats
        moments = density.moments_exact(2)
        # Mean of x*cos(x) on [-1,1] is ~0 (odd function),
        # so use atol for near-zero comparison
        bkd.assert_allclose(
            bkd.reshape(moments[0], (1,)),
            bkd.reshape(pce.mean()[0], (1,)),
            atol=1e-10,
        )
        var_from_moments = moments[1] - moments[0] ** 2
        bkd.assert_allclose(
            bkd.reshape(var_from_moments, (1,)),
            bkd.reshape(pce.variance()[0], (1,)),
            rtol=1e-10,
        )

    @slow_test
    def test_l1_convergence(self, bkd):
        """Verify PDF converges in L1 as PCE order increases."""
        marginal = UniformMarginal(-1.0, 1.0, bkd)

        # Evaluate successive PDFs on a common grid and check L1
        # convergence between consecutive orders
        orders = [10, 15, 20, 25]
        y_grid = np.linspace(-0.6, 0.6, 500)
        y_2d = bkd.reshape(bkd.asarray(y_grid), (1, -1))
        dy = (y_grid[-1] - y_grid[0]) / len(y_grid)

        prev_pdf = None
        l1_diffs = []
        for P in orders:
            pce_P = _build_pce_for_function(self._target_func, marginal, P, bkd)
            density_P = UnivariatePCEDensity(pce_P, marginal)
            pdf_P = bkd.to_numpy(density_P.pdf(y_2d)[0])
            if prev_pdf is not None:
                l1_diff = np.sum(np.abs(pdf_P - prev_pdf)) * dy
                l1_diffs.append(l1_diff)
            prev_pdf = pdf_P

        # L1 differences between consecutive orders should decrease
        # (skip check once both diffs are at machine precision)
        for ii in range(1, len(l1_diffs)):
            if l1_diffs[ii - 1] < 1e-10:
                continue
            assert l1_diffs[ii] < l1_diffs[ii - 1], (
                f"L1 diff not decreasing: "
                f"P={orders[ii]}->{orders[ii + 1]} ({l1_diffs[ii]:.6f}) >= "
                f"P={orders[ii - 1]}->{orders[ii]} ({l1_diffs[ii - 1]:.6f})"
            )
        # Final consecutive difference should be small
        assert l1_diffs[-1] < 0.01, f"Final L1 diff too large: {l1_diffs[-1]}"
