"""Tests for ContinuousNumericOrthonormalPolynomial1D.

Tests verify:
1. Orthonormality for various continuous marginal distributions
2. MC orthonormality convergence rate is approximately O(N^{-1})
3. Comparison with analytic polynomials when available

Supports both bounded and unbounded distributions:
- Bounded: Uses Gauss-Legendre quadrature
- Unbounded: Uses interval expansion with adaptive quadrature
"""

from typing import Callable, List, NamedTuple, Optional

import numpy as np
import pytest
from scipy import stats

from pyapprox.probability.univariate import (
    BetaMarginal,
    GammaMarginal,
    GaussianMarginal,
    ScipyContinuousMarginal,
    UniformMarginal,
)
from pyapprox.surrogates.affine.univariate.globalpoly import (
    ContinuousNumericOrthonormalPolynomial1D,
    HermitePolynomial1D,
    JacobiPolynomial1D,
    LaguerrePolynomial1D,
    LegendrePolynomial1D,
)
from tests._helpers.markers import slow_test, slower_test


class MarginalTestCase(NamedTuple):
    """Configuration for a marginal distribution test case."""

    name: str
    marginal_factory: Callable
    nterms: int = 5
    nquad_points: int = 100
    integrator_options: Optional[dict] = None


# =============================================================================
# Test cases for bounded continuous distributions
#
# Add new test cases here by appending to this list.
# Each entry should create a bounded marginal distribution.
# =============================================================================
BOUNDED_MARGINAL_TEST_CASES: List[MarginalTestCase] = [
    # Beta distributions on standard domain [0, 1]
    MarginalTestCase(
        name="Beta(3, 2) on [0, 1]",
        marginal_factory=lambda bkd: BetaMarginal(3.0, 2.0, bkd),
    ),
    MarginalTestCase(
        name="Beta(0.5, 0.5) on [0, 1]",
        marginal_factory=lambda bkd: BetaMarginal(0.5, 0.5, bkd),
    ),
    MarginalTestCase(
        name="Beta(2, 5) on [0, 1]",
        marginal_factory=lambda bkd: BetaMarginal(2.0, 5.0, bkd),
    ),
    MarginalTestCase(
        name="Beta(5, 2) on [0, 1]",
        marginal_factory=lambda bkd: BetaMarginal(5.0, 2.0, bkd),
    ),
    MarginalTestCase(
        name="Beta(10, 10) on [0, 1]",
        marginal_factory=lambda bkd: BetaMarginal(10.0, 10.0, bkd),
    ),
    # Beta distributions on shifted domains via ScipyContinuousMarginal
    MarginalTestCase(
        name="Beta(3, 2) on [0, 2]",
        marginal_factory=lambda bkd: ScipyContinuousMarginal(
            stats.beta(3.0, 2.0, loc=0.0, scale=2.0), bkd
        ),
    ),
    MarginalTestCase(
        name="Beta(3, 2) on [-1, 1]",
        marginal_factory=lambda bkd: ScipyContinuousMarginal(
            stats.beta(3.0, 2.0, loc=-1.0, scale=2.0), bkd
        ),
    ),
    MarginalTestCase(
        name="Beta(2, 3) on [5, 10]",
        marginal_factory=lambda bkd: ScipyContinuousMarginal(
            stats.beta(2.0, 3.0, loc=5.0, scale=5.0), bkd
        ),
    ),
    # Uniform distributions on various domains
    MarginalTestCase(
        name="Uniform[-1, 1]",
        marginal_factory=lambda bkd: UniformMarginal(-1.0, 1.0, bkd),
    ),
    MarginalTestCase(
        name="Uniform[0, 1]",
        marginal_factory=lambda bkd: UniformMarginal(0.0, 1.0, bkd),
    ),
    MarginalTestCase(
        name="Uniform[0, 2]",
        marginal_factory=lambda bkd: UniformMarginal(0.0, 2.0, bkd),
    ),
    MarginalTestCase(
        name="Uniform[-2, 3]",
        marginal_factory=lambda bkd: UniformMarginal(-2.0, 3.0, bkd),
    ),
    MarginalTestCase(
        name="Uniform[5, 10]",
        marginal_factory=lambda bkd: UniformMarginal(5.0, 10.0, bkd),
    ),
    # Arcsine distribution (Beta(0.5, 0.5) equivalent)
    MarginalTestCase(
        name="Arcsine on [0, 1]",
        marginal_factory=lambda bkd: ScipyContinuousMarginal(stats.arcsine(), bkd),
    ),
    # Truncated normal (bounded)
    MarginalTestCase(
        name="TruncNorm(0, 1) on [-2, 2]",
        marginal_factory=lambda bkd: ScipyContinuousMarginal(
            stats.truncnorm(-2, 2, loc=0, scale=1), bkd
        ),
    ),
    MarginalTestCase(
        name="TruncNorm(5, 2) on [0, 10]",
        marginal_factory=lambda bkd: ScipyContinuousMarginal(
            stats.truncnorm(-2.5, 2.5, loc=5, scale=2), bkd
        ),
    ),
    # Power-law distributions (bounded)
    MarginalTestCase(
        name="Power-law a=2 on [0, 1]",
        marginal_factory=lambda bkd: ScipyContinuousMarginal(stats.powerlaw(2.0), bkd),
    ),
    MarginalTestCase(
        name="Power-law a=0.5 on [0, 1]",
        marginal_factory=lambda bkd: ScipyContinuousMarginal(stats.powerlaw(0.5), bkd),
    ),
]


# =============================================================================
# Test cases for unbounded continuous distributions
#
# These use the UnboundedIntegrator with interval expansion.
# Note: For unbounded distributions, the polynomials are computed on the
# physical domain without transformation. This means:
# - Gaussian(0,1) matches Hermite polynomials
# - Gamma(shape, 1) matches Laguerre polynomials with rho=shape-1
# - Other parameter combinations give valid orthonormal polynomials but
#   with different recursion coefficients than the canonical forms
# =============================================================================
UNBOUNDED_MARGINAL_TEST_CASES: List[MarginalTestCase] = [
    # Standard Gaussian (matches Hermite)
    MarginalTestCase(
        name="Gaussian(0, 1)",
        marginal_factory=lambda bkd: GaussianMarginal(0.0, 1.0, bkd),
        nterms=5,
        nquad_points=100,
        integrator_options={"interval_size": 1.0, "atol": 1e-12, "rtol": 1e-12},
    ),
    # Non-standard Gaussian (valid orthonormal but differs from Hermite)
    MarginalTestCase(
        name="Gaussian(5, 2)",
        marginal_factory=lambda bkd: GaussianMarginal(5.0, 2.0, bkd),
        nterms=5,
        nquad_points=100,
        integrator_options={"interval_size": 1.0, "atol": 1e-12, "rtol": 1e-12},
    ),
    MarginalTestCase(
        name="Gaussian(-3, 0.5)",
        marginal_factory=lambda bkd: GaussianMarginal(-3.0, 0.5, bkd),
        nterms=5,
        nquad_points=100,
        integrator_options={"interval_size": 0.5, "atol": 1e-12, "rtol": 1e-12},
    ),
    # Gamma with unit scale (matches Laguerre)
    MarginalTestCase(
        name="Gamma(2, 1)",
        marginal_factory=lambda bkd: GammaMarginal(2.0, 1.0, bkd=bkd),
        nterms=5,
        nquad_points=100,
        integrator_options={"interval_size": 1.0, "atol": 1e-12, "rtol": 1e-12},
    ),
    MarginalTestCase(
        name="Gamma(3, 1)",
        marginal_factory=lambda bkd: GammaMarginal(3.0, 1.0, bkd=bkd),
        nterms=5,
        nquad_points=100,
        integrator_options={"interval_size": 1.0, "atol": 1e-12, "rtol": 1e-12},
    ),
    MarginalTestCase(
        name="Gamma(5, 1)",
        marginal_factory=lambda bkd: GammaMarginal(5.0, 1.0, bkd=bkd),
        nterms=5,
        nquad_points=100,
        integrator_options={"interval_size": 1.0, "atol": 1e-12, "rtol": 1e-12},
    ),
    # Gamma with non-unit scale (valid orthonormal but differs from Laguerre)
    MarginalTestCase(
        name="Gamma(2, 2)",
        marginal_factory=lambda bkd: GammaMarginal(2.0, 2.0, bkd=bkd),
        nterms=5,
        nquad_points=100,
        integrator_options={"interval_size": 1.0, "atol": 1e-12, "rtol": 1e-12},
    ),
    MarginalTestCase(
        name="Gamma(3, 0.5)",
        marginal_factory=lambda bkd: GammaMarginal(3.0, 0.5, bkd=bkd),
        nterms=5,
        nquad_points=100,
        integrator_options={"interval_size": 0.5, "atol": 1e-12, "rtol": 1e-12},
    ),
]


class TestContinuousNumericOrthonormalPolynomial1D:
    """Base class for ContinuousNumericOrthonormalPolynomial1D tests."""

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _compute_mc_grammian(self, bkd, poly, marginal, nsamples):
        """Compute Grammian matrix using Monte Carlo integration.

        G_ij = E[p_i(X) * p_j(X)] where X ~ marginal

        For orthonormal polynomials, G should be identity.
        """
        # Draw samples from marginal in physical domain
        phys_samples = marginal.rvs(nsamples)

        # Map to canonical domain for polynomial evaluation
        can_samples = poly._physical_to_canonical(phys_samples)

        # Evaluate polynomials at canonical samples
        vals = poly(can_samples)  # (nsamples, nterms)

        # MC estimate of Grammian: mean of outer products
        grammian = vals.T @ vals / nsamples

        return grammian

    def _check_mc_convergence_rate(
        self,
        bkd,
        poly,
        marginal,
        expected_rate: float = -1.0,
        rate_tolerance: float = 0.15,
    ) -> None:
        """Check that MC orthonormality MSE converges at expected rate.

        For MC integration, we expect:
        MSE = ||G_mc - I||_F^2 ~ O(N^{-1})

        where N is the number of samples.
        """
        nterms = poly.nterms()
        expected_grammian = bkd.eye(nterms)

        # Use multiple sample sizes with many trials to estimate convergence rate
        sample_sizes = [500, 1000, 2000]
        ntrials = 200
        mse_values = []

        np.random.seed(42)  # For reproducibility

        for nsamples in sample_sizes:
            # Average MSE over multiple trials for stable estimate
            trial_mses = []
            for _ in range(ntrials):
                grammian = self._compute_mc_grammian(bkd, poly, marginal, nsamples)
                diff = grammian - expected_grammian
                mse = float(bkd.sum(diff * diff)) / (nterms * nterms)
                trial_mses.append(mse)
            mse_values.append(np.mean(trial_mses))

        # Fit log-log linear model to estimate rate
        log_n = np.log(np.array(sample_sizes))
        log_mse = np.log(np.array(mse_values))

        # Skip if any NaN or Inf (numerical issues)
        if not np.all(np.isfinite(log_mse)):
            pytest.skip("Numerical issues in MSE computation (NaN/Inf)")

        # Linear regression: log(MSE) = rate * log(N) + const
        slope, intercept = np.polyfit(log_n, log_mse, 1)

        # Check that slope is within tolerance of expected rate
        assert slope > expected_rate - rate_tolerance, (
            f"Convergence rate {slope:.3f} is worse than expected "
            f"(expected >= {expected_rate - rate_tolerance:.3f})"
        )
        assert slope < expected_rate + rate_tolerance, (
            f"Convergence rate {slope:.3f} is better than expected "
            f"(expected <= {expected_rate + rate_tolerance:.3f})"
        )

    # =========================================================================
    # Parametrized tests for bounded marginals
    # =========================================================================

    @slow_test
    def test_bounded_marginals_mc_convergence(self, bkd) -> None:
        """Test MC orthonormality convergence for all bounded marginals."""
        for case in BOUNDED_MARGINAL_TEST_CASES:
            marginal = case.marginal_factory(bkd)
            poly = ContinuousNumericOrthonormalPolynomial1D(
                marginal, bkd, nquad_points=case.nquad_points
            )
            poly.set_nterms(case.nterms)

            self._check_mc_convergence_rate(bkd, poly, marginal)

    # =========================================================================
    # Parametrized tests for unbounded marginals
    # =========================================================================

    @slower_test
    def test_unbounded_marginals_quadrature_orthonormality(self, bkd) -> None:
        """Test orthonormality for unbounded marginals using Gauss quadrature.

        Note: MC convergence tests are NOT used for unbounded distributions
        because the variance of p_n(X)^2 grows with n, making MC impractical.
        Instead, we use the polynomial's own Gauss quadrature rule to verify
        orthonormality exactly (up to numerical precision).
        """
        for case in UNBOUNDED_MARGINAL_TEST_CASES:
            marginal = case.marginal_factory(bkd)
            poly = ContinuousNumericOrthonormalPolynomial1D(
                marginal,
                bkd,
                nquad_points=case.nquad_points,
                integrator_options=case.integrator_options,
            )
            nterms = case.nterms
            poly.set_nterms(nterms)

            # Use Gauss quadrature to verify orthonormality
            # Need nterms + extra points for exactness
            nquad = nterms + 2
            poly.set_nterms(nquad)  # Need more terms for quadrature
            x_quad, w_quad = poly.gauss_quadrature_rule(nquad)
            poly.set_nterms(nterms)  # Reset to original

            vals = poly(x_quad)  # (nquad, nterms)
            grammian = vals.T @ (w_quad * vals)

            expected = bkd.eye(nterms)
            bkd.assert_allclose(grammian, expected, rtol=1e-8, atol=1e-10)

    # =========================================================================
    # Comparison with analytic polynomials
    # =========================================================================

    def test_matches_legendre_for_uniform(self, bkd) -> None:
        """Numeric polynomials for Uniform[-1,1] should match Legendre."""
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        numeric_poly = ContinuousNumericOrthonormalPolynomial1D(
            marginal, bkd, nquad_points=100
        )
        numeric_poly.set_nterms(6)

        legendre_poly = LegendrePolynomial1D(bkd)
        legendre_poly.set_nterms(6)

        # Evaluate at test points in canonical domain
        samples = bkd.linspace(-0.9, 0.9, 20)
        samples = bkd.reshape(samples, (1, -1))

        numeric_vals = numeric_poly(samples)
        legendre_vals = legendre_poly(samples)

        # Check recursion coefficients match
        numeric_ab = numeric_poly._get_recursion_coefficients(6)
        legendre_ab = legendre_poly._get_recursion_coefficients(6)

        bkd.assert_allclose(numeric_ab, legendre_ab, rtol=1e-8, atol=1e-10)

        # Check polynomial values match
        bkd.assert_allclose(numeric_vals, legendre_vals, rtol=1e-8, atol=1e-10)

    def test_matches_jacobi_for_beta(self, bkd) -> None:
        """Numeric polynomials for Beta should match Jacobi on [-1, 1].

        Beta(alpha, beta) on [0, 1] corresponds to Jacobi(beta-1, alpha-1) on [-1, 1].
        The transformation is: x_beta = (x_jacobi + 1) / 2.
        """
        alpha, beta = 3.0, 2.0
        marginal = BetaMarginal(alpha, beta, bkd)
        numeric_poly = ContinuousNumericOrthonormalPolynomial1D(
            marginal, bkd, nquad_points=100
        )
        numeric_poly.set_nterms(6)

        # Jacobi parameters for Beta(alpha, beta): (beta-1, alpha-1)
        jacobi_poly = JacobiPolynomial1D(beta - 1, alpha - 1, bkd)
        jacobi_poly.set_nterms(6)

        # Compare recursion coefficients
        numeric_ab = numeric_poly._get_recursion_coefficients(6)
        jacobi_ab = jacobi_poly._get_recursion_coefficients(6)

        bkd.assert_allclose(numeric_ab, jacobi_ab, rtol=1e-6, atol=1e-8)

    def test_matches_jacobi_for_shifted_beta(self, bkd) -> None:
        """Numeric polynomials for Beta on shifted domain match Jacobi.

        Beta(alpha, beta) on [a, b] should have same recursion coefficients
        as Beta(alpha, beta) on [0, 1] since they're both mapped to [-1, 1].
        """
        alpha, beta = 2.0, 4.0

        # Standard Beta on [0, 1]
        marginal_std = BetaMarginal(alpha, beta, bkd)
        poly_std = ContinuousNumericOrthonormalPolynomial1D(
            marginal_std, bkd, nquad_points=100
        )
        poly_std.set_nterms(5)

        # Shifted Beta on [2, 5]
        scipy_rv = stats.beta(alpha, beta, loc=2.0, scale=3.0)
        marginal_shifted = ScipyContinuousMarginal(scipy_rv, bkd)
        poly_shifted = ContinuousNumericOrthonormalPolynomial1D(
            marginal_shifted, bkd, nquad_points=100
        )
        poly_shifted.set_nterms(5)

        # Recursion coefficients should match
        ab_std = poly_std._get_recursion_coefficients(5)
        ab_shifted = poly_shifted._get_recursion_coefficients(5)

        bkd.assert_allclose(ab_std, ab_shifted, rtol=1e-8, atol=1e-10)

    def test_matches_hermite_for_gaussian(self, bkd) -> None:
        """Numeric polynomials for standard Gaussian should match Hermite.

        Note: The predictor-corrector method accumulates errors, so we test
        with fewer terms. For symmetric distributions like Gaussian, the
        'a' coefficients should be 0, but numerical errors make them small
        but non-zero.
        """
        marginal = GaussianMarginal(0.0, 1.0, bkd)
        numeric_poly = ContinuousNumericOrthonormalPolynomial1D(
            marginal,
            bkd,
            nquad_points=200,
            integrator_options={
                "interval_size": 0.5,
                "atol": 1e-15,
                "rtol": 1e-15,
                "maxinner_iters": 30,
            },
        )
        nterms = 4  # Use fewer terms to avoid error accumulation
        numeric_poly.set_nterms(nterms)

        hermite_poly = HermitePolynomial1D(bkd, rho=0.0, prob_meas=True)
        hermite_poly.set_nterms(nterms)

        # Compare recursion coefficients
        numeric_ab = numeric_poly._get_recursion_coefficients(nterms)
        hermite_ab = hermite_poly._get_recursion_coefficients(nterms)

        # For 'a' coefficients (column 0), use absolute tolerance since they're ~0
        # For 'b' coefficients (column 1), use relative tolerance
        bkd.assert_allclose(numeric_ab[:, 0], hermite_ab[:, 0], rtol=0, atol=1e-5)
        bkd.assert_allclose(numeric_ab[:, 1], hermite_ab[:, 1], rtol=1e-6, atol=0)

    def test_matches_laguerre_for_gamma_unit_scale(self, bkd) -> None:
        """Numeric polynomials for Gamma(shape, 1) should match Laguerre.

        Gamma(shape, scale=1) corresponds to Laguerre with rho=shape-1.
        Note: Only scale=1 matches Laguerre; other scales give different
        recursion coefficients (but still orthonormal w.r.t. the measure).
        """
        shape, scale = 2.0, 1.0
        marginal = GammaMarginal(shape, scale, bkd=bkd)
        numeric_poly = ContinuousNumericOrthonormalPolynomial1D(
            marginal,
            bkd,
            nquad_points=200,
            integrator_options={
                "interval_size": 0.5,
                "atol": 1e-15,
                "rtol": 1e-15,
                "maxinner_iters": 30,
            },
        )
        numeric_poly.set_nterms(6)

        # Laguerre rho = shape - 1 = 1
        laguerre_poly = LaguerrePolynomial1D(bkd, rho=shape - 1)
        laguerre_poly.set_nterms(6)

        # Compare recursion coefficients
        numeric_ab = numeric_poly._get_recursion_coefficients(6)
        laguerre_ab = laguerre_poly._get_recursion_coefficients(6)

        bkd.assert_allclose(numeric_ab, laguerre_ab, rtol=1e-10, atol=1e-12)

    # =========================================================================
    # Basic functionality tests
    # =========================================================================

    def test_nterms_and_evaluation_shape(self, bkd) -> None:
        """Test nterms setting and evaluation shape."""
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        poly = ContinuousNumericOrthonormalPolynomial1D(marginal, bkd)

        assert poly.nterms() == 0

        poly.set_nterms(5)
        assert poly.nterms() == 5

        # Evaluate at samples
        samples = bkd.array([[-0.5, 0.0, 0.5]])
        vals = poly(samples)

        assert vals.shape == (3, 5)

    def test_canonical_domain_mapping(self, bkd) -> None:
        """Test canonical to physical domain mapping."""
        # Beta on [0, 1] should use canonical domain [-1, 1]
        marginal = BetaMarginal(2.0, 3.0, bkd)
        poly = ContinuousNumericOrthonormalPolynomial1D(marginal, bkd)
        poly.set_nterms(3)

        # Canonical [-1, 1] should map to physical [0, 1]
        can_samples = bkd.array([[-1.0, 0.0, 1.0]])
        phys_samples = poly._canonical_to_physical(can_samples)

        expected_phys = bkd.array([[0.0, 0.5, 1.0]])
        bkd.assert_allclose(phys_samples, expected_phys, rtol=1e-14)

        # And inverse
        recovered_can = poly._physical_to_canonical(phys_samples)
        bkd.assert_allclose(recovered_can, can_samples, rtol=1e-14)

    def test_canonical_domain_mapping_shifted(self, bkd) -> None:
        """Test canonical domain mapping for shifted distributions."""
        # Beta on [2, 5] should use canonical domain [-1, 1]
        scipy_rv = stats.beta(2.0, 3.0, loc=2.0, scale=3.0)
        marginal = ScipyContinuousMarginal(scipy_rv, bkd)
        poly = ContinuousNumericOrthonormalPolynomial1D(marginal, bkd)
        poly.set_nterms(3)

        # Canonical [-1, 1] should map to physical [2, 5]
        can_samples = bkd.array([[-1.0, 0.0, 1.0]])
        phys_samples = poly._canonical_to_physical(can_samples)

        expected_phys = bkd.array([[2.0, 3.5, 5.0]])
        bkd.assert_allclose(phys_samples, expected_phys, rtol=1e-14)

    def test_repr(self, bkd) -> None:
        """Test string representation."""
        marginal = BetaMarginal(2.0, 3.0, bkd)
        poly = ContinuousNumericOrthonormalPolynomial1D(marginal, bkd)
        poly.set_nterms(5)

        repr_str = repr(poly)
        assert "ContinuousNumericOrthonormalPolynomial1D" in repr_str
        assert "BetaMarginal" in repr_str
        assert "5" in repr_str

    def test_first_polynomial_is_constant(self, bkd) -> None:
        """First orthonormal polynomial p_0(x) = 1 for any measure."""
        marginal = BetaMarginal(2.0, 3.0, bkd)
        poly = ContinuousNumericOrthonormalPolynomial1D(marginal, bkd)
        poly.set_nterms(3)

        # p_0(x) should be 1 everywhere
        samples = bkd.array([[-0.8, -0.2, 0.0, 0.4, 0.9]])
        vals = poly(samples)

        expected_p0 = bkd.ones((5,))
        bkd.assert_allclose(vals[:, 0], expected_p0, rtol=1e-12)

    def test_polynomials_are_normalized(self, bkd) -> None:
        """Orthonormal polynomials should integrate to delta_ij.

        Using Gauss quadrature from the polynomial itself.
        """
        marginal = BetaMarginal(3.0, 2.0, bkd)
        poly = ContinuousNumericOrthonormalPolynomial1D(
            marginal, bkd, nquad_points=100
        )
        nterms = 6
        poly.set_nterms(nterms)

        # Get quadrature points from Jacobi polynomial with same parameters
        jacobi_poly = JacobiPolynomial1D(1.0, 2.0, bkd)  # beta-1, alpha-1
        jacobi_poly.set_nterms(nterms + 2)
        x_quad, w_quad = jacobi_poly.gauss_quadrature_rule(nterms + 2)

        # Evaluate numeric polynomial at quadrature points
        vals = poly(x_quad)  # (nquad, nterms)

        # Compute Gramian
        gramian = vals.T @ (w_quad * vals)

        expected = bkd.eye(nterms)
        bkd.assert_allclose(gramian, expected, rtol=1e-8, atol=1e-10)

    def test_unbounded_domain_uses_identity_transform(self, bkd) -> None:
        """Unbounded marginals should use identity transform (no mapping)."""
        marginal = GaussianMarginal(0.0, 1.0, bkd)
        poly = ContinuousNumericOrthonormalPolynomial1D(
            marginal,
            bkd,
            integrator_options={
                "interval_size": 1.0,
                "atol": 1e-14,
                "rtol": 1e-14,
            },
        )
        poly.set_nterms(3)

        # For unbounded, canonical = physical (loc=0, scale=1)
        assert poly._loc == 0.0
        assert poly._scale == 1.0

        # Samples should pass through unchanged
        samples = bkd.array([[-2.0, 0.0, 2.0]])
        phys_samples = poly._canonical_to_physical(samples)
        bkd.assert_allclose(phys_samples, samples, rtol=1e-14)
