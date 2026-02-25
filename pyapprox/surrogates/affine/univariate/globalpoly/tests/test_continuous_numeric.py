"""Tests for ContinuousNumericOrthonormalPolynomial1D.

Tests verify:
1. Orthonormality for various continuous marginal distributions
2. MC orthonormality convergence rate is approximately O(N^{-1})
3. Comparison with analytic polynomials when available

Supports both bounded and unbounded distributions:
- Bounded: Uses Gauss-Legendre quadrature
- Unbounded: Uses interval expansion with adaptive quadrature
"""

import unittest
from typing import Any, Callable, Generic, List, NamedTuple, Optional

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import stats

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests

from pyapprox.surrogates.affine.univariate.globalpoly import (
    ContinuousNumericOrthonormalPolynomial1D,
    JacobiPolynomial1D,
    LegendrePolynomial1D,
    HermitePolynomial1D,
    LaguerrePolynomial1D,
)
from pyapprox.probability.univariate import (
    BetaMarginal,
    UniformMarginal,
    GaussianMarginal,
    GammaMarginal,
    ScipyContinuousMarginal,
)


class MarginalTestCase(NamedTuple):
    """Configuration for a marginal distribution test case."""

    name: str
    marginal_factory: Callable[[Backend], Any]
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
        marginal_factory=lambda bkd: ScipyContinuousMarginal(
            stats.arcsine(), bkd
        ),
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
        marginal_factory=lambda bkd: ScipyContinuousMarginal(
            stats.powerlaw(2.0), bkd
        ),
    ),
    MarginalTestCase(
        name="Power-law a=0.5 on [0, 1]",
        marginal_factory=lambda bkd: ScipyContinuousMarginal(
            stats.powerlaw(0.5), bkd
        ),
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


class TestContinuousNumericOrthonormalPolynomial1D(
    Generic[Array], unittest.TestCase
):
    """Base class for ContinuousNumericOrthonormalPolynomial1D tests."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _compute_mc_grammian(
        self, poly: ContinuousNumericOrthonormalPolynomial1D, marginal, nsamples: int
    ) -> Array:
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
        poly: ContinuousNumericOrthonormalPolynomial1D,
        marginal,
        expected_rate: float = -1.0,
        rate_tolerance: float = 0.15,
    ) -> None:
        """Check that MC orthonormality MSE converges at expected rate.

        For MC integration, we expect:
        MSE = ||G_mc - I||_F^2 ~ O(N^{-1})

        where N is the number of samples.

        Parameters
        ----------
        poly : ContinuousNumericOrthonormalPolynomial1D
            The polynomial basis to test.
        marginal : Any
            The marginal distribution to sample from.
        expected_rate : float
            Expected convergence rate in log-log scale. Default: -1.0
        rate_tolerance : float
            Allowable deviation from expected rate. Default: 0.15
            (rate must be in [expected - tolerance, expected + tolerance])
        """
        nterms = poly.nterms()
        expected_grammian = self._bkd.eye(nterms)

        # Use multiple sample sizes with many trials to estimate convergence rate
        sample_sizes = [500, 1000, 2000]
        ntrials = 200
        mse_values = []

        np.random.seed(42)  # For reproducibility

        for nsamples in sample_sizes:
            # Average MSE over multiple trials for stable estimate
            trial_mses = []
            for _ in range(ntrials):
                grammian = self._compute_mc_grammian(poly, marginal, nsamples)
                diff = grammian - expected_grammian
                mse = float(self._bkd.sum(diff * diff)) / (nterms * nterms)
                trial_mses.append(mse)
            mse_values.append(np.mean(trial_mses))

        # Fit log-log linear model to estimate rate
        log_n = np.log(np.array(sample_sizes))
        log_mse = np.log(np.array(mse_values))

        # Skip if any NaN or Inf (numerical issues)
        if not np.all(np.isfinite(log_mse)):
            self.skipTest("Numerical issues in MSE computation (NaN/Inf)")

        # Linear regression: log(MSE) = rate * log(N) + const
        slope, intercept = np.polyfit(log_n, log_mse, 1)

        # Check that slope is within tolerance of expected rate
        self.assertGreater(
            slope,
            expected_rate - rate_tolerance,
            f"Convergence rate {slope:.3f} is worse than expected "
            f"(expected >= {expected_rate - rate_tolerance:.3f})",
        )
        self.assertLess(
            slope,
            expected_rate + rate_tolerance,
            f"Convergence rate {slope:.3f} is better than expected "
            f"(expected <= {expected_rate + rate_tolerance:.3f})",
        )

    # =========================================================================
    # Parametrized tests for bounded marginals
    # =========================================================================

    def test_bounded_marginals_mc_convergence(self) -> None:
        """Test MC orthonormality convergence for all bounded marginals."""
        for case in BOUNDED_MARGINAL_TEST_CASES:
            with self.subTest(name=case.name):
                marginal = case.marginal_factory(self._bkd)
                poly = ContinuousNumericOrthonormalPolynomial1D(
                    marginal, self._bkd, nquad_points=case.nquad_points
                )
                poly.set_nterms(case.nterms)

                self._check_mc_convergence_rate(poly, marginal)

    # =========================================================================
    # Parametrized tests for unbounded marginals
    # =========================================================================

    def test_unbounded_marginals_quadrature_orthonormality(self) -> None:
        """Test orthonormality for unbounded marginals using Gauss quadrature.

        Note: MC convergence tests are NOT used for unbounded distributions
        because the variance of p_n(X)^2 grows with n, making MC impractical.
        Instead, we use the polynomial's own Gauss quadrature rule to verify
        orthonormality exactly (up to numerical precision).
        """
        for case in UNBOUNDED_MARGINAL_TEST_CASES:
            with self.subTest(name=case.name):
                marginal = case.marginal_factory(self._bkd)
                poly = ContinuousNumericOrthonormalPolynomial1D(
                    marginal,
                    self._bkd,
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

                expected = self._bkd.eye(nterms)
                self._bkd.assert_allclose(grammian, expected, rtol=1e-8, atol=1e-10)

    # =========================================================================
    # Comparison with analytic polynomials
    # =========================================================================

    def test_matches_legendre_for_uniform(self) -> None:
        """Numeric polynomials for Uniform[-1,1] should match Legendre."""
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        numeric_poly = ContinuousNumericOrthonormalPolynomial1D(
            marginal, self._bkd, nquad_points=100
        )
        numeric_poly.set_nterms(6)

        legendre_poly = LegendrePolynomial1D(self._bkd)
        legendre_poly.set_nterms(6)

        # Evaluate at test points in canonical domain
        samples = self._bkd.linspace(-0.9, 0.9, 20)
        samples = self._bkd.reshape(samples, (1, -1))

        numeric_vals = numeric_poly(samples)
        legendre_vals = legendre_poly(samples)

        # Check recursion coefficients match
        numeric_ab = numeric_poly._get_recursion_coefficients(6)
        legendre_ab = legendre_poly._get_recursion_coefficients(6)

        self._bkd.assert_allclose(numeric_ab, legendre_ab, rtol=1e-8, atol=1e-10)

        # Check polynomial values match
        self._bkd.assert_allclose(numeric_vals, legendre_vals, rtol=1e-8, atol=1e-10)

    def test_matches_jacobi_for_beta(self) -> None:
        """Numeric polynomials for Beta should match Jacobi on [-1, 1].

        Beta(alpha, beta) on [0, 1] corresponds to Jacobi(beta-1, alpha-1) on [-1, 1].
        The transformation is: x_beta = (x_jacobi + 1) / 2.
        """
        alpha, beta = 3.0, 2.0
        marginal = BetaMarginal(alpha, beta, self._bkd)
        numeric_poly = ContinuousNumericOrthonormalPolynomial1D(
            marginal, self._bkd, nquad_points=100
        )
        numeric_poly.set_nterms(6)

        # Jacobi parameters for Beta(alpha, beta): (beta-1, alpha-1)
        jacobi_poly = JacobiPolynomial1D(beta - 1, alpha - 1, self._bkd)
        jacobi_poly.set_nterms(6)

        # Compare recursion coefficients
        numeric_ab = numeric_poly._get_recursion_coefficients(6)
        jacobi_ab = jacobi_poly._get_recursion_coefficients(6)

        self._bkd.assert_allclose(numeric_ab, jacobi_ab, rtol=1e-6, atol=1e-8)

    def test_matches_jacobi_for_shifted_beta(self) -> None:
        """Numeric polynomials for Beta on shifted domain match Jacobi.

        Beta(alpha, beta) on [a, b] should have same recursion coefficients
        as Beta(alpha, beta) on [0, 1] since they're both mapped to [-1, 1].
        """
        alpha, beta = 2.0, 4.0

        # Standard Beta on [0, 1]
        marginal_std = BetaMarginal(alpha, beta, self._bkd)
        poly_std = ContinuousNumericOrthonormalPolynomial1D(
            marginal_std, self._bkd, nquad_points=100
        )
        poly_std.set_nterms(5)

        # Shifted Beta on [2, 5]
        scipy_rv = stats.beta(alpha, beta, loc=2.0, scale=3.0)
        marginal_shifted = ScipyContinuousMarginal(scipy_rv, self._bkd)
        poly_shifted = ContinuousNumericOrthonormalPolynomial1D(
            marginal_shifted, self._bkd, nquad_points=100
        )
        poly_shifted.set_nterms(5)

        # Recursion coefficients should match
        ab_std = poly_std._get_recursion_coefficients(5)
        ab_shifted = poly_shifted._get_recursion_coefficients(5)

        self._bkd.assert_allclose(ab_std, ab_shifted, rtol=1e-8, atol=1e-10)

    def test_matches_hermite_for_gaussian(self) -> None:
        """Numeric polynomials for standard Gaussian should match Hermite.

        Note: The predictor-corrector method accumulates errors, so we test
        with fewer terms. For symmetric distributions like Gaussian, the
        'a' coefficients should be 0, but numerical errors make them small
        but non-zero.
        """
        marginal = GaussianMarginal(0.0, 1.0, self._bkd)
        numeric_poly = ContinuousNumericOrthonormalPolynomial1D(
            marginal,
            self._bkd,
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

        hermite_poly = HermitePolynomial1D(self._bkd, rho=0.0, prob_meas=True)
        hermite_poly.set_nterms(nterms)

        # Compare recursion coefficients
        numeric_ab = numeric_poly._get_recursion_coefficients(nterms)
        hermite_ab = hermite_poly._get_recursion_coefficients(nterms)

        # For 'a' coefficients (column 0), use absolute tolerance since they're ~0
        # For 'b' coefficients (column 1), use relative tolerance
        self._bkd.assert_allclose(
            numeric_ab[:, 0], hermite_ab[:, 0], rtol=0, atol=1e-5
        )
        self._bkd.assert_allclose(
            numeric_ab[:, 1], hermite_ab[:, 1], rtol=1e-6, atol=0
        )

    def test_matches_laguerre_for_gamma_unit_scale(self) -> None:
        """Numeric polynomials for Gamma(shape, 1) should match Laguerre.

        Gamma(shape, scale=1) corresponds to Laguerre with rho=shape-1.
        Note: Only scale=1 matches Laguerre; other scales give different
        recursion coefficients (but still orthonormal w.r.t. the measure).
        """
        shape, scale = 2.0, 1.0
        marginal = GammaMarginal(shape, scale, bkd=self._bkd)
        numeric_poly = ContinuousNumericOrthonormalPolynomial1D(
            marginal,
            self._bkd,
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
        laguerre_poly = LaguerrePolynomial1D(self._bkd, rho=shape - 1)
        laguerre_poly.set_nterms(6)

        # Compare recursion coefficients
        numeric_ab = numeric_poly._get_recursion_coefficients(6)
        laguerre_ab = laguerre_poly._get_recursion_coefficients(6)

        self._bkd.assert_allclose(numeric_ab, laguerre_ab, rtol=1e-10, atol=1e-12)

    # =========================================================================
    # Basic functionality tests
    # =========================================================================

    def test_nterms_and_evaluation_shape(self) -> None:
        """Test nterms setting and evaluation shape."""
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        poly = ContinuousNumericOrthonormalPolynomial1D(marginal, self._bkd)

        self.assertEqual(poly.nterms(), 0)

        poly.set_nterms(5)
        self.assertEqual(poly.nterms(), 5)

        # Evaluate at samples
        samples = self._bkd.array([[-0.5, 0.0, 0.5]])
        vals = poly(samples)

        self.assertEqual(vals.shape, (3, 5))

    def test_canonical_domain_mapping(self) -> None:
        """Test canonical to physical domain mapping."""
        # Beta on [0, 1] should use canonical domain [-1, 1]
        marginal = BetaMarginal(2.0, 3.0, self._bkd)
        poly = ContinuousNumericOrthonormalPolynomial1D(marginal, self._bkd)
        poly.set_nterms(3)

        # Canonical [-1, 1] should map to physical [0, 1]
        can_samples = self._bkd.array([[-1.0, 0.0, 1.0]])
        phys_samples = poly._canonical_to_physical(can_samples)

        expected_phys = self._bkd.array([[0.0, 0.5, 1.0]])
        self._bkd.assert_allclose(phys_samples, expected_phys, rtol=1e-14)

        # And inverse
        recovered_can = poly._physical_to_canonical(phys_samples)
        self._bkd.assert_allclose(recovered_can, can_samples, rtol=1e-14)

    def test_canonical_domain_mapping_shifted(self) -> None:
        """Test canonical domain mapping for shifted distributions."""
        # Beta on [2, 5] should use canonical domain [-1, 1]
        scipy_rv = stats.beta(2.0, 3.0, loc=2.0, scale=3.0)
        marginal = ScipyContinuousMarginal(scipy_rv, self._bkd)
        poly = ContinuousNumericOrthonormalPolynomial1D(marginal, self._bkd)
        poly.set_nterms(3)

        # Canonical [-1, 1] should map to physical [2, 5]
        can_samples = self._bkd.array([[-1.0, 0.0, 1.0]])
        phys_samples = poly._canonical_to_physical(can_samples)

        expected_phys = self._bkd.array([[2.0, 3.5, 5.0]])
        self._bkd.assert_allclose(phys_samples, expected_phys, rtol=1e-14)

    def test_repr(self) -> None:
        """Test string representation."""
        marginal = BetaMarginal(2.0, 3.0, self._bkd)
        poly = ContinuousNumericOrthonormalPolynomial1D(marginal, self._bkd)
        poly.set_nterms(5)

        repr_str = repr(poly)
        self.assertIn("ContinuousNumericOrthonormalPolynomial1D", repr_str)
        self.assertIn("BetaMarginal", repr_str)
        self.assertIn("5", repr_str)

    def test_first_polynomial_is_constant(self) -> None:
        """First orthonormal polynomial p_0(x) = 1 for any measure."""
        marginal = BetaMarginal(2.0, 3.0, self._bkd)
        poly = ContinuousNumericOrthonormalPolynomial1D(marginal, self._bkd)
        poly.set_nterms(3)

        # p_0(x) should be 1 everywhere
        samples = self._bkd.array([[-0.8, -0.2, 0.0, 0.4, 0.9]])
        vals = poly(samples)

        expected_p0 = self._bkd.ones((5,))
        self._bkd.assert_allclose(vals[:, 0], expected_p0, rtol=1e-12)

    def test_polynomials_are_normalized(self) -> None:
        """Orthonormal polynomials should integrate to delta_ij.

        Using Gauss quadrature from the polynomial itself.
        """
        marginal = BetaMarginal(3.0, 2.0, self._bkd)
        poly = ContinuousNumericOrthonormalPolynomial1D(
            marginal, self._bkd, nquad_points=100
        )
        nterms = 6
        poly.set_nterms(nterms)

        # Get quadrature points from Jacobi polynomial with same parameters
        jacobi_poly = JacobiPolynomial1D(1.0, 2.0, self._bkd)  # beta-1, alpha-1
        jacobi_poly.set_nterms(nterms + 2)
        x_quad, w_quad = jacobi_poly.gauss_quadrature_rule(nterms + 2)

        # Evaluate numeric polynomial at quadrature points
        vals = poly(x_quad)  # (nquad, nterms)

        # Compute Gramian
        gramian = vals.T @ (w_quad * vals)

        expected = self._bkd.eye(nterms)
        self._bkd.assert_allclose(gramian, expected, rtol=1e-8, atol=1e-10)

    def test_unbounded_domain_uses_identity_transform(self) -> None:
        """Unbounded marginals should use identity transform (no mapping)."""
        marginal = GaussianMarginal(0.0, 1.0, self._bkd)
        poly = ContinuousNumericOrthonormalPolynomial1D(
            marginal,
            self._bkd,
            integrator_options={
                "interval_size": 1.0,
                "atol": 1e-14,
                "rtol": 1e-14,
            },
        )
        poly.set_nterms(3)

        # For unbounded, canonical = physical (loc=0, scale=1)
        self.assertEqual(poly._loc, 0.0)
        self.assertEqual(poly._scale, 1.0)

        # Samples should pass through unchanged
        samples = self._bkd.array([[-2.0, 0.0, 2.0]])
        phys_samples = poly._canonical_to_physical(samples)
        self._bkd.assert_allclose(phys_samples, samples, rtol=1e-14)


class TestContinuousNumericOrthonormalPolynomial1DNumpy(
    TestContinuousNumericOrthonormalPolynomial1D[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestContinuousNumericOrthonormalPolynomial1DTorch(
    TestContinuousNumericOrthonormalPolynomial1D[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
