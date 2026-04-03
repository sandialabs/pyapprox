"""
Tests for GP statistics moments.

Tests the SeparableKernelIntegralCalculator and GaussianProcessStatistics
classes for computing statistical quantities from fitted GPs.
"""

import math

import numpy as np
import pytest

from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.statistics import (
    GaussianProcessStatistics,
    SeparableKernelIntegralCalculator,
)
from pyapprox.surrogates.kernels.composition import SeparableProductKernel
from pyapprox.surrogates.kernels.matern import (
    Matern52Kernel,
    SquaredExponentialKernel,
)
from pyapprox.surrogates.sparsegrids.basis_factory import (
    create_basis_factories,
)
from tests._helpers.markers import slow_test


def _create_quadrature_bases(
    marginals,
    nquad_points,
    bkd,
):
    """Helper to create quadrature bases from marginals."""
    factories = create_basis_factories(marginals, bkd, "gauss")
    bases = [f.create_basis() for f in factories]
    for b in bases:
        b.set_nterms(nquad_points)
    return bases


class TestSeparableKernelIntegralCalculator:
    """
    Test class for SeparableKernelIntegralCalculator.
    """

    def _setup(self, bkd):
        np.random.seed(42)

        # Create 2D GP with separable product kernel
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        gp = ExactGaussianProcess(kernel, nvars=2, bkd=bkd, nugget=1e-6)
        # Skip hyperparameter optimization for these tests
        gp.hyp_list().set_all_inactive()

        # Training data
        n_train = 10
        X_train_np = np.random.rand(2, n_train) * 2 - 1  # [-1, 1]^2
        X_train = bkd.array(X_train_np)
        # Use backend math operations, shape: (nqoi, n_train)
        y_train = bkd.reshape(
            bkd.sin(math.pi * X_train[0, :]) * bkd.cos(math.pi * X_train[1, :]), (1, -1)
        )

        gp.fit(X_train, y_train)

        # Marginal distributions
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]

        # Create quadrature bases using sparse grid infrastructure
        nquad_points = 20
        bases = _create_quadrature_bases(marginals, nquad_points, bkd)

        # Create calculator
        calc = SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)

        return calc, n_train

    def test_tau_shape(self, bkd) -> None:
        """Test tau has correct shape (N,)."""
        calc, n_train = self._setup(bkd)
        tau = calc.tau_C()
        assert tau.shape == (n_train,)

    def test_tau_positive(self, bkd) -> None:
        """Test tau values are positive (kernel is positive)."""
        calc, _ = self._setup(bkd)
        tau = calc.tau_C()
        assert bkd.all_bool(tau > 0)

    def test_P_shape(self, bkd) -> None:
        """Test P has correct shape (N, N)."""
        calc, n_train = self._setup(bkd)
        P = calc.P()
        assert P.shape == (n_train, n_train)

    def test_P_symmetric(self, bkd) -> None:
        """Test P is symmetric."""
        calc, _ = self._setup(bkd)
        P = calc.P()
        bkd.assert_allclose(P, P.T, rtol=1e-12)

    def test_P_positive_semidefinite(self, bkd) -> None:
        """Test P is positive semi-definite."""
        calc, _ = self._setup(bkd)
        P = calc.P()
        eigvals = bkd.eigvalsh(P)
        # Allow small negative eigenvalues due to numerical error
        assert bkd.all_bool(eigvals > -1e-10)

    def test_u_positive(self, bkd) -> None:
        """Test u is positive."""
        calc, _ = self._setup(bkd)
        u = calc.u()
        assert float(bkd.to_numpy(u)) > 0.0

    def test_caching(self, bkd) -> None:
        """Test that results are cached."""
        calc, _ = self._setup(bkd)
        tau1 = calc.tau_C()
        tau2 = calc.tau_C()
        # Same object (cached)
        assert tau1 is tau2

    def test_conditional_P_subset(self, bkd) -> None:
        """Test conditional_P with binary index vector.

        index[k] = 1: dimension k is CONDITIONED ON (use standard P_k)
        index[k] = 0: dimension k is INTEGRATED OUT (use P_tilde_k = tau_k tau_k^T)
        """
        calc, n_train = self._setup(bkd)
        # Condition on dimension 0, integrate out dimension 1
        # index = [1, 0] means: dim 0 conditioned (use P_0),
        # dim 1 integrated (use tau_1 tau_1^T)
        index = bkd.asarray([1.0, 0.0])
        P_cond = calc.conditional_P(index)

        # Should have correct shape
        assert P_cond.shape == (n_train, n_train)
        # P_cond should be symmetric
        bkd.assert_allclose(P_cond, P_cond.T, rtol=1e-12)
        # P_cond should be positive semi-definite
        eigvals = bkd.eigvalsh(P_cond)
        assert bkd.all_bool(eigvals > -1e-10)

    def test_conditional_u_subset(self, bkd) -> None:
        """Test conditional_u with binary index vector.

        index[k] = 1: dimension k is CONDITIONED ON (factor = 1)
        index[k] = 0: dimension k is INTEGRATED OUT (factor = u_k)
        """
        calc, _ = self._setup(bkd)
        # Condition on dimension 0, integrate out dimension 1
        # u_p = 1 * u_1 (only dim 1 contributes to the integral)
        index = bkd.asarray([1.0, 0.0])
        u_cond = calc.conditional_u(index)

        # Should be positive and <= 1 (max kernel value for normalized kernels)
        u_cond_val = float(bkd.to_numpy(u_cond))
        assert u_cond_val > 0.0
        assert u_cond_val <= 1.0 + 1e-10


class TestGaussianProcessStatistics:
    """
    Test class for GaussianProcessStatistics.
    """

    def _setup(self, bkd):
        np.random.seed(42)

        # Create 2D GP with separable product kernel
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        gp = ExactGaussianProcess(kernel, nvars=2, bkd=bkd, nugget=1e-6)
        # Skip hyperparameter optimization for these tests
        gp.hyp_list().set_all_inactive()

        # Training data
        n_train = 10
        X_train_np = np.random.rand(2, n_train) * 2 - 1
        X_train = bkd.array(X_train_np)
        # Use backend math operations, shape: (nqoi, n_train)
        y_train = bkd.reshape(
            bkd.sin(math.pi * X_train[0, :]) * bkd.cos(math.pi * X_train[1, :]), (1, -1)
        )

        gp.fit(X_train, y_train)

        # Marginal distributions
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]

        # Create quadrature bases and calculator
        bases = _create_quadrature_bases(marginals, 30, bkd)
        calc = SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)
        stats = GaussianProcessStatistics(gp, calc)

        return stats, gp, marginals

    def test_mean_of_mean_scalar(self, bkd) -> None:
        """Test mean_of_mean returns a scalar."""
        stats, _, _ = self._setup(bkd)
        eta = stats.mean_of_mean()
        # Should be a scalar (0-dim array)
        assert len(eta.shape) == 0

    def test_variance_of_mean_nonnegative(self, bkd) -> None:
        """Test Var[mu_f] >= 0."""
        stats, _, _ = self._setup(bkd)
        var_mu = stats.variance_of_mean()
        assert float(bkd.to_numpy(var_mu)) >= 0.0

    def test_mean_of_variance_nonnegative(self, bkd) -> None:
        """Test E[gamma_f] >= 0."""
        stats, _, _ = self._setup(bkd)
        mean_var = stats.mean_of_variance()
        assert float(bkd.to_numpy(mean_var)) >= 0.0

    def test_caching(self, bkd) -> None:
        """Test that results are cached."""
        stats, _, _ = self._setup(bkd)
        eta1 = stats.mean_of_mean()
        eta2 = stats.mean_of_mean()
        # Same object (cached)
        assert eta1 is eta2

    def test_variance_of_variance_nonnegative(self, bkd) -> None:
        """Test Var[gamma_f] >= 0."""
        stats, _, _ = self._setup(bkd)
        var_var = stats.variance_of_variance()
        assert float(bkd.to_numpy(var_var)) >= 0.0

    def test_variance_of_variance_scalar(self, bkd) -> None:
        """Test variance_of_variance returns a scalar."""
        stats, _, _ = self._setup(bkd)
        var_var = stats.variance_of_variance()
        # Should be a scalar (0-dim array)
        assert len(var_var.shape) == 0

    def test_variance_of_variance_caching(self, bkd) -> None:
        """Test that variance_of_variance results are cached."""
        stats, _, _ = self._setup(bkd)
        var_var1 = stats.variance_of_variance()
        var_var2 = stats.variance_of_variance()
        # Same object (cached)
        assert var_var1 is var_var2

    @slow_test
    def test_limit_many_training_points(self, bkd) -> None:
        """Test that Var[mu_f] -> 0 as N -> infinity."""
        stats, _, marginals = self._setup(bkd)
        np.random.seed(123)

        # Create GP with more training points
        n_train_large = 100
        X_train_np = np.random.rand(2, n_train_large) * 2 - 1
        X_train = bkd.array(X_train_np)
        # Use backend math operations, shape: (nqoi, n_train)
        y_train = bkd.reshape(
            bkd.sin(math.pi * X_train[0, :]) * bkd.cos(math.pi * X_train[1, :]), (1, -1)
        )

        k1 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, bkd)
        k2 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        gp_large = ExactGaussianProcess(kernel, nvars=2, bkd=bkd, nugget=1e-6)
        # Skip hyperparameter optimization for this test
        gp_large.hyp_list().set_all_inactive()
        gp_large.fit(X_train, y_train)

        # Create quadrature bases
        bases_large = _create_quadrature_bases(marginals, 30, bkd)
        calc_large = SeparableKernelIntegralCalculator(
            gp_large, bases_large, marginals, bkd=bkd
        )
        stats_large = GaussianProcessStatistics(gp_large, calc_large)

        # Variance of mean should be small with many training points
        var_mu_large = stats_large.variance_of_mean()
        var_mu_small = stats.variance_of_mean()

        var_large_val = float(bkd.to_numpy(var_mu_large))
        var_small_val = float(bkd.to_numpy(var_mu_small))

        # More data should give smaller variance
        # When both are numerically zero, they're effectively equal (which is OK)
        if var_small_val < 1e-12:
            # Original variance already tiny - just check large is also small
            assert var_large_val < 1e-10
        else:
            assert var_large_val < var_small_val


class TestMCComparison:
    """
    Test Monte Carlo comparison for GP statistics.

    Compares quadrature-based results to Monte Carlo estimates.

    CRITICAL: Test setup must have non-negligible posterior variance at
    quadrature points. Use sparse training data so the GP has uncertainty
    in regions covered by the quadrature rule.
    """

    def _setup(self, bkd):
        np.random.seed(42)

        # Create 1D GP with shorter length scale
        # Shorter length scale = more local correlation = higher posterior variance
        # between training points
        k1 = SquaredExponentialKernel([0.3], (0.1, 10.0), 1, bkd)
        kernel = k1

        gp = ExactGaussianProcess(kernel, nvars=1, bkd=bkd, nugget=1e-6)
        # Skip hyperparameter optimization for these tests
        gp.hyp_list().set_all_inactive()

        # SPARSE training data - only 3 points at boundaries and center
        # With short length scale, regions between points have high uncertainty
        _n_train = 3
        X_train = bkd.array([[-1.0, 0.0, 1.0]])  # Only 3 points
        # Use bkd.sin for backend compatibility
        y_train = bkd.reshape(bkd.sin(3.14159 * X_train[0, :]), (1, -1))

        gp.fit(X_train, y_train)

        # Marginal distributions
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]

        # Create quadrature bases and calculator
        nquad = 30
        bases = _create_quadrature_bases(marginals, nquad, bkd)
        calc = SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)
        stats = GaussianProcessStatistics(gp, calc)

        # Verify posterior variance is non-negligible at quadrature points
        quad_pts, _ = bases[0].quadrature_rule()
        post_std = gp.predict_std(quad_pts)
        mean_post_var = float(bkd.mean(post_std**2))
        assert mean_post_var > 0.01, (
            f"Posterior variance too small ({mean_post_var:.6f}). "
            "Test setup needs sparser training data."
        )

        return stats, gp, bases

    @slow_test
    def test_mc_convergence_mean_of_mean(self, bkd) -> None:
        """Compare quadrature E[mu] to MC estimate."""
        stats, gp, _ = self._setup(bkd)
        np.random.seed(12345)

        # Quadrature estimate
        eta_quad = stats.mean_of_mean()

        # MC estimate - generate random samples and convert to backend array
        n_mc = 10000
        X_mc = bkd.array(np.random.rand(1, n_mc) * 2 - 1)  # Uniform [-1, 1]

        # Evaluate GP mean at MC samples
        mu_mc = gp.predict(X_mc)  # Shape: (n_mc, 1)
        eta_mc = bkd.mean(mu_mc)

        # Compare (allow for MC error)
        eta_quad_val = float(bkd.to_numpy(eta_quad))
        eta_mc_val = float(bkd.to_numpy(eta_mc))

        # MC should be within ~3 standard errors (allowing for randomness)
        # Standard error is O(1/sqrt(n_mc)) ~ 0.01 for n_mc=10000
        assert abs(eta_quad_val - eta_mc_val) < 0.05

    @slow_test
    def test_mc_convergence_variance_of_mean(self, bkd) -> None:
        """
        Compare quadrature Var[mu_f] to MC estimate.

        Var[mu_f] = Var[integral f(z) dF(z)] requires sampling from the GP posterior.

        For each sample f^(r) from the posterior:
        1. Evaluate f^(r) at quadrature points
        2. Compute mu^(r) = Sum_j w_j f^(r)(z_j)
        3. Compute Var[{mu^(r)}] across samples

        GP posterior at points Z has:
        - Mean: m*(Z) = K(Z,X) @ A^{-1} @ y
        - Covariance: C*(Z,Z) = K(Z,Z) - K(Z,X) @ A^{-1} @ K(X,Z)

        Sample via: f = m* + L @ z where L = cholesky(C*), z ~ N(0,I)
        """
        stats, gp, bases = self._setup(bkd)
        np.random.seed(12345)

        # Quadrature estimate
        var_quad = stats.variance_of_mean()

        # Get quadrature points and weights from the bases used by the calculator
        quad_pts, quad_wts = bases[0].quadrature_rule()
        # quad_pts shape: (1, nquad), quad_wts shape: (nquad, 1)
        quad_wts = bkd.reshape(quad_wts, (-1,))  # (nquad,)
        nquad = quad_pts.shape[1]

        # Compute GP posterior mean and covariance at quadrature points
        X_train = gp.data().X()  # (1, n_train)
        y_train = gp.data().y()  # (nqoi, n_train)

        # Prior covariance matrices
        K_qq = gp.kernel()(quad_pts, quad_pts)  # (nquad, nquad)
        K_qt = gp.kernel()(quad_pts, X_train)  # (nquad, n_train)

        # Posterior mean: m* = K_qt @ A^{-1} @ y
        chol = gp.cholesky()
        alpha = chol.solve(y_train.T)  # A^{-1} @ y.T, shape (n_train, 1)
        m_star = K_qt @ alpha  # (nquad, 1)
        m_star = bkd.reshape(m_star, (-1,))  # (nquad,)

        # Posterior covariance: C* = K_qq - K_qt @ A^{-1} @ K_tq
        A_inv_K_tq = chol.solve(K_qt.T)  # (n_train, nquad)
        C_star = K_qq - K_qt @ A_inv_K_tq  # (nquad, nquad)

        # Add small jitter for numerical stability
        C_star = C_star + 1e-8 * bkd.eye(nquad)

        # Cholesky of posterior covariance
        L_star = bkd.cholesky(C_star)  # (nquad, nquad)

        # Sample from GP posterior and compute mu^(r) for each sample
        n_samples = 10000
        mu_samples = []

        for _ in range(n_samples):
            # Sample z ~ N(0, I)
            z = bkd.array(np.random.randn(nquad))

            # Sample f = m* + L @ z
            f_sample = m_star + L_star @ z  # (nquad,)

            # Compute mu^(r) = Sum_j w_j f^(r)(z_j)
            mu_r = bkd.sum(quad_wts * f_sample)
            mu_samples.append(float(bkd.to_numpy(mu_r)))

        # Compute variance across samples
        mu_samples_arr = np.array(mu_samples)
        var_mc = np.var(mu_samples_arr)

        # Compare
        var_quad_val = float(bkd.to_numpy(var_quad))

        # Allow tolerance for MC error
        # With 10000 samples and variance ~O(0.1), relative error should be ~1-2%
        # Use relative comparison for robustness
        rel_error = abs(var_quad_val - var_mc) / max(var_quad_val, 1e-10)
        assert rel_error < 0.1, (
            f"Relative error {rel_error:.4f} too large: "
            f"quad={var_quad_val:.6f}, mc={var_mc:.6f}"
        )

    @slow_test
    def test_mc_convergence_variance_of_variance(self, bkd) -> None:
        """
        Compare quadrature Var[gamma_f] to MC estimate.

        Var[gamma_f] requires sampling from the GP posterior and computing
        variance statistics across samples.

        For each sample f^(r) from the posterior:
        1. Evaluate f^(r) at quadrature points
        2. Compute kappa^(r) = Sum_j w_j [f^(r)(z_j)]**2
        3. Compute mu^(r) = Sum_j w_j f^(r)(z_j)
        4. Compute gamma^(r) = kappa^(r) - [mu^(r)]**2
        5. Compute Var[{gamma^(r)}] across samples
        """
        stats, gp, bases = self._setup(bkd)
        np.random.seed(12345)

        # Quadrature estimate
        var_var_quad = stats.variance_of_variance()

        # Get quadrature points and weights from the bases used by the calculator
        quad_pts, quad_wts = bases[0].quadrature_rule()
        # quad_pts shape: (1, nquad), quad_wts shape: (nquad, 1)
        quad_wts = bkd.reshape(quad_wts, (-1,))  # (nquad,)
        nquad = quad_pts.shape[1]

        # Compute GP posterior mean and covariance at quadrature points
        X_train = gp.data().X()  # (1, n_train)
        y_train = gp.data().y()  # (nqoi, n_train)

        # Prior covariance matrices
        K_qq = gp.kernel()(quad_pts, quad_pts)  # (nquad, nquad)
        K_qt = gp.kernel()(quad_pts, X_train)  # (nquad, n_train)

        # Posterior mean: m* = K_qt @ A^{-1} @ y
        chol = gp.cholesky()
        alpha = chol.solve(y_train.T)  # A^{-1} @ y.T, shape (n_train, 1)
        m_star = K_qt @ alpha  # (nquad, 1)
        m_star = bkd.reshape(m_star, (-1,))  # (nquad,)

        # Posterior covariance: C* = K_qq - K_qt @ A^{-1} @ K_tq
        A_inv_K_tq = chol.solve(K_qt.T)  # (n_train, nquad)
        C_star = K_qq - K_qt @ A_inv_K_tq  # (nquad, nquad)

        # Add small jitter for numerical stability
        C_star = C_star + 1e-8 * bkd.eye(nquad)

        # Cholesky of posterior covariance
        L_star = bkd.cholesky(C_star)  # (nquad, nquad)

        # Sample from GP posterior and compute gamma^(r) for each sample
        n_samples = 10000
        gamma_samples = []

        for _ in range(n_samples):
            # Sample z ~ N(0, I)
            z = bkd.array(np.random.randn(nquad))

            # Sample f = m* + L @ z
            f_sample = m_star + L_star @ z  # (nquad,)

            # Compute kappa^(r) = Sum_j w_j [f^(r)(z_j)]**2
            kappa_r = bkd.sum(quad_wts * f_sample**2)

            # Compute mu^(r) = Sum_j w_j f^(r)(z_j)
            mu_r = bkd.sum(quad_wts * f_sample)

            # Compute gamma^(r) = kappa^(r) - [mu^(r)]**2
            gamma_r = kappa_r - mu_r**2
            gamma_samples.append(float(bkd.to_numpy(gamma_r)))

        # Compute variance across samples
        gamma_samples_arr = np.array(gamma_samples)
        var_var_mc = np.var(gamma_samples_arr)

        # Compare
        var_var_quad_val = float(bkd.to_numpy(var_var_quad))

        # Allow tolerance for MC error
        # Variance of variance is a fourth-moment statistic, so has higher
        # MC error than variance of mean
        rel_error = abs(var_var_quad_val - var_var_mc) / max(var_var_quad_val, 1e-10)
        assert rel_error < 0.2, (
            f"Relative error {rel_error:.4f} too large: "
            f"quad={var_var_quad_val:.6f}, "
            f"mc={var_var_mc:.6f}"
        )


class TestKnownMoments:
    """
    Test GP statistics against functions with analytically known moments.

    When the GP interpolates a function exactly (to within numerical precision),
    the GP statistics should match the analytical moments of the function.

    For a function f(z) with z ~ Uniform[-1, 1]^d:
    - E[mu_f] = E[f(z)] (mean of the function)
    - E[gamma_f] approx Var[f(z)] (mean of GP variance approx function variance)

    CRITICAL: We first verify the GP interpolation error is small (< stats_tol)
    at test points. This is a necessary condition for accurate statistics.
    """

    def _verify_gp_interpolation(
        self,
        bkd,
        gp,
        test_func,
        X_test,
        stats_tol,
    ):
        """
        Verify GP interpolation error is below stats tolerance.

        Parameters
        ----------
        bkd : Backend
            Backend instance.
        gp : ExactGaussianProcess
            Fitted GP.
        test_func : callable
            True function, takes X of shape (nvars, nsamples).
        X_test : Array
            Test points of shape (nvars, n_test).
        stats_tol : float
            Tolerance for statistics tests.

        Returns
        -------
        max_error : float
            Maximum absolute interpolation error.

        Raises
        ------
        AssertionError
            If max error exceeds stats_tol.
        """
        # Get GP predictions
        y_pred = gp.predict(X_test)  # Shape: (n_test, 1)
        y_pred = bkd.reshape(y_pred, (-1,))

        # Get true values
        y_true = test_func(X_test)  # Shape: (1, n_test) or (n_test,)
        if y_true.ndim == 2:
            y_true = bkd.reshape(y_true, (-1,))

        # Compute max error
        errors = bkd.abs(y_pred - y_true)
        max_error = float(bkd.to_numpy(bkd.max(errors)))

        assert max_error < stats_tol, (
            f"GP interpolation error ({max_error:.2e}) exceeds stats tolerance "
            f"({stats_tol:.2e}). Increase training points or adjust kernel."
        )

        return max_error

    @slow_test
    def test_linear_function_mean(self, bkd) -> None:
        """
        Test GP on linear function f(x) = a*x_1 + b*x_2 + c.

        For z ~ Uniform[-1, 1]^2:
        - E[z_i] = 0
        - E[f] = c (since E[z_i] = 0)
        - Var[z_i] = 1/3
        - Var[f] = a**2/3 + b**2/3

        Uses dense training grid and optimized hyperparameters so GP
        interpolates to within tolerance.
        """
        np.random.seed(42)

        # Linear function parameters
        a, b, c = 2.0, 3.0, 1.5

        # Expected moments
        expected_mean = c  # E[f] = c since E[z_i] = 0
        expected_var = (a**2 + b**2) / 3.0  # Var[f] = (a**2 + b**2) * Var[z]

        # Tolerance for GP interpolation and statistics comparison
        # GP cannot achieve machine precision; use achievable tolerance
        # Use 2e-4 to account for GP interpolation error
        stats_tol = 2e-3

        # Define test function
        def linear_func(X):
            return bkd.reshape(a * X[0, :] + b * X[1, :] + c, (1, -1))

        # Create GP with separable product kernel
        # Initial length scales - will be optimized
        k1 = SquaredExponentialKernel([1.0], (0.01, 100.0), 1, bkd)
        k2 = SquaredExponentialKernel([1.0], (0.01, 100.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        gp = ExactGaussianProcess(kernel, nvars=2, bkd=bkd, nugget=1e-10)
        # Skip hyperparameter optimization since SeparableProductKernel
        # doesn't implement jacobian_wrt_params yet
        gp.hyp_list().set_all_inactive()

        # Dense training grid for good interpolation
        n_1d = 30
        x1 = bkd.linspace(-1.0, 1.0, n_1d)
        x2 = bkd.linspace(-1.0, 1.0, n_1d)
        X1, X2 = bkd.meshgrid(x1, x2)
        X_train = bkd.vstack([bkd.flatten(X1), bkd.flatten(X2)])

        # Evaluate function
        y_train = bkd.reshape(a * X_train[0, :] + b * X_train[1, :] + c, (1, -1))

        gp.fit(X_train, y_train)

        # Generate test points (different from training)
        n_test = 100
        X_test = bkd.array(np.random.rand(2, n_test) * 2 - 1)

        # Verify GP interpolation error is small enough
        self._verify_gp_interpolation(bkd, gp, linear_func, X_test, stats_tol)

        # Create quadrature bases with high order
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]
        bases = _create_quadrature_bases(marginals, 50, bkd)

        calc = SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)
        stats = GaussianProcessStatistics(gp, calc)

        # Test mean of mean - tolerance should be comparable to GP error
        eta = stats.mean_of_mean()
        eta_val = float(bkd.to_numpy(eta))
        bkd.assert_allclose(
            bkd.asarray([eta_val]),
            bkd.asarray([expected_mean]),
            rtol=stats_tol,
            err_msg=f"E[mu_f] = {eta_val}, expected {expected_mean}",
        )

        # Test mean of variance - tolerance should be comparable to GP error
        mean_var = stats.mean_of_variance()
        mean_var_val = float(bkd.to_numpy(mean_var))
        bkd.assert_allclose(
            bkd.asarray([mean_var_val]),
            bkd.asarray([expected_var]),
            rtol=stats_tol,
            err_msg=f"E[gamma_f] = {mean_var_val}, expected {expected_var}",
        )

    @slow_test
    def test_quadratic_function_mean(self, bkd) -> None:
        """
        Test GP on quadratic function f(x) = x_1**2 + x_2**2.

        For z ~ Uniform[-1, 1]^2:
        - E[z_i**2] = 1/3
        - E[f] = E[z_1**2] + E[z_2**2] = 2/3
        - Var[z_i**2] = E[z_i**4] - E[z_i**2]**2 = 1/5 - 1/9 = 4/45
        - Var[f] = Var[z_1**2] + Var[z_2**2] = 8/45
          (since x_1**2, x_2**2 are independent)

        Uses dense training grid and optimized hyperparameters so GP
        interpolates to within tolerance.
        """
        np.random.seed(42)

        # Expected moments for f = x_1**2 + x_2**2
        expected_mean = 2.0 / 3.0  # E[f] = 2 * E[z**2]
        expected_var = 8.0 / 45.0  # Var[f] = 2 * Var[z**2]

        # Tolerance for GP interpolation and statistics comparison
        stats_tol = 5e-3

        # Define test function
        def quadratic_func(X):
            return bkd.reshape(X[0, :] ** 2 + X[1, :] ** 2, (1, -1))

        # Create GP with separable product kernel
        k1 = SquaredExponentialKernel([1.0], (0.01, 100.0), 1, bkd)
        k2 = SquaredExponentialKernel([1.0], (0.01, 100.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        gp = ExactGaussianProcess(kernel, nvars=2, bkd=bkd, nugget=1e-10)
        # Skip hyperparameter optimization since SeparableProductKernel
        # doesn't implement jacobian_wrt_params yet
        gp.hyp_list().set_all_inactive()

        # Dense training grid for good interpolation
        n_1d = 30
        x1 = bkd.linspace(-1.0, 1.0, n_1d)
        x2 = bkd.linspace(-1.0, 1.0, n_1d)
        X1, X2 = bkd.meshgrid(x1, x2)
        X_train = bkd.vstack([bkd.flatten(X1), bkd.flatten(X2)])

        # Evaluate function - shape: (nqoi, n_train)
        y_train = bkd.reshape(X_train[0, :] ** 2 + X_train[1, :] ** 2, (1, -1))

        gp.fit(X_train, y_train)

        # Generate test points (different from training)
        n_test = 100
        X_test = bkd.array(np.random.rand(2, n_test) * 2 - 1)

        # Verify GP interpolation error is small enough
        self._verify_gp_interpolation(bkd, gp, quadratic_func, X_test, stats_tol)

        # Create quadrature bases with high order
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]
        bases = _create_quadrature_bases(marginals, 50, bkd)

        calc = SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)
        stats = GaussianProcessStatistics(gp, calc)

        # Test mean of mean - tolerance should be comparable to GP error
        eta = stats.mean_of_mean()
        eta_val = float(bkd.to_numpy(eta))
        bkd.assert_allclose(
            bkd.asarray([eta_val]),
            bkd.asarray([expected_mean]),
            rtol=stats_tol,
            err_msg=f"E[mu_f] = {eta_val}, expected {expected_mean}",
        )

        # Test mean of variance - tolerance should be comparable to GP error
        mean_var = stats.mean_of_variance()
        mean_var_val = float(bkd.to_numpy(mean_var))
        bkd.assert_allclose(
            bkd.asarray([mean_var_val]),
            bkd.asarray([expected_var]),
            rtol=stats_tol,
            err_msg=f"E[gamma_f] = {mean_var_val}, expected {expected_var}",
        )

    @slow_test
    def test_sinusoidal_function_mean(self, bkd) -> None:
        """
        Test GP on sinusoidal function f(x) = sin(pi*x_1).

        For z_1 ~ Uniform[-1, 1]:
        - E[sin(pi*z_1)] = integral_{-1}^{1} sin(pi*z) * (1/2) dz = 0 (odd function)
        - E[sin**2(pi*z_1)] = integral_{-1}^{1} sin**2(pi*z) * (1/2) dz = 1/2
        - Var[f] = E[f**2] - E[f]**2 = 1/2 - 0 = 1/2

        Uses dense training grid and optimized hyperparameters so GP
        interpolates to within tolerance.
        """
        np.random.seed(42)

        # Expected moments for f = sin(pi*x_1)
        expected_mean = 0.0  # E[sin(pi*z)] = 0 (odd function)
        expected_var = 0.5  # Var[sin(pi*z)] = E[sin**2] = 1/2

        # Tolerance for GP interpolation and statistics comparison
        stats_tol = 2e-3

        # Define test function using backend pi
        import math

        pi = math.pi

        def sin_func(X):
            return bkd.reshape(bkd.sin(pi * X[0, :]), (1, -1))

        # Create 1D GP
        k1 = SquaredExponentialKernel([1.0], (0.01, 100.0), 1, bkd)

        gp = ExactGaussianProcess(k1, nvars=1, bkd=bkd, nugget=1e-10)

        # Dense training grid for good interpolation
        n_train = 50
        X_train = bkd.reshape(bkd.linspace(-1.0, 1.0, n_train), (1, -1))
        # Shape: (nqoi, n_train)
        y_train = bkd.reshape(bkd.sin(pi * X_train[0, :]), (1, -1))

        # fit() optimizes hyperparameters by default for best interpolation
        gp.fit(X_train, y_train)

        # Generate test points (different from training)
        n_test = 100
        X_test = bkd.array(np.random.rand(1, n_test) * 2 - 1)

        # Verify GP interpolation error is small enough
        self._verify_gp_interpolation(bkd, gp, sin_func, X_test, stats_tol)

        # Create quadrature bases with high order
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases = _create_quadrature_bases(marginals, 50, bkd)

        calc = SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)
        stats = GaussianProcessStatistics(gp, calc)

        # Test mean of mean - use atol since expected is 0
        eta = stats.mean_of_mean()
        eta_val = float(bkd.to_numpy(eta))
        bkd.assert_allclose(
            bkd.asarray([eta_val]),
            bkd.asarray([expected_mean]),
            atol=stats_tol,
            err_msg=f"E[mu_f] = {eta_val}, expected {expected_mean}",
        )

        # Test mean of variance
        mean_var = stats.mean_of_variance()
        mean_var_val = float(bkd.to_numpy(mean_var))
        bkd.assert_allclose(
            bkd.asarray([mean_var_val]),
            bkd.asarray([expected_var]),
            rtol=stats_tol,
            err_msg=f"E[gamma_f] = {mean_var_val}, expected {expected_var}",
        )


class TestValidation:
    """Test validation and error handling."""

    def test_unfitted_gp_raises_error(self, bkd) -> None:
        """Test that unfitted GP raises RuntimeError."""
        np.random.seed(42)
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        gp = ExactGaussianProcess(k1, nvars=1, bkd=bkd)

        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases = _create_quadrature_bases(marginals, 10, bkd)

        with pytest.raises(RuntimeError):
            SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)

    def test_wrong_number_of_bases_raises_error(self, bkd) -> None:
        """Test that wrong number of quadrature bases raises ValueError."""
        np.random.seed(42)
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        gp = ExactGaussianProcess(kernel, nvars=2, bkd=bkd)
        # Skip hyperparameter optimization for this test
        gp.hyp_list().set_all_inactive()
        X = bkd.array(np.random.rand(2, 5))
        y = bkd.array(np.random.rand(1, 5))  # Shape: (nqoi, n_train)
        gp.fit(X, y)

        # Only 1 basis for 2D GP
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases = _create_quadrature_bases(marginals, 10, bkd)

        with pytest.raises(ValueError):
            SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)

    def test_non_separable_kernel_raises_error(self, bkd) -> None:
        """Test that non-separable kernel raises TypeError."""
        np.random.seed(42)
        # Matern 5/2 is NOT separable (uses combined distance in polynomial)
        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), 2, bkd)

        gp = ExactGaussianProcess(kernel, nvars=2, bkd=bkd)
        # Skip hyperparameter optimization for this test
        gp.hyp_list().set_all_inactive()
        X = bkd.array(np.random.rand(2, 5))
        y = bkd.array(np.random.rand(1, 5))  # Shape: (nqoi, n_train)
        gp.fit(X, y)

        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]
        bases = _create_quadrature_bases(marginals, 10, bkd)

        with pytest.raises(TypeError):
            SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)
