"""
Tests for GP statistics moments.

Tests the SeparableKernelIntegralCalculator and GaussianProcessStatistics
classes for computing statistical quantities from fitted GPs.
"""
import unittest
from typing import Generic, Any, List
import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests, slow_test  # noqa: F401
from pyapprox.typing.surrogates.kernels.matern import SquaredExponentialKernel
from pyapprox.typing.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.typing.probability.univariate.uniform import UniformMarginal
from pyapprox.typing.surrogates.sparsegrids.basis_factory import (
    create_basis_factories,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics import (
    SeparableKernelIntegralCalculator,
    GaussianProcessStatistics,
)


def _create_quadrature_bases(
    marginals: List[Any], nquad_points: int, bkd: Backend[Array]
) -> List[Any]:
    """Helper to create quadrature bases from marginals."""
    factories = create_basis_factories(marginals, bkd, "gauss")
    bases = [f.create_basis() for f in factories]
    for b in bases:
        b.set_nterms(nquad_points)
    return bases


class TestSeparableKernelIntegralCalculator(Generic[Array], unittest.TestCase):
    """
    Base test class for SeparableKernelIntegralCalculator.

    Derived classes must implement the bkd() method.
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        self._bkd = self.bkd()
        np.random.seed(42)

        # Create 2D GP with product kernel
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)
        k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)
        self._kernel = k1 * k2

        self._gp = ExactGaussianProcess(
            self._kernel,
            nvars=2,
            bkd=self._bkd,
            nugget=1e-6
        )

        # Training data
        self._n_train = 10
        X_train_np = np.random.rand(2, self._n_train) * 2 - 1  # [-1, 1]^2
        y_train_np = np.sin(np.pi * X_train_np[0, :]) * np.cos(np.pi * X_train_np[1, :])
        y_train_np = y_train_np.reshape(-1, 1)

        self._X_train = self._bkd.array(X_train_np)
        self._y_train = self._bkd.array(y_train_np)

        self._gp.fit(self._X_train, self._y_train)

        # Marginal distributions
        self._marginals = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]

        # Create quadrature bases using sparse grid infrastructure
        self._nquad_points = 20
        bases = _create_quadrature_bases(
            self._marginals, self._nquad_points, self._bkd
        )

        # Create calculator
        self._calc = SeparableKernelIntegralCalculator(
            self._gp, bases, bkd=self._bkd
        )

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_tau_shape(self) -> None:
        """Test tau has correct shape (N,)."""
        tau = self._calc.tau()
        self.assertEqual(tau.shape, (self._n_train,))

    def test_tau_positive(self) -> None:
        """Test tau values are positive (kernel is positive)."""
        tau = self._calc.tau()
        self.assertTrue(self._bkd.all_bool(tau > 0))

    def test_P_shape(self) -> None:
        """Test P has correct shape (N, N)."""
        P = self._calc.P()
        self.assertEqual(P.shape, (self._n_train, self._n_train))

    def test_P_symmetric(self) -> None:
        """Test P is symmetric."""
        P = self._calc.P()
        self._bkd.assert_allclose(P, P.T, rtol=1e-12)

    def test_P_positive_semidefinite(self) -> None:
        """Test P is positive semi-definite."""
        P = self._calc.P()
        eigvals = self._bkd.eigvalsh(P)
        # Allow small negative eigenvalues due to numerical error
        self.assertTrue(self._bkd.all_bool(eigvals > -1e-10))

    def test_u_positive(self) -> None:
        """Test u is positive."""
        u = self._calc.u()
        self.assertGreater(float(self._bkd.to_numpy(u)), 0.0)

    def test_caching(self) -> None:
        """Test that results are cached."""
        tau1 = self._calc.tau()
        tau2 = self._calc.tau()
        # Same object (cached)
        self.assertIs(tau1, tau2)

    def test_conditional_P_subset(self) -> None:
        """Test conditional_P excludes fixed dimensions."""
        # Fix dimension 0, integrate over dimension 1
        index = self._bkd.asarray([0])
        P_cond = self._calc.conditional_P(index)

        # Should only have contribution from dimension 1
        self.assertEqual(P_cond.shape, (self._n_train, self._n_train))
        # P_cond should be positive semi-definite
        eigvals = self._bkd.eigvalsh(P_cond)
        self.assertTrue(self._bkd.all_bool(eigvals > -1e-10))

    def test_conditional_u_subset(self) -> None:
        """Test conditional_u excludes fixed dimensions."""
        # Fix dimension 0, integrate over dimension 1
        index = self._bkd.asarray([0])
        u_cond = self._calc.conditional_u(index)

        # Should be positive
        self.assertGreater(float(self._bkd.to_numpy(u_cond)), 0.0)


class TestGaussianProcessStatistics(Generic[Array], unittest.TestCase):
    """
    Base test class for GaussianProcessStatistics.

    Derived classes must implement the bkd() method.
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        self._bkd = self.bkd()
        np.random.seed(42)

        # Create 2D GP with product kernel
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)
        k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)
        self._kernel = k1 * k2

        self._gp = ExactGaussianProcess(
            self._kernel,
            nvars=2,
            bkd=self._bkd,
            nugget=1e-6
        )

        # Training data
        self._n_train = 10
        X_train_np = np.random.rand(2, self._n_train) * 2 - 1
        y_train_np = np.sin(np.pi * X_train_np[0, :]) * np.cos(np.pi * X_train_np[1, :])
        y_train_np = y_train_np.reshape(-1, 1)

        self._X_train = self._bkd.array(X_train_np)
        self._y_train = self._bkd.array(y_train_np)

        self._gp.fit(self._X_train, self._y_train)

        # Marginal distributions
        self._marginals = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]

        # Create quadrature bases and calculator
        bases = _create_quadrature_bases(self._marginals, 30, self._bkd)
        self._calc = SeparableKernelIntegralCalculator(
            self._gp, bases, bkd=self._bkd
        )
        self._stats = GaussianProcessStatistics(self._gp, self._calc)

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_mean_of_mean_scalar(self) -> None:
        """Test mean_of_mean returns a scalar."""
        eta = self._stats.mean_of_mean()
        # Should be a scalar (0-dim array)
        self.assertEqual(len(eta.shape), 0)

    def test_variance_of_mean_nonnegative(self) -> None:
        """Test Var[mu_f] >= 0."""
        var_mu = self._stats.variance_of_mean()
        self.assertGreaterEqual(float(self._bkd.to_numpy(var_mu)), 0.0)

    def test_mean_of_variance_nonnegative(self) -> None:
        """Test E[gamma_f] >= 0."""
        mean_var = self._stats.mean_of_variance()
        self.assertGreaterEqual(float(self._bkd.to_numpy(mean_var)), 0.0)

    def test_caching(self) -> None:
        """Test that results are cached."""
        eta1 = self._stats.mean_of_mean()
        eta2 = self._stats.mean_of_mean()
        # Same object (cached)
        self.assertIs(eta1, eta2)

    def test_variance_of_variance_nonnegative(self) -> None:
        """Test Var[gamma_f] >= 0."""
        var_var = self._stats.variance_of_variance()
        self.assertGreaterEqual(float(self._bkd.to_numpy(var_var)), 0.0)

    def test_variance_of_variance_scalar(self) -> None:
        """Test variance_of_variance returns a scalar."""
        var_var = self._stats.variance_of_variance()
        # Should be a scalar (0-dim array)
        self.assertEqual(len(var_var.shape), 0)

    def test_variance_of_variance_caching(self) -> None:
        """Test that variance_of_variance results are cached."""
        var_var1 = self._stats.variance_of_variance()
        var_var2 = self._stats.variance_of_variance()
        # Same object (cached)
        self.assertIs(var_var1, var_var2)

    @slow_test
    def test_limit_many_training_points(self) -> None:
        """Test that Var[mu_f] -> 0 as N -> infinity."""
        np.random.seed(123)

        # Create GP with more training points
        n_train_large = 100
        X_train_np = np.random.rand(2, n_train_large) * 2 - 1
        y_train_np = np.sin(np.pi * X_train_np[0, :]) * np.cos(np.pi * X_train_np[1, :])
        y_train_np = y_train_np.reshape(-1, 1)

        X_train = self._bkd.array(X_train_np)
        y_train = self._bkd.array(y_train_np)

        k1 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, self._bkd)
        k2 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, self._bkd)
        kernel = k1 * k2

        gp_large = ExactGaussianProcess(kernel, nvars=2, bkd=self._bkd, nugget=1e-6)
        gp_large.fit(X_train, y_train)

        # Create quadrature bases
        bases_large = _create_quadrature_bases(self._marginals, 30, self._bkd)
        calc_large = SeparableKernelIntegralCalculator(
            gp_large, bases_large, bkd=self._bkd
        )
        stats_large = GaussianProcessStatistics(gp_large, calc_large)

        # Variance of mean should be small with many training points
        var_mu_large = stats_large.variance_of_mean()
        var_mu_small = self._stats.variance_of_mean()

        var_large_val = float(self._bkd.to_numpy(var_mu_large))
        var_small_val = float(self._bkd.to_numpy(var_mu_small))

        # More data should give smaller variance
        # When both are numerically zero, they're effectively equal (which is OK)
        if var_small_val < 1e-12:
            # Original variance already tiny - just check large is also small
            self.assertLess(var_large_val, 1e-10)
        else:
            self.assertLess(var_large_val, var_small_val)


class TestMCComparison(Generic[Array], unittest.TestCase):
    """
    Test Monte Carlo comparison for GP statistics.

    Compares quadrature-based results to Monte Carlo estimates.

    CRITICAL: Test setup must have non-negligible posterior variance at
    quadrature points. Use sparse training data so the GP has uncertainty
    in regions covered by the quadrature rule.
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment with sparse training data."""
        self._bkd = self.bkd()
        np.random.seed(42)

        # Create 1D GP with shorter length scale
        # Shorter length scale = more local correlation = higher posterior variance
        # between training points
        k1 = SquaredExponentialKernel([0.3], (0.1, 10.0), 1, self._bkd)
        self._kernel = k1

        self._gp = ExactGaussianProcess(
            self._kernel,
            nvars=1,
            bkd=self._bkd,
            nugget=1e-6
        )

        # SPARSE training data - only 3 points at boundaries and center
        # With short length scale, regions between points have high uncertainty
        self._n_train = 3
        X_train = self._bkd.array([[-1.0, 0.0, 1.0]])  # Only 3 points
        # Use bkd.sin for backend compatibility
        y_train = self._bkd.reshape(
            self._bkd.sin(3.14159 * X_train[0, :]), (-1, 1)
        )

        self._X_train = X_train
        self._y_train = y_train

        self._gp.fit(self._X_train, self._y_train)

        # Marginal distributions
        self._marginals = [UniformMarginal(-1.0, 1.0, self._bkd)]

        # Create quadrature bases and calculator
        self._nquad = 30
        bases = _create_quadrature_bases(self._marginals, self._nquad, self._bkd)
        self._bases = bases
        self._calc = SeparableKernelIntegralCalculator(
            self._gp, bases, bkd=self._bkd
        )
        self._stats = GaussianProcessStatistics(self._gp, self._calc)

        # Verify posterior variance is non-negligible at quadrature points
        quad_pts, _ = bases[0].quadrature_rule()
        post_std = self._gp.predict_std(quad_pts)
        mean_post_var = float(self._bkd.mean(post_std**2))
        assert mean_post_var > 0.01, (
            f"Posterior variance too small ({mean_post_var:.6f}). "
            "Test setup needs sparser training data."
        )

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    @slow_test
    def test_mc_convergence_mean_of_mean(self) -> None:
        """Compare quadrature E[mu] to MC estimate."""
        np.random.seed(12345)

        # Quadrature estimate
        eta_quad = self._stats.mean_of_mean()

        # MC estimate - generate random samples and convert to backend array
        n_mc = 10000
        X_mc = self._bkd.array(np.random.rand(1, n_mc) * 2 - 1)  # Uniform [-1, 1]

        # Evaluate GP mean at MC samples
        mu_mc = self._gp.predict(X_mc)  # Shape: (n_mc, 1)
        eta_mc = self._bkd.mean(mu_mc)

        # Compare (allow for MC error)
        eta_quad_val = float(self._bkd.to_numpy(eta_quad))
        eta_mc_val = float(self._bkd.to_numpy(eta_mc))

        # MC should be within ~3 standard errors (allowing for randomness)
        # Standard error is O(1/sqrt(n_mc)) ~ 0.01 for n_mc=10000
        self.assertAlmostEqual(eta_quad_val, eta_mc_val, delta=0.05)

    @slow_test
    def test_mc_convergence_variance_of_mean(self) -> None:
        """
        Compare quadrature Var[μ_f] to MC estimate.

        Var[μ_f] = Var[∫ f(z) dF(z)] requires sampling from the GP posterior.

        For each sample f^(r) from the posterior:
        1. Evaluate f^(r) at quadrature points
        2. Compute μ^(r) = Σ_j w_j f^(r)(z_j)
        3. Compute Var[{μ^(r)}] across samples

        GP posterior at points Z has:
        - Mean: m*(Z) = K(Z,X) @ A^{-1} @ y
        - Covariance: C*(Z,Z) = K(Z,Z) - K(Z,X) @ A^{-1} @ K(X,Z)

        Sample via: f = m* + L @ z where L = cholesky(C*), z ~ N(0,I)
        """
        np.random.seed(12345)

        # Quadrature estimate
        var_quad = self._stats.variance_of_mean()

        # Get quadrature points and weights from the bases used by the calculator
        quad_pts, quad_wts = self._bases[0].quadrature_rule()
        # quad_pts shape: (1, nquad), quad_wts shape: (nquad, 1)
        quad_wts = self._bkd.reshape(quad_wts, (-1,))  # (nquad,)
        nquad = quad_pts.shape[1]

        # Compute GP posterior mean and covariance at quadrature points
        X_train = self._gp.data().X()  # (1, n_train)
        y_train = self._gp.data().y()  # (n_train, 1)

        # Prior covariance matrices
        K_qq = self._gp.kernel()(quad_pts, quad_pts)  # (nquad, nquad)
        K_qt = self._gp.kernel()(quad_pts, X_train)   # (nquad, n_train)

        # Posterior mean: m* = K_qt @ A^{-1} @ y
        chol = self._gp.cholesky()
        alpha = chol.solve(y_train)  # A^{-1} @ y, shape (n_train, 1)
        m_star = K_qt @ alpha  # (nquad, 1)
        m_star = self._bkd.reshape(m_star, (-1,))  # (nquad,)

        # Posterior covariance: C* = K_qq - K_qt @ A^{-1} @ K_tq
        A_inv_K_tq = chol.solve(K_qt.T)  # (n_train, nquad)
        C_star = K_qq - K_qt @ A_inv_K_tq  # (nquad, nquad)

        # Add small jitter for numerical stability
        C_star = C_star + 1e-8 * self._bkd.eye(nquad)

        # Cholesky of posterior covariance
        L_star = self._bkd.cholesky(C_star)  # (nquad, nquad)

        # Sample from GP posterior and compute μ^(r) for each sample
        n_samples = 10000
        mu_samples = []

        for _ in range(n_samples):
            # Sample z ~ N(0, I)
            z = self._bkd.array(np.random.randn(nquad))

            # Sample f = m* + L @ z
            f_sample = m_star + L_star @ z  # (nquad,)

            # Compute μ^(r) = Σ_j w_j f^(r)(z_j)
            mu_r = self._bkd.sum(quad_wts * f_sample)
            mu_samples.append(float(self._bkd.to_numpy(mu_r)))

        # Compute variance across samples
        mu_samples_arr = np.array(mu_samples)
        var_mc = np.var(mu_samples_arr)

        # Compare
        var_quad_val = float(self._bkd.to_numpy(var_quad))

        # Allow tolerance for MC error
        # With 10000 samples and variance ~O(0.1), relative error should be ~1-2%
        # Use relative comparison for robustness
        rel_error = abs(var_quad_val - var_mc) / max(var_quad_val, 1e-10)
        self.assertLess(rel_error, 0.1,
            f"Relative error {rel_error:.4f} too large: quad={var_quad_val:.6f}, mc={var_mc:.6f}")

    @slow_test
    def test_mc_convergence_variance_of_variance(self) -> None:
        """
        Compare quadrature Var[γ_f] to MC estimate.

        Var[γ_f] requires sampling from the GP posterior and computing
        variance statistics across samples.

        For each sample f^(r) from the posterior:
        1. Evaluate f^(r) at quadrature points
        2. Compute κ^(r) = Σ_j w_j [f^(r)(z_j)]²
        3. Compute μ^(r) = Σ_j w_j f^(r)(z_j)
        4. Compute γ^(r) = κ^(r) - [μ^(r)]²
        5. Compute Var[{γ^(r)}] across samples
        """
        np.random.seed(12345)

        # Quadrature estimate
        var_var_quad = self._stats.variance_of_variance()

        # Get quadrature points and weights from the bases used by the calculator
        quad_pts, quad_wts = self._bases[0].quadrature_rule()
        # quad_pts shape: (1, nquad), quad_wts shape: (nquad, 1)
        quad_wts = self._bkd.reshape(quad_wts, (-1,))  # (nquad,)
        nquad = quad_pts.shape[1]

        # Compute GP posterior mean and covariance at quadrature points
        X_train = self._gp.data().X()  # (1, n_train)
        y_train = self._gp.data().y()  # (n_train, 1)

        # Prior covariance matrices
        K_qq = self._gp.kernel()(quad_pts, quad_pts)  # (nquad, nquad)
        K_qt = self._gp.kernel()(quad_pts, X_train)   # (nquad, n_train)

        # Posterior mean: m* = K_qt @ A^{-1} @ y
        chol = self._gp.cholesky()
        alpha = chol.solve(y_train)  # A^{-1} @ y, shape (n_train, 1)
        m_star = K_qt @ alpha  # (nquad, 1)
        m_star = self._bkd.reshape(m_star, (-1,))  # (nquad,)

        # Posterior covariance: C* = K_qq - K_qt @ A^{-1} @ K_tq
        A_inv_K_tq = chol.solve(K_qt.T)  # (n_train, nquad)
        C_star = K_qq - K_qt @ A_inv_K_tq  # (nquad, nquad)

        # Add small jitter for numerical stability
        C_star = C_star + 1e-8 * self._bkd.eye(nquad)

        # Cholesky of posterior covariance
        L_star = self._bkd.cholesky(C_star)  # (nquad, nquad)

        # Sample from GP posterior and compute γ^(r) for each sample
        n_samples = 10000
        gamma_samples = []

        for _ in range(n_samples):
            # Sample z ~ N(0, I)
            z = self._bkd.array(np.random.randn(nquad))

            # Sample f = m* + L @ z
            f_sample = m_star + L_star @ z  # (nquad,)

            # Compute κ^(r) = Σ_j w_j [f^(r)(z_j)]²
            kappa_r = self._bkd.sum(quad_wts * f_sample ** 2)

            # Compute μ^(r) = Σ_j w_j f^(r)(z_j)
            mu_r = self._bkd.sum(quad_wts * f_sample)

            # Compute γ^(r) = κ^(r) - [μ^(r)]²
            gamma_r = kappa_r - mu_r ** 2
            gamma_samples.append(float(self._bkd.to_numpy(gamma_r)))

        # Compute variance across samples
        gamma_samples_arr = np.array(gamma_samples)
        var_var_mc = np.var(gamma_samples_arr)

        # Compare
        var_var_quad_val = float(self._bkd.to_numpy(var_var_quad))

        # Allow tolerance for MC error
        # Variance of variance is a fourth-moment statistic, so has higher
        # MC error than variance of mean
        rel_error = abs(var_var_quad_val - var_var_mc) / max(var_var_quad_val, 1e-10)
        self.assertLess(rel_error, 0.2,
            f"Relative error {rel_error:.4f} too large: quad={var_var_quad_val:.6f}, mc={var_var_mc:.6f}")


class TestValidation(Generic[Array], unittest.TestCase):
    """Test validation and error handling."""

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        self._bkd = self.bkd()
        np.random.seed(42)

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_unfitted_gp_raises_error(self) -> None:
        """Test that unfitted GP raises RuntimeError."""
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)
        gp = ExactGaussianProcess(k1, nvars=1, bkd=self._bkd)

        marginals = [UniformMarginal(-1.0, 1.0, self._bkd)]
        bases = _create_quadrature_bases(marginals, 10, self._bkd)

        with self.assertRaises(RuntimeError):
            SeparableKernelIntegralCalculator(gp, bases, bkd=self._bkd)

    def test_wrong_number_of_bases_raises_error(self) -> None:
        """Test that wrong number of quadrature bases raises ValueError."""
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)
        k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)
        kernel = k1 * k2

        gp = ExactGaussianProcess(kernel, nvars=2, bkd=self._bkd)
        X = self._bkd.array(np.random.rand(2, 5))
        y = self._bkd.array(np.random.rand(5, 1))
        gp.fit(X, y)

        # Only 1 basis for 2D GP
        marginals = [UniformMarginal(-1.0, 1.0, self._bkd)]
        bases = _create_quadrature_bases(marginals, 10, self._bkd)

        with self.assertRaises(ValueError):
            SeparableKernelIntegralCalculator(gp, bases, bkd=self._bkd)

    def test_non_separable_kernel_raises_error(self) -> None:
        """Test that non-separable kernel raises TypeError."""
        # 2D kernel that is not a product kernel
        kernel = SquaredExponentialKernel([1.0, 1.0], (0.1, 10.0), 2, self._bkd)

        gp = ExactGaussianProcess(kernel, nvars=2, bkd=self._bkd)
        X = self._bkd.array(np.random.rand(2, 5))
        y = self._bkd.array(np.random.rand(5, 1))
        gp.fit(X, y)

        marginals = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]
        bases = _create_quadrature_bases(marginals, 10, self._bkd)

        with self.assertRaises(TypeError):
            SeparableKernelIntegralCalculator(gp, bases, bkd=self._bkd)


# NumPy backend tests
class TestSeparableKernelIntegralCalculatorNumpy(
    TestSeparableKernelIntegralCalculator[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussianProcessStatisticsNumpy(
    TestGaussianProcessStatistics[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMCComparisonNumpy(TestMCComparison[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestValidationNumpy(TestValidation[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestSeparableKernelIntegralCalculatorTorch(
    TestSeparableKernelIntegralCalculator[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGaussianProcessStatisticsTorch(
    TestGaussianProcessStatistics[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestMCComparisonTorch(TestMCComparison[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestValidationTorch(TestValidation[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
