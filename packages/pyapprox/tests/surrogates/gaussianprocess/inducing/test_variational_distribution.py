"""Tests for GaussianVariationalDistribution."""

import numpy as np

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.surrogates.gaussianprocess.inducing.variational_distribution import (
    GaussianVariationalDistribution,
)
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array


class TestGaussianVariationalDistribution:
    """Dual-backend tests for the variational distribution."""

    def test_kl_zero_at_prior(self, bkd):
        """Tier 0: KL = 0 when q = prior (m_tilde=0, L_tilde=I)."""
        M = 5
        qd = GaussianVariationalDistribution(M, bkd)
        kl = qd.kl_divergence_to_prior()
        bkd.assert_allclose(bkd.asarray([kl]), bkd.zeros((1,)), atol=1e-12)

    def test_kl_nonnegative(self, bkd):
        """Tier 0: KL is always non-negative."""
        np.random.seed(42)
        M = 4
        m_init = bkd.array(np.random.randn(M))
        L_raw = np.random.randn(M, M)
        L_init = bkd.array(np.tril(L_raw))
        qd = GaussianVariationalDistribution(M, bkd, m_init, L_init)
        kl = qd.kl_divergence_to_prior()
        assert float(bkd.to_numpy(bkd.asarray([kl]))[0]) >= -1e-12

    def test_kl_matches_manual_formula(self, bkd):
        """Tier 0: KL matches manual computation for known values."""
        M = 3
        m = bkd.array([1.0, -0.5, 0.3])
        L = bkd.array([[2.0, 0.0, 0.0], [0.5, 1.5, 0.0], [-0.3, 0.2, 0.8]])
        qd = GaussianVariationalDistribution(M, bkd, m, L)

        m_np = np.array([1.0, -0.5, 0.3])
        L_np = np.array([[2.0, 0.0, 0.0], [0.5, 1.5, 0.0], [-0.3, 0.2, 0.8]])
        S_np = L_np @ L_np.T
        expected = 0.5 * (
            np.dot(m_np, m_np) + np.trace(S_np) - M - np.log(np.linalg.det(S_np))
        )
        bkd.assert_allclose(
            bkd.asarray([qd.kl_divergence_to_prior()]),
            bkd.asarray([expected]),
            rtol=1e-12,
        )

    def test_unwhitened_mean(self, bkd):
        M = 3
        m_tilde = bkd.array([1.0, 2.0, 3.0])
        qd = GaussianVariationalDistribution(M, bkd, m_tilde)

        L_uu = bkd.array([[2.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.3, 0.7]])
        m = qd.mean(L_uu)
        expected = bkd.dot(L_uu, m_tilde)
        bkd.assert_allclose(m, expected, rtol=1e-12)

    def test_unwhitened_cholesky(self, bkd):
        M = 3
        L_tilde = bkd.array([[1.5, 0.0, 0.0], [0.2, 0.8, 0.0], [0.1, 0.3, 0.6]])
        qd = GaussianVariationalDistribution(M, bkd, chol_init=L_tilde)

        L_uu = bkd.array([[2.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.3, 0.7]])
        L = qd.cholesky(L_uu)
        expected = bkd.dot(L_uu, L_tilde)
        bkd.assert_allclose(L, expected, rtol=1e-12)

    def test_hyp_list_nparams(self, bkd):
        M = 4
        qd = GaussianVariationalDistribution(M, bkd)
        # M mean params + M*(M+1)/2 Cholesky params
        expected = M + M * (M + 1) // 2
        assert qd.hyp_list().nparams() == expected

    def test_prior_sampling_moments(self, bkd):
        """Tier 0: Empirical moments of prior samples match p(u) = N(0, K_uu)."""
        np.random.seed(123)
        M = 3
        qd = GaussianVariationalDistribution(M, bkd)

        A = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
        L_uu = bkd.array(np.linalg.cholesky(A))

        S = 50000
        eps = bkd.array(np.random.randn(S, M))
        samples = qd.sample(L_uu, S, eps=eps)
        assert samples.shape == (S, M)

        emp_mean = bkd.mean(samples, axis=0)
        emp_cov = bkd.dot(samples.T, samples) / S

        bkd.assert_allclose(emp_mean, bkd.zeros((M,)), atol=0.05)
        bkd.assert_allclose(emp_cov, bkd.array(A), atol=0.1)


class TestGaussianVariationalDistributionAutograd:
    """The autodiff KL gradient, validated with DerivativeChecker."""

    def test_kl_gradient_via_derivative_checker(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """Tier 0: KL gradient matches finite-difference via DerivativeChecker."""
        bkd = autodiff_bkd
        M = 3
        m_init = bkd.array([0.5, -0.3, 0.1])
        L_init = bkd.array([[1.0, 0.0, 0.0], [0.2, 0.8, 0.0], [0.1, 0.1, 0.5]])
        qd = GaussianVariationalDistribution(M, bkd, m_init, L_init)

        def kl(params_flat: Array) -> Array:
            qd.hyp_list().set_active_values(params_flat)
            return bkd.reshape(qd.kl_divergence_to_prior(), (1,))

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=qd.hyp_list().nactive_params(),
            fun=lambda params: bkd.reshape(kl(bkd.flatten(params)), (1, 1)),
            jacobian=lambda params: bkd.jacobian(kl, bkd.flatten(params)),
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)

        params = qd.hyp_list().get_active_values()
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        errors = checker.check_derivatives(
            params[:, None],
            fd_eps=fd_eps,
            relative=True,
            verbosity=0,
        )
        jac_error = errors[0]
        ratio = float(checker.error_ratio(jac_error))
        assert ratio <= 1e-6, f"KL gradient error ratio {ratio:.2e} > 1e-6"
