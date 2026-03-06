"""
Tests for ConditionalCopulaSAS distribution.
"""

import numpy as np
import pytest

from pyapprox.probability.conditional.copula_sas import ConditionalCopulaSAS
from pyapprox.probability.copula.correlation.cholesky import (
    CholeskyCorrelationParameterization,
)
from pyapprox.probability.copula.gaussian import GaussianCopula
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.probability.univariate.sas_normal import SASNormalMarginal

# TODO: Fix typing issues

def _make_copula_sas(bkd, d=2):
    """Create a ConditionalCopulaSAS for testing."""
    nchol = d * (d - 1) // 2
    chol_vals = bkd.zeros((nchol,))
    corr_param = CholeskyCorrelationParameterization(chol_vals, d, bkd)
    copula = GaussianCopula(corr_param, bkd)
    marginals = [SASNormalMarginal(0.0, 1.0, 0.0, 1.0, bkd) for _ in range(d)]
    return ConditionalCopulaSAS(copula, marginals, bkd)


class TestConditionalCopulaSAS:
    """Tests for ConditionalCopulaSAS."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd) -> None:
        self._bkd = bkd
        self._d = 2
        self._dist = _make_copula_sas(bkd, self._d)

    def test_nvars(self) -> None:
        """Test nvars returns 1 (dummy conditioning)."""
        assert self._dist.nvars() == 1

    def test_nqoi(self) -> None:
        """Test nqoi returns dimension."""
        assert self._dist.nqoi() == self._d

    def test_reparameterize_shape(self) -> None:
        """Test reparameterize output shape."""
        bkd = self._bkd
        nsamples = 10
        x = bkd.zeros((1, nsamples))
        base = bkd.asarray(np.random.randn(self._d, nsamples))
        z = self._dist.reparameterize(x, base)
        assert z.shape == (self._d, nsamples)

    def test_logpdf_shape(self) -> None:
        """Test logpdf output shape."""
        bkd = self._bkd
        nsamples = 10
        x = bkd.zeros((1, nsamples))
        z = bkd.asarray(np.random.randn(self._d, nsamples))
        logpdf = self._dist.logpdf(x, z)
        assert logpdf.shape == (1, nsamples)

    def test_identity_copula_equals_marginal_sum(self) -> None:
        """With identity correlation, logpdf = sum of marginal logpdfs."""
        bkd = self._bkd
        d = self._d
        # Identity correlation (zero off-diagonal Cholesky params)
        dist = _make_copula_sas(bkd, d)

        np.random.seed(42)
        z = bkd.asarray(np.random.randn(d, 20))
        x = bkd.zeros((1, 20))
        logpdf_copula = dist.logpdf(x, z)

        # Sum of marginal logpdfs
        log_sum = bkd.zeros((20,))
        for j in range(d):
            z_j = bkd.reshape(z[j], (1, -1))
            log_sum = log_sum + dist._marginals[j].logpdf(z_j)[0]
        expected = bkd.reshape(log_sum, (1, -1))
        bkd.assert_allclose(logpdf_copula, expected, rtol=1e-10)

    def test_gaussian_reduction(self) -> None:
        """At eps=0, delta=1 with identity correlation,
        matches IndependentJoint Gaussian."""
        bkd = self._bkd
        d = self._d
        from pyapprox.probability.joint.independent import IndependentJoint

        dist = _make_copula_sas(bkd, d)
        gauss_marginals = [GaussianMarginal(0.0, 1.0, bkd) for _ in range(d)]
        gauss_joint = IndependentJoint(gauss_marginals, bkd)

        np.random.seed(42)
        z = bkd.asarray(np.random.randn(d, 20))
        x = bkd.zeros((1, 20))
        logpdf_copula = dist.logpdf(x, z)
        logpdf_gauss = gauss_joint.logpdf(z)
        bkd.assert_allclose(logpdf_copula, logpdf_gauss, rtol=1e-10)

    def test_reparameterize_identity_matches_sas(self) -> None:
        """With identity correlation, reparameterize applies SAS per-dim."""
        bkd = self._bkd
        d = self._d
        dist = _make_copula_sas(bkd, d)

        np.random.seed(42)
        base = bkd.asarray(np.random.randn(d, 15))
        x = bkd.zeros((1, 15))
        z = dist.reparameterize(x, base)

        for j in range(d):
            base_j = bkd.reshape(base[j], (1, -1))
            z_j_expected = dist._marginals[j].reparameterize(base_j)
            bkd.assert_allclose(
                bkd.reshape(z[j], (1, -1)), z_j_expected, rtol=1e-12
            )

    def test_logpdf_finite(self) -> None:
        """Test logpdf produces finite values."""
        bkd = self._bkd
        np.random.seed(42)
        z = bkd.asarray(np.random.randn(self._d, 50))
        x = bkd.zeros((1, 50))
        logpdf = self._dist.logpdf(x, z)
        assert bkd.all_bool(bkd.isfinite(logpdf))

    def test_rvs_shape(self) -> None:
        """Test rvs output shape."""
        bkd = self._bkd
        x = bkd.zeros((1, 100))
        samples = self._dist.rvs(x)
        assert samples.shape == (self._d, 100)

    def test_base_distribution(self) -> None:
        """Test base_distribution returns IndependentJoint of Gaussians."""
        from pyapprox.probability.joint.independent import IndependentJoint

        base = self._dist.base_distribution()
        assert isinstance(base, IndependentJoint)

    def test_nparams(self) -> None:
        """Test nparams includes copula + all marginal params."""
        d = self._d
        n_copula = d * (d - 1) // 2  # Cholesky off-diagonal
        n_marginal = 4 * d  # 4 params per SAS marginal
        assert self._dist.nparams() == n_copula + n_marginal

    def test_repr(self) -> None:
        """Test string representation."""
        repr_str = repr(self._dist)
        assert "ConditionalCopulaSAS" in repr_str

    def test_wrong_marginal_count_raises(self) -> None:
        """Test mismatched marginal count raises ValueError."""
        bkd = self._bkd
        chol_vals = bkd.zeros((1,))
        corr_param = CholeskyCorrelationParameterization(chol_vals, 2, bkd)
        copula = GaussianCopula(corr_param, bkd)
        marginals = [SASNormalMarginal(0.0, 1.0, 0.0, 1.0, bkd)]
        with pytest.raises(ValueError, match="Expected 2 marginals"):
            ConditionalCopulaSAS(copula, marginals, bkd)

    def test_elbo_integration(self) -> None:
        """Test that ConditionalCopulaSAS works with make_single_problem_elbo."""
        bkd = self._bkd
        from pyapprox.inverse.variational.elbo import make_single_problem_elbo
        from pyapprox.probability.joint.independent import IndependentJoint

        d = 2
        dist = _make_copula_sas(bkd, d)
        prior = IndependentJoint(
            [GaussianMarginal(0.0, 1.0, bkd) for _ in range(d)], bkd
        )

        def log_lik(z):
            return -0.5 * bkd.sum(z**2, axis=0, keepdims=True)

        np.random.seed(42)
        base_nodes = bkd.asarray(np.random.randn(d, 50))
        base_weights = bkd.full((1, 50), 1.0 / 50)

        elbo = make_single_problem_elbo(
            dist, log_lik, prior, base_nodes, base_weights, bkd
        )

        params = bkd.zeros((elbo.nvars(), 1))
        neg_elbo = elbo(params)
        assert neg_elbo.shape == (1, 1)
        assert bkd.all_bool(bkd.isfinite(neg_elbo))

    def test_3d_copula(self) -> None:
        """Test with 3-dimensional copula."""
        bkd = self._bkd
        d = 3
        dist = _make_copula_sas(bkd, d)
        assert dist.nqoi() == 3
        assert dist.nparams() == 3 + 12  # 3 off-diag + 4*3 marginal params

        np.random.seed(42)
        z = bkd.asarray(np.random.randn(d, 10))
        x = bkd.zeros((1, 10))
        logpdf = dist.logpdf(x, z)
        assert logpdf.shape == (1, 10)
        assert bkd.all_bool(bkd.isfinite(logpdf))


class TestCholeskyAutograd:
    """Test that CholeskyCorrelationParameterization preserves autograd."""

    def test_cholesky_autograd_gradient(self, torch_bkd) -> None:
        """Verify autograd works through the Cholesky parameterization."""
        # TODO use DerivativeChecker to check gradient
        bkd = torch_bkd
        import torch

        d = 3
        chol_vals = bkd.asarray(np.array([0.3, 0.1, -0.2]))
        corr_param = CholeskyCorrelationParameterization(chol_vals, d, bkd)
        copula = GaussianCopula(corr_param, bkd)

        # Wrap in ConditionalCopulaSAS
        marginals = [
            SASNormalMarginal(0.0, 1.0, 0.0, 1.0, bkd) for _ in range(d)
        ]
        dist = ConditionalCopulaSAS(copula, marginals, bkd)
        dist.hyp_list().set_all_active()

        params = dist.hyp_list().get_active_values().clone().detach()

        def neg_elbo_scalar(p):
            dist.hyp_list().set_active_values(p)
            np.random.seed(0)
            base = bkd.asarray(np.random.randn(d, 20))
            x = bkd.zeros((1, 20))
            z = dist.reparameterize(x, base)
            logpdf = dist.logpdf(x, z)
            return -bkd.sum(logpdf)

        # Use torch.autograd.functional.jacobian
        jac = torch.autograd.functional.jacobian(neg_elbo_scalar, params)
        assert jac.shape == params.shape
        # Gradient should be finite
        assert torch.all(torch.isfinite(jac))
        # Gradient should not be all zeros (params matter)
        assert torch.any(jac != 0.0)

    def test_cholesky_correlation_matrix_autograd(self, torch_bkd) -> None:
        """Verify correlation_matrix is differentiable w.r.t. Cholesky params."""
        # TODO use DerivativeChecker to check gradient
        bkd = torch_bkd
        import torch

        d = 2
        chol_vals = bkd.asarray(np.array([0.5]))
        corr_param = CholeskyCorrelationParameterization(chol_vals, d, bkd)
        corr_param.hyp_list().set_all_active()
        params = corr_param.hyp_list().get_active_values().clone().detach()

        def corr_trace(p):
            corr_param.hyp_list().set_active_values(p)
            R = corr_param.correlation_matrix()
            return bkd.sum(R)

        jac = torch.autograd.functional.jacobian(corr_trace, params)
        assert torch.all(torch.isfinite(jac))

    def test_cholesky_log_det_autograd(self, torch_bkd) -> None:
        """Verify log_det is differentiable w.r.t. Cholesky params."""
        # TODO use DerivativeChecker to check gradient
        bkd = torch_bkd
        import torch

        d = 2
        chol_vals = bkd.asarray(np.array([0.3]))
        corr_param = CholeskyCorrelationParameterization(chol_vals, d, bkd)
        corr_param.hyp_list().set_all_active()
        params = corr_param.hyp_list().get_active_values().clone().detach()

        def log_det_fn(p):
            corr_param.hyp_list().set_active_values(p)
            return corr_param.log_det()

        jac = torch.autograd.functional.jacobian(log_det_fn, params)
        assert torch.all(torch.isfinite(jac))
        # For non-zero chol param, log_det should change
        assert torch.any(jac != 0.0)
