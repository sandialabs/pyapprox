"""Tests for CholeskyCorrelationParameterization."""

import numpy as np
import pytest

from pyapprox.probability.copula.correlation.cholesky import (
    CholeskyCorrelationParameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd


class TestCholeskyCorrelation:

    def _make_corr_param(self, bkd, nvars=3):
        # 3D example: L has 3 free params (L10, L20, L21)
        # Use specific values that give a valid correlation matrix
        chol_lower = bkd.asarray([0.5, 0.3, 0.4])
        return CholeskyCorrelationParameterization(chol_lower, nvars, bkd)

    def test_correlation_matrix_is_symmetric(self, bkd) -> None:
        corr_param = self._make_corr_param(bkd)
        sigma = corr_param.correlation_matrix()
        bkd.assert_allclose(sigma, sigma.T, rtol=1e-12)

    def test_correlation_matrix_unit_diagonal(self, bkd) -> None:
        nvars = 3
        corr_param = self._make_corr_param(bkd, nvars)
        sigma = corr_param.correlation_matrix()
        diag = bkd.get_diagonal(sigma)
        bkd.assert_allclose(diag, bkd.ones((nvars,)), rtol=1e-12)

    def test_correlation_matrix_positive_definite(self, bkd) -> None:
        corr_param = self._make_corr_param(bkd)
        sigma = corr_param.correlation_matrix()
        eigenvalues = bkd.eigvalsh(sigma)
        # All eigenvalues must be positive
        min_eig = bkd.min(eigenvalues)
        assert float(bkd.to_numpy(min_eig)) > 0.0

    def test_log_det_matches_slogdet(self, bkd) -> None:
        corr_param = self._make_corr_param(bkd)
        sigma = corr_param.correlation_matrix()
        _sign, expected_log_det = bkd.slogdet(sigma)
        actual_log_det = corr_param.log_det()
        bkd.assert_allclose(
            bkd.atleast_1d(actual_log_det),
            bkd.atleast_1d(expected_log_det),
            rtol=1e-10,
        )

    def test_quad_form_matches_explicit(self, bkd) -> None:
        nvars = 3
        corr_param = self._make_corr_param(bkd, nvars)
        np.random.seed(42)
        z_np = np.random.randn(nvars, 10)
        z = bkd.asarray(z_np)

        # Compute explicitly: z^T (Sigma^{-1} - I) z per column
        sigma = corr_param.correlation_matrix()
        sigma_inv = bkd.inv(sigma)
        sigma_inv_minus_I = sigma_inv - bkd.eye(nvars)

        expected = bkd.zeros((10,))
        for k in range(10):
            zk = z[:, k : k + 1]
            expected[k] = (zk.T @ sigma_inv_minus_I @ zk)[0, 0]

        actual = corr_param.quad_form(z)
        bkd.assert_allclose(actual, expected, rtol=1e-10)

    def test_sample_transform_covariance(self, bkd) -> None:
        nvars = 3
        corr_param = self._make_corr_param(bkd, nvars)
        np.random.seed(42)
        nsamples = 50000
        eps = bkd.asarray(
            np.random.randn(nvars, nsamples).astype(np.float64)
        )
        z = corr_param.sample_transform(eps)

        # Empirical covariance should approximate Sigma
        z_np = bkd.to_numpy(z)
        empirical_cov = np.cov(z_np)
        expected_sigma = bkd.to_numpy(corr_param.correlation_matrix())
        np_bkd = NumpyBkd()
        np_bkd.assert_allclose(empirical_cov, expected_sigma, atol=0.03)

    def test_nparams_correct(self, bkd) -> None:
        nvars = 3
        corr_param = self._make_corr_param(bkd, nvars)
        expected = nvars * (nvars - 1) // 2
        assert corr_param.nparams() == expected

    def test_hyp_list_roundtrip(self, bkd) -> None:
        corr_param = self._make_corr_param(bkd)
        hyp_list = corr_param.hyp_list()
        original_values = hyp_list.get_active_values()

        # Modify values
        new_values = bkd.asarray([0.1, 0.2, 0.3])
        hyp_list.set_active_values(new_values)

        # Verify they changed
        retrieved = hyp_list.get_active_values()
        bkd.assert_allclose(retrieved, new_values, rtol=1e-12)

        # Restore and verify
        hyp_list.set_active_values(original_values)
        restored = hyp_list.get_active_values()
        bkd.assert_allclose(restored, original_values, rtol=1e-12)

    def test_2d_correlation(self, bkd) -> None:
        """Test with a simple 2D correlation matrix (single parameter rho)."""
        rho = 0.6
        # For 2D: L = [[1, 0], [rho, sqrt(1-rho^2)]]
        # Free param is just L_{10} = rho
        chol_lower = bkd.asarray([rho])
        corr_param = CholeskyCorrelationParameterization(chol_lower, 2, bkd)
        sigma = corr_param.correlation_matrix()
        expected = bkd.asarray([[1.0, rho], [rho, 1.0]])
        bkd.assert_allclose(sigma, expected, rtol=1e-12)

    def test_identity_correlation(self, bkd) -> None:
        """Test with identity correlation (all free params = 0)."""
        d = 3
        nparams = d * (d - 1) // 2
        chol_lower = bkd.zeros((nparams,))
        corr_param = CholeskyCorrelationParameterization(chol_lower, d, bkd)
        sigma = corr_param.correlation_matrix()
        expected = bkd.eye(d)
        bkd.assert_allclose(sigma, expected, rtol=1e-12)

    def test_quad_form_input_validation(self, bkd) -> None:
        corr_param = self._make_corr_param(bkd)
        z_1d = bkd.asarray([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            corr_param.quad_form(z_1d)

    def test_wrong_nparams_raises(self, bkd) -> None:
        wrong_values = bkd.asarray([0.5, 0.3])  # 2 instead of 3
        with pytest.raises(ValueError):
            CholeskyCorrelationParameterization(wrong_values, 3, bkd)
