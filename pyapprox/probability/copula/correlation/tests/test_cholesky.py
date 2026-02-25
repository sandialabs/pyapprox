"""Tests for CholeskyCorrelationParameterization."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.probability.copula.correlation.cholesky import (
    CholeskyCorrelationParameterization,
)
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestCholeskyCorrelation(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        # 3D example: L has 3 free params (L10, L20, L21)
        # Use specific values that give a valid correlation matrix
        self._nvars = 3
        self._chol_lower = self._bkd.asarray([0.5, 0.3, 0.4])
        self._corr_param = CholeskyCorrelationParameterization(
            self._chol_lower, self._nvars, self._bkd
        )

    def test_correlation_matrix_is_symmetric(self) -> None:
        sigma = self._corr_param.correlation_matrix()
        self._bkd.assert_allclose(sigma, sigma.T, rtol=1e-12)

    def test_correlation_matrix_unit_diagonal(self) -> None:
        sigma = self._corr_param.correlation_matrix()
        diag = self._bkd.get_diagonal(sigma)
        self._bkd.assert_allclose(
            diag, self._bkd.ones((self._nvars,)), rtol=1e-12
        )

    def test_correlation_matrix_positive_definite(self) -> None:
        sigma = self._corr_param.correlation_matrix()
        eigenvalues = self._bkd.eigvalsh(sigma)
        # All eigenvalues must be positive
        min_eig = self._bkd.min(eigenvalues)
        self.assertTrue(float(self._bkd.to_numpy(min_eig)) > 0.0)

    def test_log_det_matches_slogdet(self) -> None:
        sigma = self._corr_param.correlation_matrix()
        _sign, expected_log_det = self._bkd.slogdet(sigma)
        actual_log_det = self._corr_param.log_det()
        self._bkd.assert_allclose(
            self._bkd.atleast_1d(actual_log_det),
            self._bkd.atleast_1d(expected_log_det),
            rtol=1e-10,
        )

    def test_quad_form_matches_explicit(self) -> None:
        np.random.seed(42)
        z_np = np.random.randn(self._nvars, 10)
        z = self._bkd.asarray(z_np)

        # Compute explicitly: z^T (Sigma^{-1} - I) z per column
        sigma = self._corr_param.correlation_matrix()
        sigma_inv = self._bkd.inv(sigma)
        sigma_inv_minus_I = sigma_inv - self._bkd.eye(self._nvars)

        expected = self._bkd.zeros((10,))
        for k in range(10):
            zk = z[:, k:k + 1]
            expected[k] = (zk.T @ sigma_inv_minus_I @ zk)[0, 0]

        actual = self._corr_param.quad_form(z)
        self._bkd.assert_allclose(actual, expected, rtol=1e-10)

    def test_sample_transform_covariance(self) -> None:
        np.random.seed(42)
        nsamples = 50000
        eps = self._bkd.asarray(
            np.random.randn(self._nvars, nsamples).astype(np.float64)
        )
        z = self._corr_param.sample_transform(eps)

        # Empirical covariance should approximate Sigma
        z_np = self._bkd.to_numpy(z)
        empirical_cov = np.cov(z_np)
        expected_sigma = self._bkd.to_numpy(
            self._corr_param.correlation_matrix()
        )
        np_bkd = NumpyBkd()
        np_bkd.assert_allclose(empirical_cov, expected_sigma, atol=0.03)

    def test_nparams_correct(self) -> None:
        d = self._nvars
        expected = d * (d - 1) // 2
        self.assertEqual(self._corr_param.nparams(), expected)

    def test_hyp_list_roundtrip(self) -> None:
        hyp_list = self._corr_param.hyp_list()
        original_values = hyp_list.get_active_values()

        # Modify values
        new_values = self._bkd.asarray([0.1, 0.2, 0.3])
        hyp_list.set_active_values(new_values)

        # Verify they changed
        retrieved = hyp_list.get_active_values()
        self._bkd.assert_allclose(retrieved, new_values, rtol=1e-12)

        # Restore and verify
        hyp_list.set_active_values(original_values)
        restored = hyp_list.get_active_values()
        self._bkd.assert_allclose(restored, original_values, rtol=1e-12)

    def test_2d_correlation(self) -> None:
        """Test with a simple 2D correlation matrix (single parameter rho)."""
        rho = 0.6
        # For 2D: L = [[1, 0], [rho, sqrt(1-rho^2)]]
        # Free param is just L_{10} = rho
        chol_lower = self._bkd.asarray([rho])
        corr_param = CholeskyCorrelationParameterization(
            chol_lower, 2, self._bkd
        )
        sigma = corr_param.correlation_matrix()
        expected = self._bkd.asarray([[1.0, rho], [rho, 1.0]])
        self._bkd.assert_allclose(sigma, expected, rtol=1e-12)

    def test_identity_correlation(self) -> None:
        """Test with identity correlation (all free params = 0)."""
        d = 3
        nparams = d * (d - 1) // 2
        chol_lower = self._bkd.zeros((nparams,))
        corr_param = CholeskyCorrelationParameterization(
            chol_lower, d, self._bkd
        )
        sigma = corr_param.correlation_matrix()
        expected = self._bkd.eye(d)
        self._bkd.assert_allclose(sigma, expected, rtol=1e-12)

    def test_quad_form_input_validation(self) -> None:
        z_1d = self._bkd.asarray([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            self._corr_param.quad_form(z_1d)

    def test_wrong_nparams_raises(self) -> None:
        wrong_values = self._bkd.asarray([0.5, 0.3])  # 2 instead of 3
        with self.assertRaises(ValueError):
            CholeskyCorrelationParameterization(wrong_values, 3, self._bkd)


class TestCholeskyCorrelationNumpy(TestCholeskyCorrelation[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCholeskyCorrelationTorch(TestCholeskyCorrelation[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
