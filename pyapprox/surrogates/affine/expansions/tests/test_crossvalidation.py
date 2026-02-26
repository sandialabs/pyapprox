"""Tests for cross-validation functions.

Replicates legacy tests from pyapprox/surrogates/affine/tests/test_crossvalidation.py
with dual-backend support (NumPy + PyTorch).
"""

import numpy as np
import pytest

from pyapprox.surrogates.affine.expansions.crossvalidation import (
    get_cross_validation_rsquared,
    get_random_k_fold_sample_indices,
    leave_many_out_lsq_cross_validation,
    leave_one_out_lsq_cross_validation,
)


class TestCrossValidation:
    """Base test class."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(1)

    def test_k_fold_sample_indices_cover_all_samples(self, bkd) -> None:
        """Fold indices partition all sample indices exactly once."""
        nsamples = 17
        nfolds = 5
        fold_indices = get_random_k_fold_sample_indices(
            nsamples, nfolds, random=True, bkd=bkd
        )
        assert len(fold_indices) == nfolds
        all_indices = bkd.hstack(fold_indices)
        unique_indices = bkd.unique(all_indices)
        assert unique_indices.shape[0] == nsamples

    def test_k_fold_sample_indices_deterministic(self, bkd) -> None:
        """Non-random fold indices are deterministic."""
        nsamples = 12
        nfolds = 4
        fold_indices = get_random_k_fold_sample_indices(
            nsamples, nfolds, random=False, bkd=bkd
        )
        # First fold should contain [0, 4, 8]
        expected_first = bkd.asarray([0, 4, 8], dtype=int)
        bkd.assert_allclose(fold_indices[0], expected_first)

    def test_leave_one_out_lsq_cross_validation(self, bkd) -> None:
        """LOO fast formula matches brute-force leave-one-out."""
        degree = 2
        alpha = 1e-3
        nsamples = 2 * (degree + 1)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        basis_mat = samples.T ** bkd.arange(degree + 1)
        values = bkd.exp(samples).T

        cv_errors, cv_score, coef = leave_one_out_lsq_cross_validation(
            basis_mat, values, alpha, bkd=bkd
        )

        # Brute-force LOO
        true_cv_errors = bkd.zeros(cv_errors.shape)
        for ii in range(nsamples):
            samples_ii = bkd.hstack((samples[:, :ii], samples[:, ii + 1 :]))
            basis_mat_ii = samples_ii.T ** bkd.arange(degree + 1)
            values_ii = bkd.vstack((values[:ii], values[ii + 1 :]))
            coef_ii = bkd.lstsq(
                basis_mat_ii.T @ basis_mat_ii + alpha * bkd.eye(basis_mat.shape[1]),
                basis_mat_ii.T @ values_ii,
            )
            true_cv_errors[ii] = basis_mat[ii] @ coef_ii - values[ii]

        bkd.assert_allclose(cv_errors, true_cv_errors)
        expected_score = bkd.sqrt(bkd.sum(true_cv_errors**2, axis=0) / nsamples)
        bkd.assert_allclose(cv_score, expected_score)

    def test_leave_many_out_lsq_cross_validation(self, bkd) -> None:
        """LMO fast formula matches brute-force leave-many-out."""
        degree = 2
        nsamples = 2 * (degree + 1)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        basis_mat = samples.T ** bkd.arange(degree + 1)
        values = bkd.exp(samples).T * 100
        alpha = 1e-3

        assert nsamples % 2 == 0
        nfolds = nsamples // 3
        fold_sample_indices = get_random_k_fold_sample_indices(
            nsamples, nfolds, bkd=bkd
        )

        cv_errors, cv_score, coef = leave_many_out_lsq_cross_validation(
            basis_mat, values, fold_sample_indices, alpha, bkd=bkd
        )

        # Brute-force LMO
        true_cv_errors = []
        for kk in range(len(fold_sample_indices)):
            K = bkd.ones(nsamples, dtype=bool)
            K[fold_sample_indices[kk]] = False
            basis_mat_kk = basis_mat[K, :]
            gram_mat_kk = (
                basis_mat_kk.T @ basis_mat_kk + bkd.eye(basis_mat_kk.shape[1]) * alpha
            )
            values_kk = basis_mat_kk.T @ values[K, :]
            coef_kk = bkd.lstsq(gram_mat_kk, values_kk)
            true_cv_errors.append(
                basis_mat[fold_sample_indices[kk], :] @ coef_kk
                - values[fold_sample_indices[kk]]
            )

        for ii in range(len(cv_errors)):
            bkd.assert_allclose(cv_errors[ii], true_cv_errors[ii])

        true_cv_score = bkd.sqrt(
            (bkd.stack(true_cv_errors) ** 2).sum(axis=(0, 1)) / nsamples
        )
        bkd.assert_allclose(true_cv_score, cv_score)

    def test_rsquared_coefficient_of_variation(self, bkd) -> None:
        """R-squared computed correctly from CV score."""
        train_vals = bkd.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        # Perfect fit: cv_score = 0 -> R^2 = 1
        cv_score_zero = bkd.asarray(0.0)
        rsq = get_cross_validation_rsquared(cv_score_zero, train_vals, bkd=bkd)
        bkd.assert_allclose(rsq, bkd.asarray(1.0))

        # cv_score = std -> R^2 = 0
        cv_score_std = bkd.std(train_vals)
        rsq = get_cross_validation_rsquared(cv_score_std, train_vals, bkd=bkd)
        bkd.assert_allclose(rsq, bkd.asarray(0.0), atol=1e-14)

    def test_loo_with_precomputed_coef(self, bkd) -> None:
        """LOO with precomputed coefficients matches without."""
        degree = 2
        alpha = 1e-3
        nsamples = 2 * (degree + 1)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        basis_mat = samples.T ** bkd.arange(degree + 1)
        values = bkd.exp(samples).T

        # Without precomputed coef
        cv_errors1, cv_score1, coef1 = leave_one_out_lsq_cross_validation(
            basis_mat, values, alpha, bkd=bkd
        )

        # With precomputed coef
        cv_errors2, cv_score2, coef2 = leave_one_out_lsq_cross_validation(
            basis_mat, values, alpha, coef=coef1, bkd=bkd
        )

        bkd.assert_allclose(cv_errors1, cv_errors2)
        bkd.assert_allclose(cv_score1, cv_score2)
