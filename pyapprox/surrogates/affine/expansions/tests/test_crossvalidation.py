"""Tests for cross-validation functions.

Replicates legacy tests from pyapprox/surrogates/affine/tests/test_crossvalidation.py
with dual-backend support (NumPy + PyTorch).
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests  # noqa: F401

from pyapprox.surrogates.affine.expansions.crossvalidation import (
    get_random_k_fold_sample_indices,
    leave_one_out_lsq_cross_validation,
    leave_many_out_lsq_cross_validation,
    get_cross_validation_rsquared,
)


class TestCrossValidation(Generic[Array], unittest.TestCase):
    """Base test class - NOT run directly."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(1)

    def test_k_fold_sample_indices_cover_all_samples(self) -> None:
        """Fold indices partition all sample indices exactly once."""
        bkd = self._bkd
        nsamples = 17
        nfolds = 5
        fold_indices = get_random_k_fold_sample_indices(
            nsamples, nfolds, random=True, bkd=bkd
        )
        self.assertEqual(len(fold_indices), nfolds)
        all_indices = bkd.hstack(fold_indices)
        unique_indices = bkd.unique(all_indices)
        self.assertEqual(unique_indices.shape[0], nsamples)

    def test_k_fold_sample_indices_deterministic(self) -> None:
        """Non-random fold indices are deterministic."""
        bkd = self._bkd
        nsamples = 12
        nfolds = 4
        fold_indices = get_random_k_fold_sample_indices(
            nsamples, nfolds, random=False, bkd=bkd
        )
        # First fold should contain [0, 4, 8]
        expected_first = bkd.asarray([0, 4, 8], dtype=int)
        bkd.assert_allclose(fold_indices[0], expected_first)

    def test_leave_one_out_lsq_cross_validation(self) -> None:
        """LOO fast formula matches brute-force leave-one-out."""
        bkd = self._bkd
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
            samples_ii = bkd.hstack((samples[:, :ii], samples[:, ii + 1:]))
            basis_mat_ii = samples_ii.T ** bkd.arange(degree + 1)
            values_ii = bkd.vstack((values[:ii], values[ii + 1:]))
            coef_ii = bkd.lstsq(
                basis_mat_ii.T @ basis_mat_ii
                + alpha * bkd.eye(basis_mat.shape[1]),
                basis_mat_ii.T @ values_ii,
            )
            true_cv_errors[ii] = basis_mat[ii] @ coef_ii - values[ii]

        bkd.assert_allclose(cv_errors, true_cv_errors)
        expected_score = bkd.sqrt(
            bkd.sum(true_cv_errors**2, axis=0) / nsamples
        )
        bkd.assert_allclose(cv_score, expected_score)

    def test_leave_many_out_lsq_cross_validation(self) -> None:
        """LMO fast formula matches brute-force leave-many-out."""
        bkd = self._bkd
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
                basis_mat_kk.T @ basis_mat_kk
                + bkd.eye(basis_mat_kk.shape[1]) * alpha
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

    def test_rsquared_coefficient_of_variation(self) -> None:
        """R-squared computed correctly from CV score."""
        bkd = self._bkd
        train_vals = bkd.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        # Perfect fit: cv_score = 0 -> R^2 = 1
        cv_score_zero = bkd.asarray(0.0)
        rsq = get_cross_validation_rsquared(cv_score_zero, train_vals, bkd=bkd)
        bkd.assert_allclose(rsq, bkd.asarray(1.0))

        # cv_score = std -> R^2 = 0
        cv_score_std = bkd.std(train_vals)
        rsq = get_cross_validation_rsquared(cv_score_std, train_vals, bkd=bkd)
        bkd.assert_allclose(rsq, bkd.asarray(0.0), atol=1e-14)

    def test_loo_with_precomputed_coef(self) -> None:
        """LOO with precomputed coefficients matches without."""
        bkd = self._bkd
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


class TestCrossValidationNumpy(TestCrossValidation[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCrossValidationTorch(TestCrossValidation[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
