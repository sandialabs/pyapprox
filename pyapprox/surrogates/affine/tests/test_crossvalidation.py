import unittest

import numpy as np

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.surrogates.univariate.orthopoly import LegendrePolynomial1D
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.basisexp import PolynomialChaosExpansion
from pyapprox.surrogates.affine.linearsystemsolvers import (
    LstSqSolver,
    OMPSolver,
)
from pyapprox.surrogates.affine.multiindex import compute_hyperbolic_indices
from pyapprox.surrogates.affine.crossvalidation import (
    KFoldCrossValidation,
    GridSearchStructureParameterIterator,
    CrossValidationStructureSearch,
    PolynomialDegreeIterator,
    OMPNTermsIterator,
    get_random_k_fold_sample_indices,
    leave_one_out_lsq_cross_validation,
    leave_many_out_lsq_cross_validation,
)


class TestCrossValidation:
    def setUp(self):
        np.random.seed(1)

    def _check_pce_cv(self, nvars):
        bkd = self.get_backend()
        degree = 2
        polys_1d = [LegendrePolynomial1D(backend=bkd) for dd in range(nvars)]
        basis = OrthonormalPolynomialBasis(polys_1d)
        basis.set_indices(
            bkd.asarray(compute_hyperbolic_indices(basis.nvars(), degree, 1.0))
        )
        pce = PolynomialChaosExpansion(
            basis, solver=LstSqSolver(backend=bkd), nqoi=1
        )

        def fun(samples):
            return bkd.sum(samples**2, axis=0)[:, None]

        ntrain_samples = basis.nterms() * 5
        train_samples = bkd.asarray(
            np.random.uniform(-1, 1, (nvars, ntrain_samples))
        )
        train_values = fun(train_samples)
        kcv = KFoldCrossValidation(train_samples, train_values, pce)
        score = kcv.run()
        assert score < 1e-13

    def test_pce_cv(self):
        for nvars in range(1, 3):
            self._check_pce_cv(nvars)

    def _check_pce_cv_search(self, nvars, ntrain_samples):
        bkd = self.get_backend()
        degree = 3
        polys_1d = [LegendrePolynomial1D(backend=bkd) for dd in range(nvars)]
        basis = OrthonormalPolynomialBasis(polys_1d)
        basis.set_hyperbolic_indices(degree, 1.0)
        pce = PolynomialChaosExpansion(
            basis, solver=OMPSolver(backend=bkd, verbosity=2, rtol=0), nqoi=1
        )

        def fun(samples):
            return bkd.sum(samples**3, axis=0)[:, None]

        train_samples = bkd.asarray(
            np.random.uniform(-1, 1, (nvars, ntrain_samples))
        )
        train_values = fun(train_samples)

        kcv = KFoldCrossValidation(train_samples, train_values, pce)
        search1 = PolynomialDegreeIterator([1, 2, 3], [0.1, 1.0])
        search2 = OMPNTermsIterator([2, 3**nvars + 1, np.inf])
        search = GridSearchStructureParameterIterator([search1, search2])
        cv_search = CrossValidationStructureSearch(kcv, search)
        best_structure_params, results, best_idx = cv_search.run()
        self.assertEqual(best_structure_params, ((3, 0.1), 3**nvars + 1))
        assert results[best_idx][0] < 1e-15
        print(pce._solver._termination_flag)
        # if ntrain_samples is too small then algorithm may termite with
        # flag 2
        assert pce._solver._termination_flag == 1

    def test_pce_cv_search(self):
        test_cases = [[1, 20], [2, 100]]
        for test_case in test_cases[-1:]:
            self._check_pce_cv_search(*test_case)

    def test_least_squares_loo_cross_validation(self):
        bkd = self.get_backend()
        degree = 2
        alpha = 1e-3
        nsamples = 2 * (degree + 1)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        basis_mat = samples.T ** bkd.arange(degree + 1)
        values = bkd.exp(samples).T
        cv_errors, cv_score, coef = leave_one_out_lsq_cross_validation(
            basis_mat, values, alpha, bkd=bkd
        )
        true_cv_errors = bkd.empty_like(cv_errors)
        for ii in range(nsamples):
            samples_ii = bkd.hstack((samples[:, :ii], samples[:, ii + 1 :]))
            basis_mat_ii = samples_ii.T ** bkd.arange(degree + 1)
            values_ii = bkd.vstack((values[:ii], values[ii + 1 :]))
            coef_ii = bkd.lstsq(
                basis_mat_ii.T @ basis_mat_ii
                + alpha * bkd.eye(basis_mat.shape[1]),
                basis_mat_ii.T @ values_ii,
            )
            true_cv_errors[ii] = basis_mat[ii] @ coef_ii - values[ii]
        assert bkd.allclose(cv_errors, true_cv_errors)
        assert bkd.allclose(
            cv_score, bkd.sqrt(bkd.sum(true_cv_errors**2, axis=0) / nsamples)
        )

    def test_leave_many_out_lsq_cross_validation(self):
        bkd = self.get_backend()
        degree = 2
        nsamples = 2 * (degree + 1)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        basis_mat = samples.T ** bkd.arange(degree + 1)
        values = bkd.exp(samples).T * 100
        alpha = 1e-3  # ridge regression regularization parameter value

        assert nsamples % 2 == 0
        nfolds = nsamples // 3
        fold_sample_indices = get_random_k_fold_sample_indices(
            nsamples, nfolds
        )
        cv_errors, cv_score, coef = leave_many_out_lsq_cross_validation(
            basis_mat, values, fold_sample_indices, alpha, bkd=bkd
        )
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
            assert bkd.allclose(cv_errors[ii], true_cv_errors[ii])
        true_cv_score = bkd.sqrt(
            (bkd.stack(true_cv_errors) ** 2).sum(axis=(0, 1)) / nsamples
        )
        assert bkd.allclose(true_cv_score, cv_score)


class TestNumpyCrossValidation(TestCrossValidation, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchCrossValidation(TestCrossValidation, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
