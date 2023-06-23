import unittest
import numpy as np
from scipy import stats

from pyapprox.util.utilities import (
    cartesian_product, outer_product, evaluate_quadratic_form,
    leave_many_out_lsq_cross_validation, leave_one_out_lsq_cross_validation,
    get_random_k_fold_sample_indices,
    get_cross_validation_rsquared_coefficient_of_variation,
    integrate_using_univariate_gauss_legendre_quadrature_unbounded,
    split_indices
)


class TestUtilities(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_cartesian_product(self):
        # test when num elems = 1
        s1 = np.arange(0, 3)
        s2 = np.arange(3, 5)

        sets = np.array([[0, 3], [1, 3], [2, 3], [0, 4],
                         [1, 4], [2, 4]], np.int64)
        output_sets = cartesian_product([s1, s2], 1)
        assert np.array_equal(output_sets.T, sets)

        # # test when num elems > 1
        # s1 = np.arange( 0, 6 )
        # s2 = np.arange( 6, 10 )

        # sets = np.array( [[ 0, 1, 6, 7], [ 2, 3, 6, 7],
        #                   [ 4, 5, 6, 7], [ 0, 1, 8, 9],
        #                   [ 2, 3, 8, 9], [ 4, 5, 8, 9]], np.int64 )
        # output_sets = cartesian_product( [s1,s2], 2 )
        # assert np.array_equal( output_sets.T, sets )

    def test_outer_product(self):
        s1 = np.arange(0, 3)
        s2 = np.arange(3, 5)

        test_vals = np.array([0., 3., 6., 0., 4., 8.])
        output = outer_product([s1, s2])
        assert np.allclose(test_vals, output)

        output = outer_product([s1])
        assert np.allclose(output, s1)

    def test_evaluate_quadratic_form(self):
        nvars, nsamples = 3, 10
        A = np.random.normal(0, 1, nvars)
        A = A.T.dot(A)
        samples = np.random.uniform(0, 1, (nvars, nsamples))
        values1 = evaluate_quadratic_form(A, samples)

        values2 = np.zeros(samples.shape[1])
        for ii in range(samples.shape[1]):
            values2[ii] = samples[:, ii:ii+1].T.dot(A).dot(samples[:, ii:ii+1])

        assert np.allclose(values1, values2)

    def test_least_squares_loo_cross_validation(self):
        degree = 2
        alpha = 1e-3
        nsamples = 2*(degree+1)
        samples = np.random.uniform(-1, 1, (1, nsamples))
        basis_mat = samples.T**np.arange(degree+1)
        values = np.exp(samples).T
        cv_errors, cv_score, coef = leave_one_out_lsq_cross_validation(
            basis_mat, values, alpha)
        true_cv_errors = np.empty_like(cv_errors)
        for ii in range(nsamples):
            samples_ii = np.hstack((samples[:, :ii], samples[:, ii+1:]))
            basis_mat_ii = samples_ii.T**np.arange(degree+1)
            values_ii = np.vstack((values[:ii], values[ii+1:]))
            coef_ii = np.linalg.lstsq(
                basis_mat_ii.T.dot(basis_mat_ii) +
                alpha*np.eye(basis_mat.shape[1]
                             ), basis_mat_ii.T.dot(values_ii),
                rcond=None)[0]
            true_cv_errors[ii] = (basis_mat[ii].dot(coef_ii)-values[ii])
        assert np.allclose(cv_errors, true_cv_errors)
        assert np.allclose(
            cv_score, np.sqrt(np.sum(true_cv_errors**2, axis=0)/nsamples))

    def test_leave_many_out_lsq_cross_validation(self):
        degree = 2
        nsamples = 2*(degree+1)
        samples = np.random.uniform(-1, 1, (1, nsamples))
        basis_mat = samples.T**np.arange(degree+1)
        values = np.exp(samples).T*100
        alpha = 1e-3  # ridge regression regularization parameter value

        assert nsamples % 2 == 0
        nfolds = nsamples//3
        fold_sample_indices = get_random_k_fold_sample_indices(
            nsamples, nfolds)
        cv_errors, cv_score, coef = leave_many_out_lsq_cross_validation(
            basis_mat, values, fold_sample_indices, alpha)

        true_cv_errors = np.empty_like(cv_errors)
        for kk in range(len(fold_sample_indices)):
            K = np.ones(nsamples, dtype=bool)
            K[fold_sample_indices[kk]] = False
            basis_mat_kk = basis_mat[K, :]
            gram_mat_kk = basis_mat_kk.T.dot(basis_mat_kk) + np.eye(
                basis_mat_kk.shape[1])*alpha
            values_kk = basis_mat_kk.T.dot(values[K, :])
            coef_kk = np.linalg.lstsq(gram_mat_kk, values_kk, rcond=None)[0]
            true_cv_errors[kk] = basis_mat[fold_sample_indices[kk], :].dot(
                coef_kk)-values[fold_sample_indices[kk]]
        # print(cv_errors, true_cv_errors)
        assert np.allclose(cv_errors, true_cv_errors)
        true_cv_score = np.sqrt((true_cv_errors**2).sum(axis=(0, 1))/nsamples)
        assert np.allclose(true_cv_score, cv_score)

        rsq = get_cross_validation_rsquared_coefficient_of_variation(
            cv_score, values)

        print(rsq)

    def test_integrate_using_univariate_gauss_legendre_quadrature_unbounded(
            self):

        # unbounded
        rv = stats.norm(0, 1)

        def integrand(x):
            return rv.pdf(x)[:, None]
        lb, ub = rv.interval(1)
        nquad_samples = 100
        res = integrate_using_univariate_gauss_legendre_quadrature_unbounded(
            integrand, lb, ub, nquad_samples, interval_size=2)
        assert np.allclose(res, 1)

        # left bounded
        rv = stats.gamma(1)

        def integrand(x):
            return rv.pdf(x)[:, None]
        lb, ub = rv.interval(1)
        nquad_samples = 100
        res = integrate_using_univariate_gauss_legendre_quadrature_unbounded(
            integrand, lb, ub, nquad_samples, interval_size=2)
        assert np.allclose(res, 1)

        # bounded
        rv = stats.beta(20, 10, -2, 5)

        def integrand(x):
            return rv.pdf(x)[:, None]
        lb, ub = rv.interval(1)
        nquad_samples = 100
        res = integrate_using_univariate_gauss_legendre_quadrature_unbounded(
            integrand, lb, ub, nquad_samples, interval_size=2)
        assert np.allclose(res, 1)

        # multiple qoi
        rv = stats.norm(2, 3)

        def integrand(x):
            return rv.pdf(x)[:, None]*x[:, None]**np.arange(3)[None, :]
        lb, ub = rv.interval(1)
        nquad_samples = 100
        res = integrate_using_univariate_gauss_legendre_quadrature_unbounded(
            integrand, lb, ub, nquad_samples, interval_size=2)
        assert np.allclose(res, [1, 2, 3**2+2**2])

    def _check_split_indices(self, nelems, nsplits):
        indices = split_indices(nelems, nsplits)
        split_array = np.array_split(np.arange(nelems), nsplits)
        true_indices = np.hstack([0]+[[a[-1]+1] for a in split_array])
        assert np.allclose(true_indices, indices)

    def test_split_indices(self):
        test_cases = [[10, 3], [6, 3], [3, 3]]
        for test_case in test_cases:
            self._check_split_indices(*test_case)


if __name__ == "__main__":
    utilities_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestUtilities)
    unittest.TextTestRunner(verbosity=2).run(utilities_test_suite)
