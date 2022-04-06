import unittest
import numpy as np
from scipy.special import binom, factorial

from pyapprox.surrogates.interp.indexing import (
    compute_hyperbolic_indices, argsort_indices_leixographically
)
from pyapprox.surrogates.interp.monomial import monomial_basis_matrix
from pyapprox.surrogates.interp.manipulate_polynomials import (
    multiply_multivariate_polynomials, compress_and_sort_polynomial,
    multinomial_coefficient, multinomial_coeffs_of_power_of_nd_linear_monomial,
    multinomial_coefficients, coeffs_of_power_of_nd_linear_monomial,
    group_like_terms, add_polynomials, coeffs_of_power_of_monomial,
    substitute_polynomials_for_variables_in_single_basis_term,
    substitute_polynomials_for_variables_in_another_polynomial
)


class TestManipulatePolynomials(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_multiply_multivariate_polynomials(self):
        num_vars = 2
        degree1 = 1
        degree2 = 2

        indices1 = compute_hyperbolic_indices(num_vars, degree1, 1.0)
        coeffs1 = np.ones((indices1.shape[1], 1), dtype=float)
        indices2 = compute_hyperbolic_indices(num_vars, degree2, 1.0)
        coeffs2 = 2.0*np.ones((indices2.shape[1], 1), dtype=float)

        indices, coeffs = multiply_multivariate_polynomials(
            indices1, coeffs1, indices2, coeffs2)

        samples = np.random.uniform(-1, 1, (num_vars, indices.shape[1]*3))
        values = monomial_basis_matrix(indices1, samples).dot(coeffs1) * \
            monomial_basis_matrix(indices2, samples).dot(coeffs2)

        true_indices = compute_hyperbolic_indices(
            num_vars, degree1+degree2, 1.0)
        basis_mat = monomial_basis_matrix(true_indices, samples)
        true_coeffs = np.linalg.lstsq(basis_mat, values, rcond=None)[0]
        true_indices, true_coeffs = compress_and_sort_polynomial(
            true_coeffs, true_indices)
        indices, coeffs = compress_and_sort_polynomial(coeffs, indices)
        assert np.allclose(true_indices, indices)
        assert np.allclose(true_coeffs, coeffs)

    def test_multinomial_coefficient(self):
        index = np.array([1, 3, 2])
        coef = multinomial_coefficient(index)
        level = index.sum()
        denom = np.prod(factorial(index))
        true_coef = factorial(level)/denom

    def test_multinomial_coefficients(self):
        num_vars = 2
        degree = 5
        coeffs, indices = multinomial_coeffs_of_power_of_nd_linear_monomial(
            num_vars, degree)

        true_coeffs = np.empty(coeffs.shape[0], float)
        for i in range(0, degree+1):
            true_coeffs[i] = binom(degree, i)
        assert true_coeffs.shape[0] == coeffs.shape[0]
        coeffs = coeffs[argsort_indices_leixographically(indices)]
        assert np.allclose(coeffs, true_coeffs)

        num_vars = 3
        degree = 3
        coeffs, indices = multinomial_coeffs_of_power_of_nd_linear_monomial(
            num_vars, degree)
        coeffs = multinomial_coefficients(indices)
        coeffs = coeffs[argsort_indices_leixographically(indices)]

        true_coeffs = np.array([1, 3, 3, 1, 3, 6, 3, 3, 3, 1])
        assert np.allclose(coeffs, true_coeffs)

    def test_coeffs_of_power_of_nd_linear_monomial(self):
        num_vars = 3
        degree = 2
        linear_coeffs = [2., 3., 4.]
        coeffs, indices = coeffs_of_power_of_nd_linear_monomial(
            num_vars, degree, linear_coeffs)
        sorted_idx = argsort_indices_leixographically(indices)
        true_coeffs = [
            linear_coeffs[2]**2, 2*linear_coeffs[1]*linear_coeffs[2],
            linear_coeffs[1]**2, 2 *
            linear_coeffs[0]*linear_coeffs[2],
            2*linear_coeffs[0]*linear_coeffs[1], linear_coeffs[0]**2]
        assert np.allclose(true_coeffs, coeffs[sorted_idx])

    def test_group_like_terms(self):
        num_vars = 2
        degree = 2

        # define two set of indices that have a non-empty intersection
        indices1 = compute_hyperbolic_indices(num_vars, degree, 1.0)
        indices2 = compute_hyperbolic_indices(num_vars, degree-1, 1.0)
        num_indices1 = indices1.shape[1]
        coeffs = np.arange(num_indices1+indices2.shape[1])
        indices1 = np.hstack((indices1, indices2))

        # make it so coefficients increase by 1 with lexiographical order of
        # combined indices
        indices = indices1[:, argsort_indices_leixographically(indices1)]
        indices, coeffs = group_like_terms(coeffs, indices)

        # Check that only unique indices remain
        assert indices.shape[1] == num_indices1
        # print_indices(indices,num_vars)
        true_indices = np.asarray(
            [[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]]).T
        sorted_idx = argsort_indices_leixographically(indices)
        assert np.allclose(true_indices, indices[:, sorted_idx])

        # check that the coefficients of the unique indices are the sum of
        # all original common indices
        true_coeffs = [1, 5, 9, 6, 7, 8]
        assert np.allclose(coeffs[sorted_idx][:, 0], true_coeffs)

    def test_add_polynomials(self):
        num_vars = 2
        degree = 2

        # define two set of indices that have a non-empty intersection
        indices1 = compute_hyperbolic_indices(num_vars, degree, 1.0)
        indices1 = indices1[:, argsort_indices_leixographically(indices1)]
        coeffs1 = np.arange(indices1.shape[1])[:, np.newaxis]
        indices2 = compute_hyperbolic_indices(num_vars, degree-1, 1.0)
        indices2 = indices2[:, argsort_indices_leixographically(indices2)]
        coeffs2 = np.arange(indices2.shape[1])[:, np.newaxis]

        indices, coeffs = add_polynomials(
            [indices2, indices1], [coeffs2, coeffs1])

        # check that the coefficients of the new polynomial are the union
        # of the original polynomials
        true_indices = np.asarray(
            [[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]]).T
        sorted_idx = argsort_indices_leixographically(indices)
        assert np.allclose(true_indices, indices[:, sorted_idx])

        # check that the coefficients of the new polynomials are the sum of
        # all original polynomials
        true_coeffs = np.asarray([[0, 2, 4, 3, 4, 5]]).T
        assert np.allclose(coeffs[sorted_idx], true_coeffs)

        num_vars = 2
        degree = 2

        # define two set of indices that have a non-empty intersection
        indices3 = compute_hyperbolic_indices(num_vars, degree+1, 1.0)
        indices3 = indices3[:, argsort_indices_leixographically(indices3)]
        coeffs3 = np.arange(indices3.shape[1])[:, np.newaxis]

        indices, coeffs = add_polynomials(
            [indices2, indices1, indices3], [coeffs2, coeffs1, coeffs3])

        # check that the coefficients of the new polynomial are the union
        # of the original polynomials
        true_indices = np.asarray(
            [[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0], [0, 3],
             [1, 2], [2, 1], [3, 0]]).T
        sorted_idx = argsort_indices_leixographically(indices)
        assert np.allclose(true_indices, indices[:, sorted_idx])

        # check that the coefficients of the new polynomials are the sum of
        # all original polynomials
        true_coeffs = np.asarray([[0, 3, 6, 6, 8, 10, 6, 7, 8, 9]]).T
        assert np.allclose(coeffs[sorted_idx], true_coeffs)

    def test_coeffs_of_power_of_monomial(self):
        num_vars, degree, power = 1, 2, 3
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        coeffs = np.ones((indices.shape[1], 1))
        new_indices, new_coeffs = coeffs_of_power_of_monomial(
            indices, coeffs, power)
        assert new_indices.shape[1] == 10
        true_indices = np.asarray(
            [[0], [1], [2], [2], [3], [4], [3], [4], [5], [6]]).T
        true_coeffs = np.asarray([[1, 3, 3, 3, 6, 3, 1, 3, 3, 1]]).T
        # must include coefficient in sort because when there are mulitple
        # entries with same index, argsort can return different orders
        # if initial orders of indices is different.
        # new_sorted_idx = argsort_indices_leixographically(
        #    np.hstack([new_indices.T,new_coeffs]).T)
        # true_sorted_idx = argsort_indices_leixographically(
        #    np.hstack([true_indices.T,true_coeffs]).T)
        # alternatively just group like terms before sort
        new_indices, new_coeffs = group_like_terms(new_coeffs, new_indices)
        true_indices, true_coeffs = group_like_terms(true_coeffs, true_indices)
        new_sorted_idx = argsort_indices_leixographically(new_indices)
        true_sorted_idx = argsort_indices_leixographically(true_indices)

        assert np.allclose(
            true_indices[:, true_sorted_idx], new_indices[:, new_sorted_idx])
        assert np.allclose(
            true_coeffs[true_sorted_idx], new_coeffs[new_sorted_idx])

        num_vars, degree, power = 2, 1, 2
        indices = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]).T
        coeffs = np.ones((indices.shape[1], 1))
        new_indices, new_coeffs = coeffs_of_power_of_monomial(
            indices, coeffs, power)
        true_indices = np.asarray(
            [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [1, 1], [2, 1], [0, 2],
             [1, 2], [2, 2]]).T
        true_coeffs = np.asarray([[1, 2, 2, 2, 1, 2, 2, 1, 2, 1]]).T
        new_indices, new_coeffs = group_like_terms(new_coeffs, new_indices)
        true_indices, true_coeffs = group_like_terms(true_coeffs, true_indices)
        new_sorted_idx = argsort_indices_leixographically(new_indices)
        true_sorted_idx = argsort_indices_leixographically(true_indices)
        assert np.allclose(
            true_indices[:, true_sorted_idx], new_indices[:, new_sorted_idx])
        assert np.allclose(
            true_coeffs[true_sorted_idx], new_coeffs[new_sorted_idx])

        num_vars, degree, power = 2, 1, 2
        indices = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]).T
        coeffs = np.ones((indices.shape[1], 2))
        coeffs[:, 1] = 2
        new_indices, new_coeffs = coeffs_of_power_of_monomial(
            indices, coeffs, power)
        true_indices = np.asarray(
            [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [1, 1], [2, 1], [0, 2],
             [1, 2], [2, 2]]).T
        true_coeffs = np.asarray([[1, 2, 2, 2, 1, 2, 2, 1, 2, 1]]).T
        new_indices, new_coeffs = group_like_terms(new_coeffs, new_indices)
        true_indices, true_coeffs = group_like_terms(true_coeffs, true_indices)
        new_sorted_idx = argsort_indices_leixographically(new_indices)
        true_sorted_idx = argsort_indices_leixographically(true_indices)
        assert np.allclose(
            true_indices[:, true_sorted_idx], new_indices[:, new_sorted_idx])
        # print(new_coeffs[new_sorted_idx, 1], true_coeffs[true_sorted_idx, 0])
        assert np.allclose(
            true_coeffs[true_sorted_idx, 0], new_coeffs[new_sorted_idx, 0])
        assert np.allclose(
            true_coeffs[true_sorted_idx, 0]*4, new_coeffs[new_sorted_idx, 1])

    def test_substitute_polynomial_for_variables_in_single_basis_term(self):
        """
        Substitute
          y1 = (1+x1+x2+x1*x2)
          y2 = (2+2*x1+2*x1*x3)
        into
          y3 = y1*x4**3*y2    (test1)

        Global ordering of variables in y3
        [y1,x4,y2] = [x1,x2,x4,x1,x3]
        Only x4ant unique variables so reduce to
        [x1,x2,x4,x3]
        """
        def y1(x):
            x1, x2 = x[:2, :]
            return 1+x1+x2+x1*x2

        def y2(x):
            x1, x3 = x[[0, 2], :]
            return 2+2*x1+2*x1*x3

        def y3(x):
            x4 = x[3, :]
            y1, y2 = x[4:, :]
            return y1**2*x4**3*y2

        global_var_idx = [np.array([0, 1]), np.array([0, 2])]
        indices_in = [np.array([[0, 0], [1, 0], [0, 1], [1, 1]]).T,
                      np.array([[0, 0], [1, 0], [1, 1]]).T]
        coeffs_in = [np.ones((indices_in[0].shape[1], 1)),
                     2*np.ones((indices_in[1].shape[1], 1))]

        basis_index = np.array([[2, 3, 1]]).T
        basis_coeff = np.array([[1]])
        var_indices = np.array([0, 2])
        new_indices, new_coeffs = \
            substitute_polynomials_for_variables_in_single_basis_term(
                indices_in, coeffs_in, basis_index, basis_coeff, var_indices,
                global_var_idx)
        assert new_coeffs.shape[0] == new_indices.shape[1]
        assert new_indices.shape[1] == 21

        nvars = 4
        degree = 10 # degree needs to be high enough to be able to exactly
        # represent y3 which is the composition of lower degree polynomimals
        true_indices = compute_hyperbolic_indices(nvars, degree, 1)
        nsamples = true_indices.shape[1]*3
        samples = np.random.uniform(-1, 1, (nvars, nsamples))
        values1 = y1(samples)
        values2 = y2(samples)
        values = y3(np.vstack([samples, values1[None, :], values2[None, :]]))
        basis_mat = monomial_basis_matrix(true_indices, samples)
        true_coef = np.linalg.lstsq(basis_mat, values[:, None], rcond=None)[0]
        true_indices, true_coef = compress_and_sort_polynomial(
            true_coef, true_indices)
        new_indices, new_coeffs = compress_and_sort_polynomial(
            new_coeffs, new_indices)
        assert np.allclose(new_indices, true_indices)
        # print((true_coef, new_coeffs))
        assert np.allclose(true_coef, new_coeffs)

    def test_substitute_polynomials_for_variables_in_another_polynomial(self):
        """
        Substitute
          y1 = (1+x1+x2+x1*x2)
          y2 = (2+2*x1+2*x1*x3)
        into
          y3 = 1+y1+y2+x4+y1*y2+y2*x4+y1*y2*x4
        """
        def y1(x):
            x1, x2 = x[:2, :]
            return 1+x1+x2+x1*x2

        def y2(x):
            x1, x3 = x[[0, 2], :]
            return 2+2*x1+2*x1*x3

        def y3(x):
            x4 = x[3, :]
            y1, y2 = x[4:, :]
            return 1+y1+y2+x4+3*y1*y2+y2*x4+5*y1*y2*x4

        nvars = 4
        nsamples = 300
        degree = 5
        samples = np.random.uniform(-1, 1, (nvars, nsamples))
        values1 = y1(samples)
        values2 = y2(samples)
        values = y3(np.vstack([samples, values1[None, :], values2[None, :]]))

        true_indices = compute_hyperbolic_indices(nvars, degree, 1)
        basis_mat = monomial_basis_matrix(true_indices, samples)
        true_coef = np.linalg.lstsq(basis_mat, values[:, None], rcond=None)[0]

        validation_samples = np.random.uniform(-1, 1, (nvars, 1000))
        validation_values1 = y1(validation_samples)
        validation_values2 = y2(validation_samples)
        validation_values = y3(
            np.vstack([validation_samples, validation_values1[None, :],
                       validation_values2[None, :]]))

        validation_basis_mat = monomial_basis_matrix(
            true_indices, validation_samples)
        assert np.allclose(
            validation_values[:, None],
            validation_basis_mat.dot(true_coef), rtol=1e-12)

        global_var_idx = [np.array([0, 1]), np.array([0, 2])]
        indices_in = [np.array([[0, 0], [1, 0], [0, 1], [1, 1]]).T,
                      np.array([[0, 0], [1, 0], [1, 1]]).T]
        coeffs_in = [np.ones((indices_in[0].shape[1], 1)),
                     2*np.ones((indices_in[1].shape[1], 1))]
        var_idx = np.array([0, 1])  # must be related to how variables
        # enter indices below
        indices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
             [0, 1, 1], [1, 1, 1]]).T
        coeffs = np.ones((indices.shape[1], 1))
        coeffs[4] = 3
        coeffs[6] = 5
        new_indices, new_coef = \
            substitute_polynomials_for_variables_in_another_polynomial(
                indices_in, coeffs_in, indices, coeffs, var_idx,
                global_var_idx)

        true_indices, true_coef = compress_and_sort_polynomial(
            true_coef, true_indices)
        new_indices, new_coef = compress_and_sort_polynomial(
            new_coef, new_indices)
        assert np.allclose(new_indices, true_indices)
        # print(new_coef[:, 0])
        # print(true_coef)
        assert np.allclose(new_coef, true_coef)

    def test_substitute_polynomials_for_variables_in_another_polynomial_II(
            self):
        global_var_idx = [np.array([0, 1, 2, 3, 4, 5]), np.array([0, 1, 2])]
        indices_in = [
            compute_hyperbolic_indices(global_var_idx[0].shape[0], 2, 1),
            compute_hyperbolic_indices(global_var_idx[1].shape[0], 3, 1)]
        coeffs_in = [np.ones((indices_in[0].shape[1], 1)),
                     2*np.ones((indices_in[1].shape[1], 1))]
        var_idx = np.array([0, 1])  # must be related to how variables
        # enter indices below
        indices = compute_hyperbolic_indices(3, 5, 1)
        coeffs = np.ones((indices.shape[1], 1))
        new_indices, new_coef = \
            substitute_polynomials_for_variables_in_another_polynomial(
                indices_in, coeffs_in, indices, coeffs, var_idx,
                global_var_idx)

        nvars = np.unique(np.concatenate(global_var_idx)).shape[0] + (
            indices.shape[0] - var_idx.shape[0])
        validation_samples = np.random.uniform(-1, 1, (nvars, 1000))
        validation_values1 = monomial_basis_matrix(
            indices_in[0], validation_samples[global_var_idx[0], :]).dot(
                coeffs_in[0])
        validation_values2 = monomial_basis_matrix(
            indices_in[1], validation_samples[global_var_idx[1], :]).dot(
                coeffs_in[1])
        # inputs to polynomial which are not themselves polynomials
        other_global_var_idx = np.setdiff1d(
            np.arange(nvars), np.unique(np.concatenate(global_var_idx)))
        print(other_global_var_idx)
        validation_values = np.dot(
            monomial_basis_matrix(
                indices,
                np.vstack([validation_values1.T, validation_values2.T,
                           validation_samples[other_global_var_idx, :], ])),
            coeffs)
        assert np.allclose(validation_values, monomial_basis_matrix(
            new_indices, validation_samples).dot(new_coef))



if __name__ == "__main__":
    manipulate_polynomials_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(
            TestManipulatePolynomials)
    unittest.TextTestRunner(verbosity=2).run(manipulate_polynomials_test_suite)

"""
TODO
        Need example where
          y1 = (1+x1+x2+x1*x2)
          y2 = (2+2*x1+0*x2+2*x1*x2)
          y3 = (3+3*x1+3*x3)
        where y1 and y2 but not y3
        are functions of the same variables. The first example above
        arises when two separate scalar output models are inputs to a
        downstream model C, e.g.

        A(x1,x2) \
                  C
        A(x1,x2) /

        This second model arises when two outputs of the same vector output
        model A are inputs to a downstream model C as is one output of another
        upstream scalar output model B, e.g.

        A(x1,x2) \
                  C
        B(x1,x3) /

        In later case set
        indices_in = [np.array([[0,0],[1,0],[0,1],[1,1]]).T,
                      np.array([[0,0],[1,0],[0,1]]).T]
        coeffs_in = [np.ones((indices_in[0].shape[1],2)),
                     np.ones((indices_in[1].shape[1],1))]
        coeffs_in[0][:,1]*=2; coeffs_in[0][2,1]=0

        Actually may be better to treat all inputs as if from three seperate
        models
        indices_in = [np.array([[0,0],[1,0],[0,1],[1,1]]).T,
                      np.array([[0,0],[1,0],[0,1],[1,1]]).T
                      np.array([[0,0],[1,0],[0,1]]).T]
        coeffs_in = [np.ones((indices_in[0].shape[1],1)),
                     2*np.ones((indices_in[0].shape[1],1)),
                     3*np.ones((indices_in[1].shape[1],1))]
        coeffs_in[1][2,0]=0
"""
