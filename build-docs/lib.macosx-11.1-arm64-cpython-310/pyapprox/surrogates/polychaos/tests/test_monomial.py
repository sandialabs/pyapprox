import unittest
import numpy as np

from pyapprox.surrogates.interp.indexing import (
    compute_hyperbolic_indices, argsort_indices_leixographically
)
from pyapprox.surrogates.interp.monomial import (
    monomial_mean_uniform_variables, monomial_variance_uniform_variables,
    evaluate_monomial, monomial_basis_matrix, multiply_multivariate_polynomials
)


class TestMonomial(unittest.TestCase):

    def test_monomial_mean_uniform_variables(self):
        indices = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0]]).T
        coeffs = np.ones((indices.shape[1], 1))
        assert np.allclose(
            monomial_mean_uniform_variables(indices, coeffs), 4./3.)

    def test_monomial_variance_uniform_variables(self):
        num_vars = 2
        degree = 1
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        coeffs = np.ones((indices.shape[1], 1))
        squared_indices = compute_hyperbolic_indices(num_vars, 2*degree, 1.0)
        squared_coeffs = np.ones((squared_indices.shape[1], 1))
        true_variance = monomial_mean_uniform_variables(
            squared_indices, squared_coeffs)-monomial_mean_uniform_variables(
                indices, coeffs)**2
        assert np.allclose(
            monomial_variance_uniform_variables(indices, coeffs),
            true_variance)

    def test_evaluate_monomial(self):
        indices = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0]]).T
        coeffs = np.ones((indices.shape[1]))
        samples = np.array([[0.0, 0.0], [1.0, 1.0]]).T
        values = evaluate_monomial(indices, coeffs, samples)
        exact_values = np.array([1.0, 5.0])
        assert np.allclose(values[:, 0], exact_values)

    def test_basis_matrix(self):
        num_vars = 2
        num_samples = 10

        samples = np.random.uniform(-1, 1, (num_vars, num_samples))
        #samples = np.array([[0.5,0.5],[1.0,1.0]]).T
        indices = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0]]).T
        basis_matrix = monomial_basis_matrix(indices, samples)

        true_basis_matrix = np.hstack(
            (np.ones((num_samples, 1), dtype=float), samples.T))
        true_basis_matrix = np.hstack(
            (true_basis_matrix, (samples[0:1, :]*samples[1:2, :]).T))
        true_basis_matrix = np.hstack(
            (true_basis_matrix, samples[0:1, :].T**2))
        assert np.allclose(basis_matrix, true_basis_matrix)

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
        indices = indices[:, argsort_indices_leixographically(indices)]

        true_indices = compute_hyperbolic_indices(
            num_vars, degree1+degree2, 1.0)
        true_indices = \
            true_indices[:, argsort_indices_leixographically(true_indices)]
        assert np.allclose(true_indices, indices)
        true_coeffs = np.array([2, 4, 4, 4, 4, 6, 2, 4, 4, 2])
        assert np.allclose(true_coeffs[:, None], coeffs)

    def test_monomial_variance(self):
        pass


if __name__ == "__main__":
    monomial_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMonomial)
    unittest.TextTestRunner(verbosity=2).run(monomial_test_suite)
