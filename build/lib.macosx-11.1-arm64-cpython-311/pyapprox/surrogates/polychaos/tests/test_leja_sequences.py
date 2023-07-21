import unittest
from functools import partial
from scipy import stats
import numpy as np

from pyapprox.surrogates.polychaos.leja_sequences import (
    leja_objective_and_gradient, compute_finite_difference_derivative,
    leja_objective, compute_coefficients_of_leja_interpolant,
    get_leja_sequence_1d
)
from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices
from pyapprox.variables.transforms import (
    define_iid_random_variable_transformation
)
from pyapprox.variables.density import (
    beta_pdf_on_ab, beta_pdf_derivative
)
from pyapprox.surrogates.polychaos.gpc import (
    PolynomialChaosExpansion, define_poly_options_from_variable_transformation
)
from pyapprox.surrogates.interp.tensorprod import (
    evaluate_tensor_product_function, gradient_of_tensor_product_function
)


class TestLejaSequences(unittest.TestCase):

    def setup(self, num_vars, alpha_stat, beta_stat):

        def univariate_weight_function(x): return beta_pdf_on_ab(
            alpha_stat, beta_stat, -1, 1, x)
        def univariate_weight_function_deriv(x): return beta_pdf_derivative(
            alpha_stat, beta_stat, (x+1)/2)/4

        weight_function = partial(
            evaluate_tensor_product_function,
            [univariate_weight_function]*num_vars)

        weight_function_deriv = partial(
            gradient_of_tensor_product_function,
            [univariate_weight_function]*num_vars,
            [univariate_weight_function_deriv]*num_vars)

        assert np.allclose(
            (univariate_weight_function(0.5+1e-6) -
             univariate_weight_function(0.5))/1e-6,
            univariate_weight_function_deriv(0.5), atol=1e-6)

        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            stats.uniform(-2, 1), num_vars)
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)

        return weight_function, weight_function_deriv, poly

    def test_leja_objective_1d(self):
        num_vars = 1
        alpha_stat, beta_stat = [2, 2]
        # alpha_stat,beta_stat = [1,1]
        weight_function, weight_function_deriv, poly = self.setup(
            num_vars, alpha_stat, beta_stat)

        leja_sequence = np.array([[0.2, -1., 1.]])
        degree = leja_sequence.shape[1]-1
        indices = np.arange(degree+1)
        poly.set_indices(indices)
        new_indices = np.asarray([degree+1])

        coeffs = compute_coefficients_of_leja_interpolant(
            leja_sequence, poly, new_indices, weight_function)

        samples = np.linspace(-0.99, 0.99, 21)
        for sample in samples:
            sample = np.array([[sample]])
            func = partial(leja_objective, leja_sequence=leja_sequence,
                           poly=poly,
                           new_indices=new_indices, coeff=coeffs,
                           weight_function=weight_function,
                           weight_function_deriv=weight_function_deriv)
            fd_deriv = compute_finite_difference_derivative(
                func, sample, fd_eps=1e-8)

            residual, jacobian = leja_objective_and_gradient(
                sample, leja_sequence, poly, new_indices, coeffs,
                weight_function, weight_function_deriv, deriv_order=1)

            assert np.allclose(fd_deriv, np.dot(
                jacobian.T, residual), atol=1e-5)

    def test_leja_objective_2d(self):
        num_vars = 2
        alpha_stat, beta_stat = [2, 2]
        # alpha_stat,beta_stat = [1,1]

        weight_function, weight_function_deriv, poly = self.setup(
            num_vars, alpha_stat, beta_stat)

        leja_sequence = np.array([[-1.0, -1.0], [1.0, 1.0]]).T
        degree = 1
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        # sort lexographically to make testing easier
        II = np.lexsort((indices[0, :], indices[1, :], indices.sum(axis=0)))
        indices = indices[:, II]
        poly.set_indices(indices[:, :2])
        new_indices = indices[:, 2:3]

        coeffs = compute_coefficients_of_leja_interpolant(
            leja_sequence, poly, new_indices, weight_function)

        sample = np.asarray([0.5, -0.5])[:, np.newaxis]
        func = partial(leja_objective, leja_sequence=leja_sequence, poly=poly,
                       new_indices=new_indices, coeff=coeffs,
                       weight_function=weight_function,
                       weight_function_deriv=weight_function_deriv)
        fd_eps = 1e-7
        fd_deriv = compute_finite_difference_derivative(
            func, sample, fd_eps=fd_eps)

        residual, jacobian = leja_objective_and_gradient(
            sample, leja_sequence, poly, new_indices, coeffs,
            weight_function, weight_function_deriv, deriv_order=1)

        grad = np.dot(jacobian.T, residual)
        assert np.allclose(fd_deriv, grad, atol=fd_eps*100)

        # num_samples = 20
        # samples = np.linspace(-1, 1, num_samples)
        # samples = cartesian_product([samples]*num_vars)
        # objective_vals = func(samples)
        # f, ax = plt.subplots(1, 1, figsize=(8, 6))
        # X = samples[0, :].reshape(num_samples, num_samples)
        # Y = samples[1, :].reshape(num_samples, num_samples)
        # Z = objective_vals.reshape(num_samples, num_samples)
        # cset = ax.contourf(
        #     X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 30),
        #     cmap=None)
        # plt.colorbar(cset)
        # plt.plot(leja_sequence[0, :], leja_sequence[1, :], 'ko', ms=20)
        # plt.show()

    def test_optimize_leja_objective_1d(self):
        num_vars = 1
        num_leja_samples = 3
        # alpha_stat, beta_stat = 2, 2
        alpha_stat, beta_stat = 1, 1
        weight_function, weight_function_deriv, poly = self.setup(
            num_vars, alpha_stat, beta_stat)

        ranges = [-1, 1]
        # initial_points = np.asarray([[0.2, -1, 1]])
        initial_points = np.asarray([[0.]])

        # plt.clf()
        leja_sequence = get_leja_sequence_1d(
            num_leja_samples, initial_points, poly,
            weight_function, weight_function_deriv, ranges, plot=False)
        # print(leja_sequence)
        assert np.allclose(leja_sequence, [0, 1, -1])
        # plt.show()

    # def test_optimize_leja_objective_2d(self):
    #     num_vars = 2
    #     alpha_stat, beta_stat = [2, 2]
    #     weight_function, weight_function_deriv, poly = self.setup(
    #         num_vars, alpha_stat, beta_stat)

    #     leja_sequence = np.array([[-1.0, -1.0], [1.0, 1.0]]).T
    #     degree = 1
    #     indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
    #     # sort lexographically to make testing easier
    #     I = np.lexsort((indices[0, :], indices[1, :], indices.sum(axis=0)))
    #     indices = indices[:, I]
    #     poly.set_indices(indices[:, :2])
    #     new_indices = indices[:, 2:3]

    #     coeffs = compute_coefficients_of_leja_interpolant(
    #         leja_sequence, poly, new_indices, weight_function)

    #     obj = LejaObjective(poly, weight_function, weight_function_deriv)
    #     objective_args = (leja_sequence, new_indices, coeffs)
    #     ranges = [-1, 1, -1, 1]
    #     initial_guess = np.asarray([0.5, -0.5])[:, np.newaxis]
    #     #print((optimize(obj,initial_guess,ranges,objective_args) ))


if __name__ == "__main__":
    leja_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestLejaSequences)
    unittest.TextTestRunner(verbosity=2).run(leja_test_suite)
