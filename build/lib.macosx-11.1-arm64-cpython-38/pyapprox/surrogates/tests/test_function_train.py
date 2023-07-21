import unittest
import numpy as np
import copy
from functools import partial

from pyapprox.surrogates.function_train import (
    generate_additive_function_in_function_train_format,
    generate_homogeneous_function_train,
    function_trains_equal, evaluate_function_train_grad,
    ft_parameter_finite_difference_gradient,
    ft_linear_least_squares_regression, num_univariate_functions,
    ft_non_linear_least_squares_regression,
    generate_random_sparse_function_train,
    compress_homogeneous_function_train,
    evaluate_function_train, modify_and_evaluate_function_train,
    apply_function_train_adjoint_jacobian
)
from pyapprox.surrogates.orthopoly.orthonormal_polynomials import (
   evaluate_orthonormal_polynomial_1d
)
from pyapprox.surrogates.orthopoly.orthonormal_recursions import jacobi_recurrence
from pyapprox.optimization.l1_minimization import (
    s_sparse_projection, orthogonal_matching_pursuit,
    iterative_hard_thresholding
)


def additive_polynomial(samples, univariate_function_params, recursion_coeffs,
                        return_univariate_vals=False):
    num_vars, num_samples = samples.shape
    values = np.zeros((num_samples, 1), dtype=float)
    univariate_values = np.empty((num_samples, num_vars), dtype=float)
    for ii in range(num_vars):
        degree = univariate_function_params[ii].shape[0]-1
        basis_matrix = evaluate_orthonormal_polynomial_1d(
            samples[ii, :], degree, recursion_coeffs)
        univariate_values[:, ii] = np.dot(
            basis_matrix, univariate_function_params[ii])
        values += univariate_values[:, ii]
    if not return_univariate_vals:
        return values
    else:
        return values, univariate_values


class TestFunctionTrain(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_evaluate_function_train_additive_function(self):
        """
        Test the evaluation of a function train representation of an additive
        function.

        Assume same parameterization for each core and for each univariate
        function within a core.

        Use polynomial basis for each univariate function.
        """
        alpha = 0
        beta = 0
        degree = 2
        num_vars = 3
        num_samples = 1
        recursion_coeffs = jacobi_recurrence(
            degree+1, alpha=alpha, beta=beta, probability=True)

        univariate_function_params = [
            np.random.normal(0., 1., (degree+1))]*num_vars
        ft_data = generate_additive_function_in_function_train_format(
            univariate_function_params, True)

        samples = np.random.uniform(-1., 1., (num_vars, num_samples))
        values = evaluate_function_train(
            samples, ft_data, recursion_coeffs)

        true_values = additive_polynomial(
            samples, univariate_function_params, recursion_coeffs)

        assert np.allclose(values, true_values)

    def test_generate_homogeneous_function_train(self):
        num_vars = 3
        num_params_1d = 3

        # Generate function train representation of an additive function
        univariate_function_params = [
            np.random.normal(0., 1., (num_params_1d))]*num_vars
        ft_data_1 = generate_additive_function_in_function_train_format(
            univariate_function_params, False)

        ranks, ft_params = ft_data_1[:2]
        ft_data_2 = generate_homogeneous_function_train(
            ranks, num_params_1d, ft_params)

        assert function_trains_equal(ft_data_1, ft_data_2, True)

    def test_gradient_function_train_additive_function(self):
        """
        Test the gradient of a function train representation of an additive
        function. Gradient is with respect to coefficients of the univariate
        functions

        Assume different parameterization for some univariate functions.
        Zero and ones are stored as a constant basis. Where as other entries
        are stored as a polynomial of a fixed degree d.
        """
        alpha = 0
        beta = 0
        degree = 2
        num_vars = 3
        recursion_coeffs = jacobi_recurrence(
            degree+1, alpha=alpha, beta=beta, probability=True)

        univariate_function_params = [
            np.random.normal(0., 1., (degree+1))]*num_vars
        ft_data = generate_additive_function_in_function_train_format(
            univariate_function_params, True)

        sample = np.random.uniform(-1., 1., (num_vars, 1))
        value, ft_gradient = evaluate_function_train_grad(
            sample, ft_data, recursion_coeffs)

        true_values, univariate_values = additive_polynomial(
            sample, univariate_function_params, recursion_coeffs,
            return_univariate_vals=True)
        true_value = true_values[0, 0]
        assert np.allclose(value, true_value)

        true_gradient = np.empty((0), dtype=float)
        # var 0 univariate function 1,1
        basis_matrix_var_0 = evaluate_orthonormal_polynomial_1d(
            sample[0, :], degree, recursion_coeffs)
        true_gradient = np.append(true_gradient, basis_matrix_var_0)
        # var 0 univariate function 1,2
        true_gradient = np.append(
            true_gradient, np.sum(univariate_values[0, 1:]))

        basis_matrix_var_1 = evaluate_orthonormal_polynomial_1d(
            sample[1, :], degree, recursion_coeffs)
        # var 1 univariate function 1,1
        true_gradient = np.append(true_gradient, univariate_values[0, 0])
        # var 1 univariate function 2,1
        true_gradient = np.append(true_gradient, basis_matrix_var_1)
        # var 1 univariate function 1,2
        true_gradient = np.append(
            true_gradient, univariate_values[0, 0]*univariate_values[0, 2])
        # var 1 univariate function 2,2
        true_gradient = np.append(true_gradient, univariate_values[0, 2])

        basis_matrix_var_2 = evaluate_orthonormal_polynomial_1d(
            sample[2, :], degree, recursion_coeffs)
        # var 2 univariate function 1,1
        true_gradient = np.append(
            true_gradient, univariate_values[0, :2].sum())
        # var 2 univariate function 2,1
        true_gradient = np.append(true_gradient, basis_matrix_var_2)

        fd_gradient = ft_parameter_finite_difference_gradient(
            sample, ft_data, recursion_coeffs)

        # print 'true',true_gradient
        # print 'ft  ',ft_gradient
        # print fd_gradient
        assert np.allclose(fd_gradient, true_gradient)
        assert np.allclose(ft_gradient, true_gradient)

    def test_gradient_random_function_train(self):
        """
        Test the gradient of a random function train.
        Gradient is with respect to coefficients of the univariate functions

        Assume different parameterization for some univariate functions.
        Zero and ones are stored as a constant basis. Where as other entries
        are stored as a polynomial of a fixed degree d.
        """
        alpha = 0
        beta = 0
        degree = 2
        num_vars = 3
        rank = 2
        recursion_coeffs = jacobi_recurrence(
            degree+1, alpha=alpha, beta=beta, probability=True)

        ranks = np.ones((num_vars+1), dtype=int)
        ranks[1:-1] = rank
        num_params_1d = degree+1
        num_1d_functions = num_univariate_functions(ranks)
        ft_params = np.random.normal(0., 1., (num_params_1d*num_1d_functions))
        ft_data = generate_homogeneous_function_train(
            ranks, num_params_1d, ft_params)

        sample = np.random.uniform(-1., 1., (num_vars, 1))
        value, ft_gradient = evaluate_function_train_grad(
            sample, ft_data, recursion_coeffs)

        fd_gradient = ft_parameter_finite_difference_gradient(
            sample, ft_data, recursion_coeffs)

        assert np.allclose(fd_gradient, ft_gradient)

    def test_least_squares_regression(self):
        """
        Use non-linear least squares to estimate the coefficients of the
        function train approximation of a rank-2 bivariate function.
        """
        alpha = 0
        beta = 0
        degree = 5
        num_vars = 3
        rank = 2
        num_samples = 100
        recursion_coeffs = jacobi_recurrence(
            degree+1, alpha=alpha, beta=beta, probability=True)

        ranks = np.ones((num_vars+1), dtype=int)
        ranks[1:-1] = rank

        def function(samples): return np.cos(
            samples.sum(axis=0))[:, np.newaxis]

        samples = np.random.uniform(-1, 1, (num_vars, num_samples))
        values = function(samples)
        assert values.shape[0] == num_samples

        linear_ft_data = ft_linear_least_squares_regression(
            samples, values, degree, perturb=None)

        initial_guess = linear_ft_data[1].copy()

        # test jacobian
        # residual_func = partial(
        #    least_squares_residual,samples,values,linear_ft_data,
        #    recursion_coeffs)
        #
        # jacobian = least_squares_jacobian(
        #     samples,values,linear_ft_data,recursion_coeffs,initial_guess)
        # finite difference is expensive check on subset of points
        # for ii in range(2):
        #    func = lambda x: residual_func(x)[ii]
        #    assert np.allclose(
        #        scipy.optimize.approx_fprime(initial_guess, func, 1e-7),
        #        jacobian[ii,:])

        lstsq_ft_params = ft_non_linear_least_squares_regression(
            samples, values, linear_ft_data, recursion_coeffs, initial_guess)
        lstsq_ft_data = copy.deepcopy(linear_ft_data)
        lstsq_ft_data[1] = lstsq_ft_params

        num_valid_samples = 100
        validation_samples = np.random.uniform(
            -1., 1., (num_vars, num_valid_samples))
        validation_values = function(validation_samples)

        ft_validation_values = evaluate_function_train(
            validation_samples, lstsq_ft_data, recursion_coeffs)
        ft_error = np.linalg.norm(
            validation_values-ft_validation_values)/np.sqrt(num_valid_samples)
        assert ft_error < 1e-3, ft_error

        # compare against tensor-product linear least squares
        from pyapprox.surrogates.interp.monomial import monomial_basis_matrix, evaluate_monomial
        from pyapprox.surrogates.interp.indexing import tensor_product_indices
        indices = tensor_product_indices([degree]*num_vars)
        basis_matrix = monomial_basis_matrix(indices, samples)
        coef = np.linalg.lstsq(basis_matrix, values, rcond=None)[0]
        monomial_validation_values = evaluate_monomial(
            indices, coef, validation_samples)
        monomial_error = np.linalg.norm(
            validation_values-monomial_validation_values)/np.sqrt(
                num_valid_samples)
        assert ft_error < monomial_error

    def test_restricted_least_squares_regression_additive_function(self):
        """
        Use non-linear least squares to estimate the coefficients of the
        function train approximation of a rank-2 bivariate function,
        optimizing only over a subset of the FT parameters.
        """
        alpha = 0
        beta = 0
        degree = 5
        num_vars = 3
        rank = 2
        num_samples = 20
        recursion_coeffs = jacobi_recurrence(
            degree+1, alpha=alpha, beta=beta, probability=True)

        ranks = np.ones((num_vars+1), dtype=int)
        ranks[1:-1] = rank
        num_params_1d = degree+1

        univariate_function_params = [
            np.random.normal(0., 1., (degree+1))]*num_vars
        ft_data = generate_additive_function_in_function_train_format(
            univariate_function_params, False)

        def function(samples): return evaluate_function_train(
            samples, ft_data, recursion_coeffs)

        samples = np.random.uniform(-1, 1, (num_vars, num_samples))
        values = function(samples)
        assert values.shape[0] == num_samples

        linear_ft_data = ft_linear_least_squares_regression(
            samples, values, degree, perturb=None)

        initial_guess = linear_ft_data[1].copy()

        active_indices = []
        active_indices += list(range(num_params_1d))
        active_indices += list(range(3*num_params_1d, 4*num_params_1d))
        active_indices += list(range(7*num_params_1d, 8*num_params_1d))
        active_indices = np.asarray(active_indices)
        # active_indices = np.where((ft_data[1]!=0)&(ft_data[1]!=1))[0]
        initial_guess = initial_guess[active_indices]

        lstsq_ft_params = ft_non_linear_least_squares_regression(
            samples, values, linear_ft_data, recursion_coeffs, initial_guess,
            active_indices=active_indices)
        lstsq_ft_data = copy.deepcopy(linear_ft_data)
        lstsq_ft_data[1] = lstsq_ft_params

        num_valid_samples = 100
        validation_samples = np.random.uniform(
            -1., 1., (num_vars, num_valid_samples))
        validation_values = function(validation_samples)

        ft_validation_values = evaluate_function_train(
            validation_samples, lstsq_ft_data, recursion_coeffs)
        ft_error = np.linalg.norm(
            validation_values-ft_validation_values)/np.sqrt(num_valid_samples)
        assert ft_error < 1e-3, ft_error

    def test_restricted_least_squares_regression_sparse(self):
        """
        Use non-linear least squares to estimate the coefficients of the
        function train approximation of a rank-2 bivariate function,
        optimizing only over a subset of the FT parameters.
        """
        alpha = 0
        beta = 0
        degree = 5
        num_vars = 3
        rank = 2
        num_samples = 20
        sparsity_ratio = 0.2
        recursion_coeffs = jacobi_recurrence(
            degree+1, alpha=alpha, beta=beta, probability=True)

        ranks = np.ones((num_vars+1), dtype=int)
        ranks[1:-1] = rank

        ft_data = generate_random_sparse_function_train(
            num_vars, rank, degree+1, sparsity_ratio)

        def function(samples): return evaluate_function_train(
            samples, ft_data, recursion_coeffs)

        samples = np.random.uniform(-1, 1, (num_vars, num_samples))
        values = function(samples)
        assert values.shape[0] == num_samples

        active_indices = np.where((ft_data[1] != 0) & (ft_data[1] != 1))[0]

        linear_ft_data = ft_linear_least_squares_regression(
            samples, values, degree, perturb=None)
        initial_guess = linear_ft_data[1].copy()
        initial_guess = initial_guess[active_indices]
        # initial_guess = true_sol[active_indices] + np.random.normal(
        #    0.,1.,(active_indices.shape[0]))

        lstsq_ft_params = ft_non_linear_least_squares_regression(
            samples, values, ft_data, recursion_coeffs, initial_guess,
            active_indices=active_indices)
        lstsq_ft_data = copy.deepcopy(ft_data)
        lstsq_ft_data[1] = lstsq_ft_params

        num_valid_samples = 100
        validation_samples = np.random.uniform(
            -1., 1., (num_vars, num_valid_samples))
        validation_values = function(validation_samples)

        ft_validation_values = evaluate_function_train(
            validation_samples, lstsq_ft_data, recursion_coeffs)
        ft_error = np.linalg.norm(
            validation_values-ft_validation_values)/np.sqrt(num_valid_samples)
        # print ft_error
        assert ft_error < 1e-3, ft_error

    def test_compress_homogeneous_function_train(self):
        degree = 2
        num_vars = 3

        univariate_function_params = [
            np.random.normal(0., 1., (degree+1))]*num_vars
        ft_data = generate_additive_function_in_function_train_format(
            univariate_function_params, False)

        true_compressed_ft_data = \
            generate_additive_function_in_function_train_format(
                univariate_function_params, True)

        compressed_ft_data = compress_homogeneous_function_train(
            ft_data, tol=0)

        assert compressed_ft_data[1].shape[0] < ft_data[1].shape[0]

        assert function_trains_equal(
            compressed_ft_data, true_compressed_ft_data, verbose=True)

    def test_random_function_train(self):
        np.random.seed(5)
        num_vars = 2
        degree = 5
        rank = 2

        sparsity_ratio = 0.2
        sample_ratio = .9

        ranks = rank*np.ones(num_vars+1, dtype=np.uint64)
        ranks[0] = 1
        ranks[-1] = 1

        alpha = 0
        beta = 0
        recursion_coeffs = jacobi_recurrence(
            degree+1, alpha=alpha, beta=beta, probability=True)

        ft_data = generate_random_sparse_function_train(
            num_vars, rank, degree+1, sparsity_ratio)
        true_sol = ft_data[1]

        num_ft_params = true_sol.shape[0]
        num_samples = int(sample_ratio*num_ft_params)
        samples = np.random.uniform(-1., 1., (num_vars, num_samples))

        def function(samples): return evaluate_function_train(
            samples, ft_data, recursion_coeffs)

        values = function(samples)

        assert np.linalg.norm(values) > 0, (np.linalg.norm(values))

        num_validation_samples = 100
        validation_samples = np.random.uniform(
            -1., 1., (num_vars, num_validation_samples))
        validation_values = function(validation_samples)

        zero_ft_data = copy.deepcopy(ft_data)
        zero_ft_data[1] = np.zeros_like(zero_ft_data[1])
        # DO NOT use ft_data in following two functions.
        # These function only overwrites parameters associated with the
        # active indices the rest of the parameters are taken from ft_data.
        # If ft_data is used some of the true data will be kept and give
        # an unrealisticaly accurate answer
        approx_eval = partial(modify_and_evaluate_function_train, samples,
                              zero_ft_data, recursion_coeffs, None)

        apply_approx_adjoint_jacobian = partial(
            apply_function_train_adjoint_jacobian, samples, zero_ft_data,
            recursion_coeffs, 1e-3)

        sparsity = np.where(true_sol != 0)[0].shape[0]
        print(('sparsity', sparsity, 'num_samples', num_samples))

        # sparse
        project = partial(s_sparse_projection, sparsity=sparsity)
        # non-linear least squres
        #project = partial(s_sparse_projection,sparsity=num_ft_params)

        # use uninormative initial guess
        #initial_guess = np.zeros_like(true_sol)

        # use linear approximation as initial guess
        linear_ft_data = ft_linear_least_squares_regression(
            samples, values, degree, perturb=None)
        initial_guess = linear_ft_data[1]

        # use initial guess that is close to true solution
        # num_samples required to obtain accruate answer decreases signficantly
        # over linear or uniformative guesses. As size of perturbation from
        # truth increases num_samples must increase
        initial_guess = true_sol.copy()+np.random.normal(0., .1, (num_ft_params))

        tol = 5e-3
        max_iter = 1000
        result = iterative_hard_thresholding(
            approx_eval, apply_approx_adjoint_jacobian, project,
            values[:, 0], initial_guess, tol, max_iter, verbosity=1)
        sol = result[0]
        residnorm = result[1]

        recovered_ft_data = copy.deepcopy(ft_data)
        recovered_ft_data[1] = sol
        ft_validation_values = evaluate_function_train(
            validation_samples, recovered_ft_data, recursion_coeffs)

        validation_error = np.linalg.norm(
            validation_values-ft_validation_values)
        rel_validation_error = validation_error / \
            np.linalg.norm(validation_values)
        # compare relative error because exit condition is based upon
        # relative residual
        assert rel_validation_error < 10*tol, rel_validation_error

        # interestingly enough the error in the function can be low
        # but the error in the ft parameters can be large
        # assert np.allclose(true_sol,sol,atol=10*tol)

    def test_sparse_function_train(self):
        np.random.seed(5)
        num_vars = 2
        degree = 5
        rank = 2

        tol = 1e-5

        sparsity_ratio = 0.2
        sample_ratio = 0.6

        ranks = rank*np.ones(num_vars+1, dtype=np.uint64)
        ranks[0] = 1
        ranks[-1] = 1

        alpha = 0
        beta = 0
        recursion_coeffs = jacobi_recurrence(
            degree+1, alpha=alpha, beta=beta, probability=True)

        ft_data = generate_random_sparse_function_train(
            num_vars, rank, degree+1, sparsity_ratio)
        true_sol = ft_data[1]

        num_ft_params = true_sol.shape[0]
        num_samples = int(sample_ratio*num_ft_params)
        samples = np.random.uniform(-1., 1., (num_vars, num_samples))

        def function(samples): return evaluate_function_train(
            samples, ft_data, recursion_coeffs)
        #function = lambda samples: np.cos(samples.sum(axis=0))[:,np.newaxis]

        values = function(samples)
        print(values.shape)

        assert np.linalg.norm(values) > 0, (np.linalg.norm(values))

        num_validation_samples = 100
        validation_samples = np.random.uniform(
            -1., 1., (num_vars, num_validation_samples))
        validation_values = function(validation_samples)

        zero_ft_data = copy.deepcopy(ft_data)
        zero_ft_data[1] = np.zeros_like(zero_ft_data[1])
        # DO NOT use ft_data in following two functions.
        # These function only overwrites parameters associated with the
        # active indices the rest of the parameters are taken from ft_data.
        # If ft_data is used some of the true data will be kept and give
        # an unrealisticaly accurate answer
        approx_eval = partial(modify_and_evaluate_function_train, samples,
                              zero_ft_data, recursion_coeffs, None)

        apply_approx_adjoint_jacobian = partial(
            apply_function_train_adjoint_jacobian, samples, zero_ft_data,
            recursion_coeffs, 1e-3)

        def least_squares_regression(indices, initial_guess):
            # if initial_guess is None:
            st0 = np.random.get_state()
            np.random.seed(1)
            initial_guess = np.random.normal(0., .01, indices.shape[0])
            np.random.set_state(st0)
            result = ft_non_linear_least_squares_regression(
                samples, values, ft_data, recursion_coeffs, initial_guess,
                indices, {'gtol': tol, 'ftol': tol, 'xtol': tol, 'verbosity': 0})
            return result[indices]

        sparsity = np.where(true_sol != 0)[0].shape[0]
        print(('sparsity', sparsity, 'num_samples', num_samples,
               'num_ft_params', num_ft_params))

        print(true_sol)
        active_indices = None

        use_omp = True
        #use_omp = False
        if not use_omp:
            sol = least_squares_regression(np.arange(num_ft_params), None)
        else:
            result = orthogonal_matching_pursuit(
                approx_eval, apply_approx_adjoint_jacobian,
                least_squares_regression, values[:, 0], active_indices,
                num_ft_params, tol, min(num_samples, num_ft_params), verbosity=1)
            sol = result[0]
            residnorm = result[1]

        recovered_ft_data = copy.deepcopy(ft_data)
        recovered_ft_data[1] = sol
        ft_validation_values = evaluate_function_train(
            validation_samples, recovered_ft_data, recursion_coeffs)

        validation_error = np.linalg.norm(
            validation_values-ft_validation_values)
        rel_validation_error = validation_error / \
            np.linalg.norm(validation_values)
        # compare relative error because exit condition is based upon
        # relative residual
        print(rel_validation_error)
        assert rel_validation_error < 100*tol, rel_validation_error

        # interestingly enough the error in the function can be low
        # but the error in the ft parameters can be large
        # print np.where(true_sol!=0)[0]
        # print np.where(sol!=0)[0]
        # assert np.allclose(true_sol,sol,atol=100*tol)



if __name__ == "__main__":
    function_train_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestFunctionTrain)
    unittest.TextTestRunner(verbosity=2).run(function_train_test_suite)
