import unittest
from matplotlib import pyplot as plt
from functools import partial
from pyapprox.leja_sequences import *
from pyapprox.utilities import cartesian_product, beta_pdf_derivative
from pyapprox.indexing import compute_hyperbolic_indices
from pyapprox.variable_transformations import \
    define_iid_random_variable_transformation
from scipy.stats import beta, uniform
from scipy import stats
from scipy.special import beta as beta_fn
from pyapprox.utilities import beta_pdf_on_ab, beta_pdf, beta_pdf_derivative,\
    gaussian_pdf, gaussian_pdf_derivative
from pyapprox.univariate_leja import *
from pyapprox.optimization import check_gradients
from pyapprox.orthonormal_polynomials_1d import *


class TestLeja1DSequences(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_sqrt_christoffel_inv_gradients(self):
        degree = 2
        ab = jacobi_recurrence(degree+1, 0, 0, True)
        basis_fun = partial(
            evaluate_orthonormal_polynomial_1d, nmax=degree, ab=ab)
        basis_fun_and_jac = partial(
            evaluate_orthonormal_polynomial_deriv_1d, nmax=degree, ab=ab,
            deriv_order=1)
        sample = np.random.uniform(-1, 1, (1, 1))
        #sample = np.atleast_2d(-0.99)

        fun = partial(sqrt_christoffel_function_inv_1d, basis_fun)
        jac = partial(sqrt_christoffel_function_inv_jac_1d, basis_fun_and_jac)

        #xx = np.linspace(-1, 1, 101); plt.plot(xx, fun(xx[None, :]));
        #plt.plot(sample[0], fun(sample), 'o'); plt.show()

        err = check_gradients(fun, jac, sample)
        assert err.max() > .5 and err.min() < 1e-7

        basis_fun_jac_hess = partial(
            evaluate_orthonormal_polynomial_deriv_1d, nmax=degree, ab=ab,
            deriv_order=2)
        hess = partial(
            sqrt_christoffel_function_inv_hess_1d, basis_fun_jac_hess,
            normalize=False)
        err = check_gradients(jac, hess, sample)
        assert err.max() > .5 and err.min() < 1e-7

    def test_christoffel_inv_gradients(self):
        degree = 2
        ab = jacobi_recurrence(degree+1, 0, 0, True)
        basis_fun = partial(
            evaluate_orthonormal_polynomial_1d, nmax=degree, ab=ab)
        basis_fun_and_jac = partial(
            evaluate_orthonormal_polynomial_deriv_1d, nmax=degree, ab=ab,
            deriv_order=1)
        sample = np.random.uniform(-1, 1, (1, 1))
        #sample = np.atleast_2d(-0.99)

        fun = partial(christoffel_function_inv_1d, basis_fun)
        jac = partial(christoffel_function_inv_jac_1d, basis_fun_and_jac)

        #xx = np.linspace(-1, 1, 101); plt.plot(xx, fun(xx[None, :]));
        #plt.plot(sample[0], fun(sample), 'o'); plt.show()

        err = check_gradients(fun, jac, sample)
        assert err.max() > .5 and err.min() < 1e-7

        basis_fun_jac_hess = partial(
            evaluate_orthonormal_polynomial_deriv_1d, nmax=degree, ab=ab,
            deriv_order=2)
        hess = partial(
            christoffel_function_inv_hess_1d, basis_fun_jac_hess,
            normalize=False)
        err = check_gradients(jac, hess, sample)
        assert err.max() > .5 and err.min() < 1e-7

    def test_christoffel_leja_objective_gradients(self):
        #leja_sequence = np.array([[-1, 1]])
        leja_sequence = np.array([[-1, 0, 1]])
        degree = leja_sequence.shape[1]-1
        ab = jacobi_recurrence(degree+2, 0, 0, True)
        basis_fun = partial(
            evaluate_orthonormal_polynomial_1d, nmax=degree+1, ab=ab)
        tmp = basis_fun(leja_sequence[0, :])
        nterms = degree+1
        basis_mat = tmp[:, :nterms]
        new_basis = tmp[:, nterms:]
        coef = compute_coefficients_of_christoffel_leja_interpolant_1d(
            basis_mat, new_basis)

        fun = partial(christoffel_leja_objective_fun_1d, basis_fun, coef)

        #xx = np.linspace(-1, 1, 101); plt.plot(xx, fun(xx[None, :]));
        #plt.plot(leja_sequence[0, :], fun(leja_sequence), 'o'); plt.show()

        basis_fun_and_jac = partial(
            evaluate_orthonormal_polynomial_deriv_1d, nmax=degree+1, ab=ab,
            deriv_order=1)
        jac = partial(
            christoffel_leja_objective_jac_1d, basis_fun_and_jac, coef)

        sample = sample = np.random.uniform(-1, 1, (1, 1))
        err = check_gradients(fun, jac, sample)
        assert err.max() > 0.5 and err.min() < 1e-7

        basis_fun_jac_hess = partial(
            evaluate_orthonormal_polynomial_deriv_1d, nmax=degree+1, ab=ab,
            deriv_order=2)
        hess = partial(
            christoffel_leja_objective_hess_1d, basis_fun_jac_hess, coef)
        err = check_gradients(jac, hess, sample)
        assert err.max() > .5 and err.min() < 1e-7

    def test_pdf_weighted_leja_objective_gradients(self):
        #leja_sequence = np.array([[-1, 1]])
        leja_sequence = np.array([[-1, 0, 1]])
        degree = leja_sequence.shape[1]-1
        ab = jacobi_recurrence(degree+2, 0, 0, True)
        basis_fun = partial(
            evaluate_orthonormal_polynomial_1d, nmax=degree+1, ab=ab)

        def pdf(x):
            return beta_pdf(1, 1, (x+1)/2)/2

        def pdf_jac(x):
            return beta_pdf_derivative(1, 1, (x+1)/2)/4

        tmp = basis_fun(leja_sequence[0, :])
        nterms = degree+1
        basis_mat = tmp[:, :nterms]
        new_basis = tmp[:, nterms:]
        coef = compute_coefficients_of_pdf_weighted_leja_interpolant_1d(
            pdf(leja_sequence[0, :]), basis_mat, new_basis)

        fun = partial(pdf_weighted_leja_objective_fun_1d, pdf, basis_fun, coef)

        #xx = np.linspace(-1, 1, 101); plt.plot(xx, fun(xx[None, :]));
        #plt.plot(leja_sequence[0, :], fun(leja_sequence), 'o'); plt.show()

        basis_fun_and_jac = partial(
            evaluate_orthonormal_polynomial_deriv_1d, nmax=degree+1, ab=ab,
            deriv_order=1)
        jac = partial(
            pdf_weighted_leja_objective_jac_1d, pdf, pdf_jac,
            basis_fun_and_jac, coef)

        sample = sample = np.random.uniform(-1, 1, (1, 1))
        err = check_gradients(fun, jac, sample)
        assert err.max() > 0.4 and err.min() < 1e-7

        # hessian not currently used in optimization. To activate
        # need to compute the 2nd derivative of the pdf of each supported
        # variable

        # basis_fun_jac_hess = partial(
        #     evaluate_orthonormal_polynomial_deriv_1d, nmax=degree+1, ab=ab,
        #     deriv_order=2)
        # hess = partial(
        #     pdf_weighted_leja_objective_hess_1d, pdf, pdf_jac, pdf_hess,
        #     basis_fun_jac_hess, coef)
        # err = check_gradients(jac, hess, sample)
        # assert err.max() > .5 and err.min() < 1e-7

    def test_uniform_christoffel_leja_sequence_1d(self):
        max_nsamples = 3
        initial_points = np.array([[0]])
        ab = jacobi_recurrence(max_nsamples+1, 0, 0, True)
        basis_fun = partial(
            evaluate_orthonormal_polynomial_deriv_1d, ab=ab)

        def callback(leja_sequence, coef, new_samples, obj_vals,
                     initial_guesses):
            degree = coef.shape[0]-1

            def plot_fun(x):
                return -christoffel_leja_objective_fun_1d(
                    partial(basis_fun, nmax=degree+1, deriv_order=0), coef,
                    x[None, :])
            xx = np.linspace(-1, 1, 101)
            plt.plot(xx, plot_fun(xx))
            plt.plot(leja_sequence[0, :], plot_fun(leja_sequence[0, :]), 'o')
            plt.plot(new_samples[0, :], obj_vals, 's')
            plt.plot(
                initial_guesses[0, :], plot_fun(initial_guesses[0, :]), '*')
            plt.show()

        leja_sequence = get_christoffel_leja_sequence_1d(
            max_nsamples, initial_points, [-1, 1], basis_fun,
            {'gtol': 1e-8, 'verbose': False}, callback=None)

        from pyapprox.univariate_quadrature import leja_growth_rule
        level = 3
        __basis_fun = partial(basis_fun, nmax=max_nsamples-1, deriv_order=0)
        weights = get_christoffel_leja_quadrature_weights_1d(
            leja_sequence, leja_growth_rule, __basis_fun, level, True)
        assert np.allclose((leja_sequence**2).dot(weights[-1]), 1/3)

    def test_hermite_christoffel_leja_sequence_1d(self):
        import warnings
        warnings.filterwarnings('error')
        # for unbounded variables can get overflow warnings because when
        # optimizing interval with one side unbounded and no local minima
        # exists then optimization will move towards inifinity
        max_nsamples = 20
        initial_points = np.array([[0, stats.norm(0, 1).ppf(0.75)]])
        #initial_points = np.array([[stats.norm(0, 1).ppf(0.75)]])
        ab = hermite_recurrence(max_nsamples+1, 0, False)
        basis_fun = partial(
            evaluate_orthonormal_polynomial_deriv_1d, ab=ab)

        plot_degree = np.inf  # max_nsamples-1
        # assert plot_degree < max_nsamples

        def callback(leja_sequence, coef, new_samples, obj_vals,
                     initial_guesses):
            degree = coef.shape[0]-1
            new_basis_degree = degree+1
            if new_basis_degree != plot_degree:
                return
            plt.clf()

            def plot_fun(x):
                return -christoffel_leja_objective_fun_1d(
                    partial(basis_fun, nmax=new_basis_degree, deriv_order=0),
                    coef, x[None, :])
            xx = np.linspace(-20, 20, 1001)
            plt.plot(xx, plot_fun(xx))
            plt.plot(leja_sequence[0, :], plot_fun(leja_sequence[0, :]), 'o')
            I = np.argmin(obj_vals)
            plt.plot(new_samples[0, I], obj_vals[I], 's')
            plt.plot(
                initial_guesses[0, :], plot_fun(initial_guesses[0, :]), '*')
            #plt.xlim(-10, 10)
            print('s', new_samples[0], obj_vals)

        leja_sequence = get_christoffel_leja_sequence_1d(
            max_nsamples, initial_points, [-np.inf, np.inf], basis_fun,
            {'gtol': 1e-8, 'verbose': False, 'iprint': 2}, callback)

        # compare to lu based leja samples
        # given the same set of initial samples the next sample chosen
        # should be close with the one from the gradient based method having
        # slightly better objective value
        num_candidate_samples = 10001

        def generate_candidate_samples(n):
            return np.linspace(-20, 20, n)[None, :]

        from pyapprox.univariate_leja import sqrt_christoffel_function_inv_1d
        from pyapprox import truncated_pivoted_lu_factorization
        for ii in range(initial_points.shape[1], max_nsamples):
            degree = ii-1
            new_basis_degree = degree+1
            candidate_samples = generate_candidate_samples(
                num_candidate_samples)
            candidate_samples = np.hstack(
                [leja_sequence[:, :ii], candidate_samples])
            bfun = partial(basis_fun, nmax=new_basis_degree, deriv_order=0)
            basis_mat = bfun(candidate_samples[0, :])
            basis_mat = sqrt_christoffel_function_inv_1d(
                bfun, candidate_samples)[:, None]*basis_mat
            LU, pivots = truncated_pivoted_lu_factorization(
                basis_mat, ii+1, ii, False)
            # cannot use get_candidate_based_leja_sequence_1d
            # because it uses christoffel function that is for maximum
            # degree
            discrete_leja_sequence = candidate_samples[:, pivots[:ii+1]]

            # if new_basis_degree == plot_degree:
            #     # mulitply by LU[ii, ii]**2 to account for LU factorization
            #     # dividing the column by this number
            #     discrete_obj_vals = -LU[:, ii]**2*LU[ii, ii]**2
            #     # account for pivoting of ith column of LU factor
            #     # value of best objective can be found in the iith pivot
            #     discrete_obj_vals[ii] = discrete_obj_vals[pivots[ii]]
            #     discrete_obj_vals[pivots[ii]] = -LU[ii, ii]**2
            #     I = np.argsort(candidate_samples[0, ii:])+ii
            #     plt.plot(candidate_samples[0, I], discrete_obj_vals[I], '--')
            #     plt.plot(candidate_samples[0, pivots[ii]],
            #              -LU[ii, ii]**2, 'k^')
            #     plt.show()

            def objective_value(sequence):
                tmp = bfun(sequence[0, :])
                basis_mat = tmp[:, :-1]
                new_basis = tmp[:, -1:]
                coef = compute_coefficients_of_christoffel_leja_interpolant_1d(
                    basis_mat, new_basis)
                return christoffel_leja_objective_fun_1d(
                    bfun, coef, sequence[:, -1:])

            discrete_obj_val = objective_value(
                discrete_leja_sequence[:, :ii+1])
            obj_val = objective_value(leja_sequence[:, :ii+1])

            diff = candidate_samples[0, -1]-candidate_samples[0, -2]
            #print(ii, obj_val - discrete_obj_val)
            print(leja_sequence[:, :ii+1], discrete_leja_sequence)
            #assert obj_val >= discrete_obj_val
            # obj_val will not always be greater than because of optimization
            # tolerance and discretization of candidate samples
            # assert abs(obj_val - discrete_obj_val) < 1e-4
            assert (abs(leja_sequence[0, ii] - discrete_leja_sequence[0, -1]) <
                    diff)
        # plt.show()

    def test_hermite_pdf_weighted_leja_sequence_1d(self):
        max_nsamples = 3
        initial_points = np.array([[0]])
        ab = hermite_recurrence(max_nsamples+1, True)
        basis_fun = partial(
            evaluate_orthonormal_polynomial_deriv_1d, ab=ab)

        pdf = partial(gaussian_pdf, 0, 1)
        pdf_jac = partial(gaussian_pdf_derivative, 0, 1)

        def callback(leja_sequence, coef, new_samples, obj_vals,
                     initial_guesses):
            degree = coef.shape[0]-1

            def plot_fun(x):
                return -pdf_weighted_leja_objective_fun_1d(
                    pdf, partial(basis_fun, nmax=degree +
                                 1, deriv_order=0), coef,
                    x[None, :])
            xx = np.linspace(-10, 10, 101)
            plt.plot(xx, plot_fun(xx))
            plt.plot(leja_sequence[0, :], plot_fun(leja_sequence[0, :]), 'o')
            plt.plot(new_samples[0, :], obj_vals, 's')
            plt.plot(
                initial_guesses[0, :], plot_fun(initial_guesses[0, :]), '*')
            plt.show()

        leja_sequence = get_pdf_weighted_leja_sequence_1d(
            max_nsamples, initial_points, [-np.inf, np.inf], basis_fun,
            pdf, pdf_jac,
            {'gtol': 1e-8, 'verbose': False}, callback=None)

        # compare to lu based leja samples
        # given the same set of initial samples the next sample chosen
        # should be close with the one from the gradient based method having
        # slightly better objective value
        num_candidate_samples = 1001

        def generate_candidate_samples(n):
            return np.linspace(-5, 5, n)[None, :]

        discrete_leja_sequence = \
            get_candidate_based_pdf_weighted_leja_sequence_1d(
                max_nsamples, ab, generate_candidate_samples,
                num_candidate_samples, pdf,
                initial_points=leja_sequence[:, :max_nsamples-1])

        degree = max_nsamples-1
        tmp = basis_fun(
            discrete_leja_sequence[0, :-1], nmax=degree+1,  deriv_order=0)
        basis_mat = tmp[:, :-1]
        new_basis = tmp[:, -1:]
        coef = compute_coefficients_of_pdf_weighted_leja_interpolant_1d(
            pdf(discrete_leja_sequence[0, :-1]), basis_mat, new_basis)
        discrete_obj_val = pdf_weighted_leja_objective_fun_1d(
            pdf, partial(basis_fun, nmax=degree+1, deriv_order=0), coef,
            discrete_leja_sequence[:, -1:])

        tmp = basis_fun(
            leja_sequence[0, :-1], nmax=degree+1,  deriv_order=0)
        basis_mat = tmp[:, :-1]
        new_basis = tmp[:, -1:]
        coef = compute_coefficients_of_pdf_weighted_leja_interpolant_1d(
            pdf(leja_sequence[0, :-1]), basis_mat, new_basis)
        obj_val = pdf_weighted_leja_objective_fun_1d(
            pdf, partial(basis_fun, nmax=degree+1, deriv_order=0), coef,
            leja_sequence[:, -1:])

        x = generate_candidate_samples(num_candidate_samples)
        assert obj_val > discrete_obj_val
        assert (abs(leja_sequence[0, -1] - discrete_leja_sequence[0, -1]) <
                x[0, 1]-x[0, 0])


class TestLejaSequences(unittest.TestCase):

    def setup(self, num_vars, alpha_stat, beta_stat):

        # univariate_weight_function=lambda x: beta(alpha_stat,beta_stat).pdf(
        #    (x+1)/2)/2
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
            uniform(-2, 1), num_vars)
        poly_opts = {'alpha_poly': beta_stat-1, 'beta_poly': alpha_stat-1,
                     'var_trans': var_trans, 'poly_type': 'jacobi'}
        poly.configure(poly_opts)

        return weight_function, weight_function_deriv, poly

    def test_leja_objective_1d(self):
        num_vars = 1
        alpha_stat, beta_stat = [2, 2]
        #alpha_stat,beta_stat = [1,1]
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
            func = partial(leja_objective, leja_sequence=leja_sequence, poly=poly,
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
        #alpha_stat,beta_stat = [1,1]

        weight_function, weight_function_deriv, poly = self.setup(
            num_vars, alpha_stat, beta_stat)

        leja_sequence = np.array([[-1.0, -1.0], [1.0, 1.0]]).T
        degree = 1
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        # sort lexographically to make testing easier
        I = np.lexsort((indices[0, :], indices[1, :], indices.sum(axis=0)))
        indices = indices[:, I]
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

        num_samples = 20
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

        plt.clf()
        leja_sequence = get_leja_sequence_1d(
            num_leja_samples, initial_points, poly,
            weight_function, weight_function_deriv, ranges, plot=False)
        print(leja_sequence)
        assert np.allclose(leja_sequence, [0, 1, -1])
        # plt.show()

    def test_optimize_leja_objective_2d(self):
        num_vars = 2
        alpha_stat, beta_stat = [2, 2]
        weight_function, weight_function_deriv, poly = self.setup(
            num_vars, alpha_stat, beta_stat)

        leja_sequence = np.array([[-1.0, -1.0], [1.0, 1.0]]).T
        degree = 1
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        # sort lexographically to make testing easier
        I = np.lexsort((indices[0, :], indices[1, :], indices.sum(axis=0)))
        indices = indices[:, I]
        poly.set_indices(indices[:, :2])
        new_indices = indices[:, 2:3]

        coeffs = compute_coefficients_of_leja_interpolant(
            leja_sequence, poly, new_indices, weight_function)

        obj = LejaObjective(poly, weight_function, weight_function_deriv)
        objective_args = (leja_sequence, new_indices, coeffs)
        ranges = [-1, 1, -1, 1]
        initial_guess = np.asarray([0.5, -0.5])[:, np.newaxis]
        #print((optimize(obj,initial_guess,ranges,objective_args) ))


if __name__ == "__main__":
    leja1d_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestLeja1DSequences)
    unittest.TextTestRunner(verbosity=2).run(leja1d_test_suite)

    leja_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestLejaSequences)
    unittest.TextTestRunner(verbosity=2).run(leja_test_suite)
